import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image
import yaml


UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmms_eval"
    / "tasks"
    / "superchem_official"
    / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("superchem_official_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(utils)


class _YamlLoader(yaml.SafeLoader):
    pass


_YamlLoader.add_multi_constructor(
    "!",
    lambda loader, tag_suffix, node: loader.construct_scalar(node),
)


def _base_doc():
    return {
        "uuid": "question-1",
        "question_en": "Which option is correct?",
        "options_en": {"A": "Alpha", "B": "Beta", "C": None},
        "answer_en": ["B"],
        "question_type": "multiple_choice",
        "question_images": {},
        "options_images": {},
    }


def test_official_prompt_matches_upstream_template():
    assert utils.doc_to_text_multimodal(_base_doc()) == (
        "Question:\n\n"
        "Which option is correct?\n\n"
        "A: Alpha\n"
        "B: Beta\n\n\n"
        "First conduct step-by-step reasoning, then finally provide the answer in a "
        "$\\boxed{<answer>}$ box, like $\\boxed{A}. Note that The box should contain "
        "the single character of choice only."
    )


def test_text_only_mode_replaces_image_link_with_alt_text():
    doc = _base_doc()
    doc["question_en"] = (
        "Inspect <MultiModal>![reaction scheme](scheme.png)</MultiModal> carefully."
    )

    prompt = utils.doc_to_text_text_only(doc)

    assert (
        "<MultiModal>reaction scheme</MultiModal>"
        in prompt
    )
    assert "scheme.png" not in prompt


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_multimodal_mode_interleaves_images_in_official_tag_order():
    doc = _base_doc()
    doc["question_en"] = "Q <MultiModal>![question image](q.png)</MultiModal>"
    doc["options_en"]["B"] = (
        "B image <MultiModal>![option image](b.png)</MultiModal>"
    )
    doc["question_images"] = {"q.png": _png_bytes("red")}
    doc["options_images"] = {"b.png": _png_bytes("blue")}

    prompt = utils.doc_to_text_multimodal(doc)
    visuals = utils.doc_to_visual_multimodal(doc)

    assert prompt.count("<image>") == 2
    assert "<MultiModal>" not in prompt
    assert [image.getpixel((0, 0)) for image in visuals] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_official_parser_uses_last_box_and_has_no_unboxed_fallback():
    assert utils.parse_official_answer(
        r"First guess \boxed{A}; corrected result \boxed{\textbf{B}}."
    ) == "B"

    with pytest.raises(ValueError, match=r"boxed"):
        utils.parse_official_answer("The answer is B.")


def test_process_results_reports_official_eight_attempt_metrics():
    responses = [
        r"reasoning \boxed{B}",
        r"reasoning \boxed{A}",
        "B without the required box",
        r"reasoning \boxed{B}",
        r"reasoning \boxed{A}",
        r"reasoning \boxed{A}",
        r"reasoning \boxed{B}",
        r"reasoning \boxed{A}",
    ]

    assert utils.process_results(_base_doc(), responses) == {
        "superchem_official_pass1_first": 1,
        "superchem_official_mean_reliability": 3 / 8,
        "superchem_official_pass8": 1,
        "superchem_official_valid_rate": 7 / 8,
    }


def test_official_fill_blank_and_question_type_validation():
    doc = _base_doc()
    doc["answer_en"] = ["H2O"]
    doc["question_type"] = "fill_blank"
    assert utils.score_official_answer(doc, "h2o") == 1
    assert utils.score_official_answer(doc, "CO2") == 0

    doc["question_type"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported question type"):
        utils.score_official_answer(doc, "A")


def test_official_yaml_exposes_both_modes_and_protocol_defaults():
    task_dir = UTILS_PATH.parent
    group = yaml.load(
        (task_dir / "superchem_official.yaml").read_text(),
        Loader=_YamlLoader,
    )
    text = yaml.load(
        (task_dir / "superchem_official_text.yaml").read_text(),
        Loader=_YamlLoader,
    )
    multimodal = yaml.load(
        (task_dir / "superchem_official_multimodal.yaml").read_text(),
        Loader=_YamlLoader,
    )

    assert group["group"] == "superchem_official"
    assert group["task"] == [
        "superchem_official_text",
        "superchem_official_multimodal",
    ]
    for config in (text, multimodal):
        assert config["repeats"] == 8
        assert config["generation_kwargs"] == {
            "until": [],
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": 32768,
            "reasoning_effort": "high",
            "stream": True,
            "stream_options": {"include_usage": True},
            "official_request": "superchem",
            "request_timeout": 600,
            "request_max_retries": 5,
            "request_concurrency": 10,
        }
