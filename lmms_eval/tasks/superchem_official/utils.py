import io
import re

from PIL import Image


OFFICIAL_PROMPT_TEMPLATE = """Question:

{question}

{options}

First conduct step-by-step reasoning, then finally provide the answer in a $\\boxed{{<answer>}}$ box, like $\\boxed{{A}}. Note that The box should contain the single character of choice only."""


def get_options_string(doc):
    options = doc["options_en"]
    rendered = ""
    for key in sorted(options):
        value = options[key]
        if value is None:
            break
        rendered += f"{key}: {value}\n"
    return rendered


def _render_official_prompt(doc):
    return OFFICIAL_PROMPT_TEMPLATE.format(
        question=doc["question_en"],
        options=get_options_string(doc),
    )


def doc_to_text_multimodal(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return re.sub(
        r"<MultiModal>(.*?)</MultiModal>",
        "<image>",
        _render_official_prompt(doc),
        flags=re.IGNORECASE | re.DOTALL,
    )


def doc_to_text_text_only(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    prompt = _render_official_prompt(doc)
    tag_pattern = r"<MultiModal>(.*?)</MultiModal>"

    def replace_image_link(match):
        link_match = re.match(
            r"!?\s*\[(.+)\]\s*\([^)]+\)",
            match.group(1),
            flags=re.MULTILINE | re.DOTALL,
        )
        if link_match:
            return f"<MultiModal>{link_match.group(1)}</MultiModal>"
        return match.group(0)

    return re.sub(
        tag_pattern,
        replace_image_link,
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )


def doc_to_visual_multimodal(doc):
    prompt = _render_official_prompt(doc)
    images = {
        **(doc.get("question_images") or {}),
        **(doc.get("options_images") or {}),
    }
    visuals = []
    for tag_match in re.finditer(
        r"<MultiModal>(.*?)</MultiModal>",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        tag_content = tag_match.group(1)
        link_match = re.match(
            r"!?\s*\[(.+)\]\s*\(([^)]+)\)",
            tag_content,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not link_match:
            raise ValueError(f"No link found in multimodal tag: {tag_content}")

        image_url = link_match.group(2)
        image_bytes = images.get(image_url)
        if image_bytes is None:
            raise FileNotFoundError(f"Image file not found: {image_url}")
        if not (
            image_bytes.startswith(b"\xff\xd8\xff")
            or image_bytes.startswith(b"\x89PNG")
        ):
            raise ValueError(f"Unsupported image format for {image_url}")

        visuals.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return visuals


def doc_to_visual_text(doc, kwargs=None):
    del doc, kwargs
    return []


def doc_to_target(doc, kwargs=None):
    del kwargs
    return {
        "ref": doc["answer_en"],
        "type": doc["question_type"],
    }


def parse_official_answer(response):
    matches = list(re.finditer(r"\\boxed\{(.+?)\}", response, re.DOTALL))
    if not matches:
        raise ValueError(
            "Could not find the answer in the required $\\boxed{...}$ format."
        )
    answer = matches[-1].group(1).strip()
    return (
        answer.replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("text", "")
        .replace("math", "")
        .replace("bf", "")
        .replace("rm", "")
        .strip()
    )


def score_official_answer(doc, answer):
    ref_answer = doc["answer_en"]
    question_type = doc["question_type"]
    if question_type == "multiple_choice":
        return int(
            len(answer) == len(ref_answer)
            and all(character.upper() in ref_answer for character in answer)
        )
    if question_type == "fill_blank":
        return int(
            bool(ref_answer)
            and answer.lower() == str(ref_answer[0]).lower()
        )
    raise ValueError(f"Unsupported question type: {question_type}")


def process_results(doc, results):
    if (
        len(results) == 1
        and isinstance(results[0], (list, tuple))
    ):
        results = list(results[0])
    if not results:
        raise ValueError("SUPERChem official scoring requires at least one response.")

    scores = []
    valid_count = 0
    for response in results:
        try:
            answer = parse_official_answer(str(response))
        except ValueError:
            scores.append(0)
            continue
        valid_count += 1
        scores.append(score_official_answer(doc, answer))

    sample_count = len(scores)
    return {
        "superchem_official_pass1_first": scores[0],
        "superchem_official_mean_reliability": sum(scores) / sample_count,
        "superchem_official_pass8": int(any(scores)),
        "superchem_official_valid_rate": valid_count / sample_count,
    }
