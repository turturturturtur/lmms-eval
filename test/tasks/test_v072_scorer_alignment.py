from pathlib import Path


def test_ocrbench_hme_preserves_case_but_regular_text_does_not():
    from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results

    hme_doc = {
        "answer": ["p"],
        "dataset": "HME100k",
        "question_type": "Handwritten Mathematical Expression Recognition",
    }
    assert ocrbench_process_results(hme_doc, ["P"])["ocrbench_accuracy"]["score"] == 0
    assert ocrbench_process_results(hme_doc, ["p"])["ocrbench_accuracy"]["score"] == 1

    regular_doc = {
        "answer": ["CENTRE"],
        "dataset": "IIIT5K",
        "question_type": "Regular Text Recognition",
    }
    assert ocrbench_process_results(regular_doc, ["centre"])["ocrbench_accuracy"]["score"] == 1


def test_shared_mcq_extractor_prefers_explicit_final_answer():
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    response = "Option A is tempting. After checking the image, the final answer is (B)."
    assert extract_mcq_answer(response, choices=["A", "B", "C", "D"]) == "B"


def test_mmstar_accepts_reasoning_with_final_answer():
    from lmms_eval.tasks.mmstar.utils import exact_match

    assert exact_match("I considered A and C. The answer is B.", "B") == 1.0


def test_realworldqa_uses_mcq_extraction_and_keeps_open_answer_exact():
    from lmms_eval.tasks.realworldqa.utils import realworldqa_process_results

    assert realworldqa_process_results({"answer": "C"}, ["The correct answer is (C)."])["exact_match"] == 1.0
    assert realworldqa_process_results({"answer": "Downhill"}, ["Downhill."])["exact_match"] == 1.0


def test_embspatial_reports_source_metrics_without_changing_total_entry():
    from lmms_eval.tasks.embspatial.utils import embspatial_process_results

    result = embspatial_process_results(
        {
            "answer": 1,
            "answer_options": ["left", "right", "front", "behind"],
            "question_id": "mp3d_1",
            "relation": "right",
            "data_source": "mp3d",
        },
        ["B"],
    )
    assert set(result) == {
        "embspatial_acc",
        "ai2thor_accuracy",
        "mp3d_accuracy",
        "scannet_accuracy",
    }
    assert result["embspatial_acc"]["is_correct"] is True
    assert result["embspatial_acc"]["data_source"] == "mp3d"


def test_mmbench_static_task_and_aggregator_exist():
    from lmms_eval.tasks.mmbench import en_utils

    task_root = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "mmbench"
    yaml_text = (task_root / "mmbench_en_dev_static.yaml").read_text(encoding="utf-8")
    assert "mmbench_aggregate_dev_results_static" in yaml_text
    assert callable(en_utils.mmbench_aggregate_dev_results_static)
