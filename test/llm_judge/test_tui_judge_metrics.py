"""Tests for WebUI metrics parsing of standalone judge outputs."""

import json

from lmms_eval.tui.server import _build_metric_rows, _result_status_for_root


def test_build_metric_rows_from_judged_samples_only(tmp_path):
    """Historical judge dirs with only judged samples should expose real judge metrics."""
    result_root = tmp_path / "judge"
    result_root.mkdir()
    sample_file = result_root / "20260701_031944_samples_ocrbench.jsonl"
    rows = [
        {
            "doc_id": 0,
            "metrics": {
                "ocrbench_accuracy": {
                    "score": 1,
                    "llm_judge_score": 1,
                    "llm_judge_success": True,
                },
                "llm_judge_score": 1,
                "llm_judge_success": True,
                "llm_judge_failed": False,
            },
            "judge_mode": "llm_judge",
        },
        {
            "doc_id": 1,
            "metrics": {
                "ocrbench_accuracy": {
                    "score": 0,
                    "llm_judge_score": 0,
                    "llm_judge_success": True,
                },
                "llm_judge_score": 0,
                "llm_judge_success": True,
                "llm_judge_failed": False,
            },
            "judge_mode": "llm_judge",
        },
    ]
    sample_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    metric_rows, summary_files = _build_metric_rows(result_root)

    assert summary_files == []
    assert _result_status_for_root(result_root) == "has_results"
    assert len(metric_rows) == 1
    row = metric_rows[0]
    assert row.display_name == "ocrbench"
    assert row.metric_name == "ocrbench_accuracy.score"
    assert row.value == 0.5
    assert row.n_samples == 2
    assert row.sample_jsonls == [str(sample_file)]
    assert row.value_source == "judged_samples"


def test_build_metric_rows_ignores_plain_samples_only(tmp_path):
    """Plain eval samples without judge metrics should not become pseudo metrics."""
    result_root = tmp_path / "plain"
    result_root.mkdir()
    sample_file = result_root / "20260701_031944_samples_plain_task.jsonl"
    sample_file.write_text(json.dumps({"doc_id": 0, "accuracy": 1}) + "\n", encoding="utf-8")

    metric_rows, _summary_files = _build_metric_rows(result_root)

    assert metric_rows == []
    assert _result_status_for_root(result_root) == "empty"
