import pytest
import json

from lmms_eval.tui import server


@pytest.mark.parametrize("empty_value", [None, "", [], {}])
def test_primary_metrics_skips_empty_values(empty_value):
    result_data = {
        "results": {
            "vstar_bench": {
                "alias": "vstar_bench",
                "vstar_direct_attributes_acc,none": empty_value,
                "vstar_direct_attributes_acc_stderr,none": [],
                "vstar_overall_acc,none": 78.53403141361257,
                "vstar_overall_acc_stderr,none": "N/A",
            }
        }
    }

    assert server._primary_metrics(result_data) == [
        ("vstar_bench", "vstar_overall_acc", 78.53403141361257, "N/A")
    ]


@pytest.mark.parametrize("valid_value", [0, 0.0, False])
def test_primary_metrics_keeps_falsey_scores(valid_value):
    result_data = {
        "results": {
            "zero_score_bench": {
                "alias": "zero_score_bench",
                "accuracy,none": valid_value,
                "fallback_accuracy,none": 99.0,
            }
        }
    }

    assert server._primary_metrics(result_data) == [
        ("zero_score_bench", "accuracy", valid_value, None)
    ]


@pytest.mark.parametrize("stage", [None, "generation"])
def test_unjudged_scores_are_hidden_even_for_legacy_runs(tmp_path, stage):
    data = {"results": {"ocrbench": {"accuracy,none": 0.8}}}
    if stage:
        data["execution_stage"] = stage
    path = tmp_path / "run_results.json"
    path.write_text(json.dumps(data))
    row = server._result_json_metric_rows(idx_start=0, result_json=path)[0]
    assert row.status == "pending_judge"
    assert row.value is None
    assert row.stderr is None


def test_summary_does_not_hide_new_benchmarks_or_judge_results(tmp_path):
    first = tmp_path / "first_results.json"
    first.write_text(json.dumps({"results": {"ocrbench": {"accuracy": 0.1}}}))
    (tmp_path / "summary.jsonl").write_text(json.dumps({"result_json": str(first)}) + "\n")
    (tmp_path / "second_results.json").write_text(json.dumps({"results": {"ai2d": {"accuracy": 0.9}}}))
    (tmp_path / "judge_results.json").write_text(json.dumps({
        "results": {"ocrbench": {"llm_judge_success": 1.0, "accuracy": 0.7}},
        "judge": {"failed_samples": 0},
    }))
    rows, _ = server._build_metric_rows(tmp_path)
    assert {row.lmms_tasks: row.value for row in rows} == {"ai2d": 0.9, "ocrbench": 0.7}
    assert all(row.status == "success" for row in rows)


def test_judge_transport_success_is_not_a_score():
    assert server._preferred_judged_sample_metric({"llm_judge_success": 1.0}) == ""


@pytest.mark.parametrize("task,category,overall", [
    ("mmstar", "coarse perception", "average"),
    ("seedbench_2_plus", "seedbench_2_plus_Chart", "seedbench_2_plus_all"),
])
def test_task_overall_score_precedes_first_category(task, category, overall):
    data = {"results": {task: {f"{category},none": 0.8, f"{overall},none": 0.7,
                               f"{overall}_stderr,none": 0.01}}}
    assert server._primary_metrics(data) == [(task, overall, 0.7, 0.01)]
