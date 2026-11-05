"""Regression coverage for the full WebUI run's stale/duplicated judge scores."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lmms_eval.cli import judge_cmd
from lmms_eval.llm_judge.aggregator import Aggregator
from lmms_eval.llm_judge.standalone import JudgeRunner


@pytest.mark.parametrize("task", [
    "simplevqa", "mmmu_val_qwen3_official", "mmmu_pro_standard_reasoning_qwen3_official",
    "mmbench_en_dev", "sfe-en",
])
def test_native_aggregation_receives_final_metrics(task):
    aggregator = Aggregator()
    config = aggregator._get_special_config(task)
    assert config is not None
    key = config["data_key"]
    native = MagicMock(return_value=50.0)
    module = SimpleNamespace(**{config["accuracy_func"]: native})
    samples = [{key: {"state": "pending"}, "metrics": {key: {"state": "judged"}}}]
    with patch("lmms_eval.llm_judge.aggregator.importlib.import_module", return_value=module):
        assert aggregator.aggregate(samples, task) == {"accuracy": 50.0}
    native.assert_called_once_with([{"state": "judged"}])


def test_simplevqa_real_native_aggregate_uses_completed_judgments():
    samples = [
        {"simplevqa_judge": {"judge_score": None, "judge_success": False},
         "metrics": {"simplevqa_judge": {"judge_score": score, "judge_success": True}}}
        for score in [1, 0]
    ]
    assert Aggregator().aggregate(samples, "simplevqa") == {"accuracy": 50.0}


def test_native_aggregation_cannot_silently_drop_a_sample():
    with pytest.raises(ValueError, match="simplevqa_judge"):
        Aggregator().aggregate([
            {"metrics": {"simplevqa_judge": {"judge_score": 1, "judge_success": True}}},
            {"metrics": {}},
        ], "simplevqa")


def test_final_flat_scores_are_counted_once():
    samples = [{"judge_mode": "llm_judge", "metrics": {
        "exact_match": score, "llm_judge_score": score, "llm_judge_success": True,
    }} for score in [1, 0, 0, 1]]
    assert JudgeRunner().compute_summary(samples)["total_acc"] == 0.5


def test_sfe_rule_scored_samples_are_not_overwritten_by_llm_judge():
    runner = JudgeRunner(judge_mode="judge")
    assert not runner._needs_llm_judge({"sfe_info": {"question_type": "open_ended"},
                                       "exact_match": 0.0, "needs_llm_judge": False})
    assert runner._needs_llm_judge({"sfe_info": {"question_type": "mcq"},
                                   "exact_match": 0.0, "needs_llm_judge": True})


def test_sfe_bbox_uses_iou_even_when_question_type_is_open_ended():
    info = {"id": "E011/1", "question_type": "open_ended", "iou_score": [0.8],
            "rouge_score": [-1], "llm_score": [-1], "bertscore": [-1],
            "bleu_score": [-1], "meteor_score": [-1], "execute_success_rate": [1]}
    assert Aggregator().aggregate([{"metrics": {"sfe_info": info}}], "sfe-en")["exact_match"] == 0.8


@pytest.mark.parametrize("rating", [2.9, 2.4, 8.9, 0.0, 10.0])
def test_sfe_judge_sample_score_agrees_with_native_integer_rating(rating):
    """Replay decimal judge outputs that disagreed with SFE's saved integer rating."""
    runner = JudgeRunner(judge_mode="judge")
    runner._current_task_name = "sfe-en"
    runner._judge_provider = MagicMock()
    runner._judge_provider.evaluate_score.return_value = {
        "result": rating, "raw_response": str(rating), "model": "fixture", "success": True,
    }
    info = {"id": "C006/0010", "question_type": "exact_match", "iou_score": [-1],
            "rouge_score": [-1], "llm_score": [-1], "bertscore": [-1],
            "bleu_score": [-1], "meteor_score": [-1], "execute_success_rate": [-1]}
    original = {"exact_match": 0.0, "needs_llm_judge": True, "sfe_info": info}
    judged = runner._apply_llm_judge(
        doc={"question": "Estimate the value to one decimal place.", "answer": "2.2"},
        results=["2.9"], fallback_metrics=original,
    )
    native = Aggregator().aggregate([{"metrics": judged}], "sfe-en")
    assert judged["exact_match"] == pytest.approx(native["exact_match"])
    assert judged["llm_judge_score"] == pytest.approx(native["exact_match"] * 10)
    assert judged["llm_judge_raw"] == str(rating)
    assert original["sfe_info"]["llm_score"] == [-1]


def _run_args(source, output):
    return argparse.Namespace(input_result=str(source), task="fixture", output=None,
        output_dir=str(output), judge_mode="llm", judge_model="fixture", judge_api_key="fixture",
        judge_base_url="http://localhost/v1", parallel=1, dry_run=False, verbose=False)


@pytest.mark.parametrize("merged", [False, True])
def test_aggregation_failure_retains_samples_and_exits_nonzero(tmp_path, merged):
    source = tmp_path / "input"
    source.mkdir()
    inputs = [source / f"{i}_samples_fixture.jsonl" for i in range(2 if merged else 1)]
    for item in inputs:
        item.write_text('{"doc_id": 0}\n')
    output = tmp_path / "judged"
    runner = MagicMock()
    runner.judge_file.return_value = [{"doc_id": 0, "metrics": {"exact_match": 1}}]
    runner.compute_summary.return_value = {"total_acc": 1.0}
    aggregate = MagicMock()
    aggregate.aggregate.side_effect = ([{}, {}, ValueError("native aggregation failed")]
                                      if merged else ValueError("native aggregation failed"))
    with patch.object(judge_cmd, "JudgeRunner", return_value=runner), \
         patch.object(judge_cmd, "Aggregator", return_value=aggregate), \
         patch.object(judge_cmd, "_expand_group_tasks", side_effect=lambda tasks: tasks), \
         patch.object(judge_cmd, "_make_table", return_value="table"):
        with pytest.raises(SystemExit) as error:
            judge_cmd.run_judge(_run_args(str(source / "*_samples_fixture.jsonl") if merged else inputs[0], output))
    assert error.value.code == 1
    assert runner.save_results.call_count == len(inputs)
    if not merged:
        assert not list(output.glob("*_results.json"))


def test_native_task_score_takes_priority_over_generic_sample_mean(tmp_path):
    """MMBench circular accuracy is not its per-row exact match mean."""
    source = tmp_path / "samples_fixture.jsonl"
    source.write_text('{"doc_id": 0}\n')
    runner = MagicMock()
    runner.judge_file.return_value = [{"metrics": {"exact_match": 1}}]
    runner.compute_summary.return_value = {"total_acc": 0.97, "rule_acc": 0.97}
    aggregate = MagicMock()
    aggregate.aggregate.return_value = {"accuracy": 85.0}
    with patch.object(judge_cmd, "JudgeRunner", return_value=runner), \
         patch.object(judge_cmd, "Aggregator", return_value=aggregate), \
         patch.object(judge_cmd, "_write_judge_results_json") as write, \
         patch.object(judge_cmd, "_expand_group_tasks", side_effect=lambda tasks: tasks), \
         patch.object(judge_cmd, "_make_table", return_value="table"):
        judge_cmd.run_judge(_run_args(source, tmp_path / "judged"))
    summary = write.call_args.kwargs["summary"]
    assert summary["accuracy"] == 85.0
    assert "total_acc" not in summary


def test_real_mmbench_native_circular_accuracy_is_published(tmp_path, monkeypatch):
    from lmms_eval.tasks.mmbench import en_utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(en_utils.mmbench_evaluator, "get_chat_response",
                        MagicMock(side_effect=AssertionError("No API request expected for explicit choices")))
    samples = []
    for index, prediction in [(1, "A"), (1000001, "B"), (2, "A")]:
        metric = {"index": index, "answer": "A", "prediction": prediction,
                  "question": "Choose A", "hint": "", "source": "fixture", "split": "dev",
                  "category": "fixture", "L2-category": "fixture", "A": "yes", "B": "no",
                  "C": "nan", "D": "nan", "E": "nan"}
        samples.append({"metrics": {"gpt_eval_score": metric}})
    # 2/3 rows match, but only 1/2 questions pass all circular variants.
    judge_cmd._load_judge_runtime_objects()
    summary = judge_cmd._summarize_judged_results(JudgeRunner(), samples, "mmbench_en_dev")
    assert summary["accuracy"] == 50.0
    assert "total_acc" not in summary
    en_utils.mmbench_evaluator.get_chat_response.assert_not_called()


@pytest.mark.parametrize("score", [-1.0, float("nan"), float("inf")])
def test_pending_or_nonfinite_native_score_is_failure(score):
    runner = MagicMock()
    runner.compute_summary.return_value = {"total_acc": 1.0}
    with patch.object(judge_cmd, "Aggregator") as factory:
        factory.return_value.aggregate.return_value = {"accuracy": score}
        with pytest.raises(ValueError, match="accuracy"):
            judge_cmd._summarize_judged_results(runner, [{}], "fixture")
