"""Replay the DLC judge artifact contract through the WebUI API."""

import json
import math

import pytest
from fastapi.testclient import TestClient

from lmms_eval.tui import server


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_retry_job_is_classified_as_judge():
    assert server._job_stage_from_name("eval_judge_retry_dlceygc22c70nk7g_bc5b41da") == "judge"
    assert server._job_stage_from_name("judge_eval_qwen35_2026-10-06_05-21-41") == "judge"
    assert server._job_stage_from_name("eval_qwen35_9b_judge_test") == "eval"


def test_judge_runtime_does_not_append_eval_timestamp(tmp_path):
    output = tmp_path / "results" / "judge"
    output.mkdir(parents=True)
    config = write_json(tmp_path / "judge_runtime_config.json", {
        "judge": {"model": "Qwen3.5-9B"},
        "eval": {"output_path": str(output), "timestamp": "2026-10-06_05-21-41"},
    })
    assert server._job_runtime_paths({"UserCommand": f"bash run_judge.sh {config}"})[1] == output


@pytest.mark.parametrize("option", ["--output-dir", "--output_dir", "--output_path"])
def test_judge_cli_output_directory(tmp_path, option):
    assert server._job_runtime_paths({"UserCommand": f"lmms-eval judge -i input.jsonl {option} '{tmp_path}'"})[1] == tmp_path


@pytest.mark.parametrize("task,metrics,expected", [
    ("mmbench_en_dev", {"total_acc": 0.9691, "accuracy": 84.27835051546391}, ("accuracy", 84.27835051546391)),
    ("simplevqa", {"total_acc": 0.6946, "exact_match": 0.4474}, ("exact_match", 0.4474)),
    ("ocrbench", {"llm_judge_success": 1.0, "total_acc": 0.858}, ("total_acc", 0.858)),
])
def test_native_scores_precede_legacy_generic_totals(tmp_path, task, metrics, expected):
    write_json(tmp_path / "run_results.json", {"results": {task: metrics}, "judge": {"failed_samples": 0}})
    row = server._build_metric_rows(tmp_path)[0][0]
    assert (row.metric_name, row.value) == expected


@pytest.mark.parametrize("metrics,judge", [
    ({"total_acc": 0.6694, "accuracy": -1.0}, {}),
    ({"accuracy": math.nan}, {}),
    ({"accuracy": math.inf}, {}),
    ({"accuracy": 80, "llm_judge_failed": 0.1}, {}),
    ({"accuracy": 80, "llm_judge_success": 0.9}, {}),
    ({"accuracy": 80}, {"failed_samples": 1}),
])
def test_invalid_judge_aggregate_does_not_publish_score(tmp_path, metrics, judge):
    write_json(tmp_path / "run_results.json", {
        "results": {"mmmu_val_qwen3_official": metrics}, "judge": {"mode": "judge", **judge},
    })
    row = server._build_metric_rows(tmp_path)[0][0]
    assert row.status == "judge_failed"
    assert row.value is None


@pytest.mark.parametrize("sample,expected_status", [
    ({"judge_mode": "llm_judge", "metrics": {"llm_judge_score": 1, "llm_judge_success": True}}, "pending_judge"),
    ({"judge_mode": "error", "metrics": {"llm_judge_success": False}}, "judge_failed"),
])
def test_samples_without_native_aggregate_are_inspectable_but_not_final(tmp_path, sample, expected_status):
    samples = tmp_path / "run_samples_mmbench_en_dev.jsonl"
    samples.write_text(json.dumps(sample) + "\n")
    row = server._build_metric_rows(tmp_path)[0][0]
    assert row.status == expected_status
    assert row.value is None
    assert row.sample_jsonls == [str(samples)]


def test_one_failed_judgment_is_not_rounded_away_in_a_large_file(tmp_path):
    sample = {"judge_mode": "llm_judge", "metrics": {"llm_judge_score": 1, "llm_judge_success": True}}
    samples = tmp_path / "run_samples_ocrbench.jsonl"
    samples.write_text((json.dumps(sample) + "\n") * 20000 + json.dumps({"judge_mode": "error"}) + "\n")
    row = server._build_metric_rows(tmp_path)[0][0]
    assert row.status == "judge_failed"
    assert row.n_samples == 20001


def test_judge_results_api_uses_declared_sample_file(tmp_path, monkeypatch):
    """Custom --output and default *_judged.jsonl names must stay clickable."""
    task = "mmbench_en_dev"
    samples = tmp_path / "judgments.jsonl"
    samples.write_text(json.dumps({"doc_id": 0, "target": "B", "filtered_resps": ["A"],
                                   "judge_mode": "llm_judge", "metrics": {"llm_judge_score": 0}}) + "\n")
    result = write_json(tmp_path / "run_results.json", {
        "results": {task: {"accuracy": 84.27835051546391, "total_acc": 0.9691}},
        "n-samples": {task: {"original": 1, "effective": 1}},
        "judge": {"model": "Qwen3.5-9B", "samples_file": samples.name, "failed_samples": 0},
    })
    monkeypatch.setattr(server, "_get_dlc_job_detail", lambda _: {
        "DisplayName": "eval_judge_retry_dlcsource_abcd1234", "UserCommand": f"lmms-eval judge --output-dir {tmp_path}",
    })
    session, _ = server._create_auth_session({"username": "test", "role": "user", "access_key_id": "fake", "secret_access_key": "fake"})
    with TestClient(server.app) as client:
        client.cookies.set(server.AUTH_COOKIE_NAME, session)
        response = client.get("/dlc/jobs/dlctest/metrics")
        assert response.status_code == 200, response.text
        row, = response.json()["metrics"]
        assert row["result_json"] == str(result)
        assert row["sample_jsonls"] == [str(samples)]
        assert row["judge_model"] == "Qwen3.5-9B"
        assert row["value"] == 84.27835051546391
        response = client.get(f"/dlc/jobs/dlctest/metrics/{row['metric_id']}/samples?sample_filter=wrong")
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1


def test_declared_sample_cannot_escape_result_directory(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"doc_id": 0}\n')
    root = tmp_path / "judge"
    write_json(root / "run_results.json", {"results": {"ocrbench": {"total_acc": 1}},
                                          "judge": {"samples_file": "../outside.jsonl"}})
    assert server._build_metric_rows(root)[0][0].sample_jsonls == []
