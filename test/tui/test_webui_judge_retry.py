from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmms_eval.tui import server


@pytest.fixture
def retry_client(monkeypatch):
    session, _ = server._create_auth_session({"username": "owner", "role": "user", "aliyun_user_id": "owner-id",
                                             "access_key_id": "fake-key", "secret_access_key": "fake-secret"})
    detail = {"DisplayName": "eval_original", "Status": "Stopped", "UserId": "owner-id"}
    monkeypatch.setattr(server, "_get_dlc_job_detail", lambda _: detail)
    monkeypatch.setattr(server, "_job_runtime_paths", lambda _: (Path("/source/runtime_config.json"), Path("/source"), None))
    config = {"model": {"source_path": "/mnt/nasA/tianleniu/models/Qwen3.5-9B"}, "eval": {
        "tasks": "ai2d,mmmu_val_qwen3_official", "limit": -1, "gen_kwargs": "temperature=0", "output_path": "/mnt/nasA/tianleniu/eval_result/source"}}
    monkeypatch.setattr(server, "validate_judge_retry_source", lambda _: (config, Path("/source"), []))
    calls = []
    def build(request, **kwargs):
        calls.append((request, kwargs))
        return "safe preview" if kwargs.get("mask_secrets") else "sleep 0.5"
    monkeypatch.setattr(server, "_build_dlc_command", build)
    with TestClient(server.app) as client:
        client.cookies.set(server.AUTH_COOKIE_NAME, session)
        yield client, detail, calls


def test_retry_is_authenticated_and_reuses_one_active_workflow(retry_client):
    client, _, calls = retry_client
    response = client.post("/dlc/jobs/dlcsource/retry-judge")
    assert response.status_code == 200, response.text
    duplicate = client.post("/dlc/jobs/dlcsource/retry-judge")
    assert duplicate.json()["job_id"] == response.json()["job_id"]
    assert duplicate.json()["reused_workflow"] is True
    assert len(calls) == 2
    request, options = calls[-1]
    config = options["eval_config_override"]
    assert request.tasks == ["ai2d", "mmmu_val_qwen3_official"]
    assert request.judge_backend == "vllm"
    assert config["eval"]["limit"] == -1
    assert config["eval"]["gen_kwargs"] == "temperature=0"
    assert config["eval"]["judge_only_source_config"] == "/source/runtime_config.json"
    assert "eval_judge_retry_" in config["eval"]["output_path"]
    assert options["auth_user"]["access_key_id"] == "fake-key"


@pytest.mark.parametrize("status,owner,expected", [("Running", "owner-id", 409), ("Stopping", "owner-id", 409), ("Stopped", "someone-else", 403)])
def test_retry_checks_terminal_status_and_owner(retry_client, status, owner, expected):
    client, detail, calls = retry_client
    detail.update(Status=status, UserId=owner)
    assert client.post("/dlc/jobs/dlcsource/retry-judge").status_code == expected
    assert not calls


def test_retry_refuses_incomplete_inference(retry_client, monkeypatch):
    client, _, calls = retry_client
    def fail(_):
        raise ValueError("Source sample coverage is incomplete: mmmu")
    monkeypatch.setattr(server, "validate_judge_retry_source", fail)
    response = client.post("/dlc/jobs/dlcsource/retry-judge")
    assert response.status_code == 400
    assert "incomplete" in response.json()["detail"]
    assert not calls


def test_detail_reads_judge_requirements_from_runtime_config(retry_client, monkeypatch):
    client, _, _ = retry_client
    monkeypatch.setattr(server, "_load_json_path", lambda _: {"eval": {"tasks": "ai2d,mmmu_val_qwen3_official"}})
    response = client.get("/dlc/jobs/dlcsource")
    assert response.status_code == 200
    assert response.json()["requires_llm_judge"] is True
    assert response.json()["llm_judge_tasks"] == ["mmmu_val_qwen3_official"]
