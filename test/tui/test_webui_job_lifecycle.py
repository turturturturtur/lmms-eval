"""Exercise submission independently of SSE connections, using real subprocesses."""

import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lmms_eval.tui import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_build_command", lambda *a, **kw: "redacted preview")
    session, _ = server._create_auth_session({
        "username": "review-a", "role": "user", "access_key_id": "fake-a",
        "secret_access_key": "fake-secret-a",
    })
    with TestClient(server.app) as client:
        client.cookies.set(server.AUTH_COOKIE_NAME, session)
        yield client


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate()


def start(client):
    response = client.post("/eval/start", json={"model": "test", "tasks": ["ai2d"]})
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def test_start_runs_once_without_log_subscription_and_replays(client, monkeypatch, tmp_path):
    marker = tmp_path / "executions"
    monkeypatch.setattr(server, "_build_shell_command", lambda *a, **kw: f"echo run >> {shlex.quote(str(marker))}")
    job_id = start(client)
    wait_until(marker.exists)
    wait_until(lambda: server._jobs[job_id]["status"] == "completed")
    first = client.get(f"/eval/{job_id}/stream")
    second = client.get(f"/eval/{job_id}/stream")
    assert first.status_code == second.status_code == 200
    assert '"type": "done"' in second.text
    assert marker.read_text().splitlines() == ["run"]


def test_failed_process_is_not_completed(client, monkeypatch):
    monkeypatch.setattr(server, "_build_shell_command", lambda *a, **kw: "exit 3")
    job_id = start(client)
    wait_until(lambda: server._jobs[job_id]["status"] == "error")
    assert '"exit_code": 3' in client.get(f"/eval/{job_id}/stream").text


def test_other_user_cannot_stream_inspect_or_stop(client, monkeypatch):
    monkeypatch.setattr(server, "_build_shell_command", lambda *a, **kw: "exit 0")
    job_id = start(client)
    session, _ = server._create_auth_session({
        "username": "review-b", "role": "user", "access_key_id": "fake-b",
        "secret_access_key": "fake-secret-b",
    })
    client.cookies.set(server.AUTH_COOKIE_NAME, session)
    assert client.get(f"/eval/{job_id}/stream").status_code == 403
    assert client.get(f"/eval/{job_id}").status_code == 403
    assert client.post(f"/eval/{job_id}/stop").status_code == 403


def test_stop_during_submission_stops_late_remote_receipt(client, monkeypatch):
    # Cloud creation can finish after the user has clicked Stop.
    command = 'sleep 0.1; echo "[WEBUI_DLC_JOB] eval dlcreview"; while [ ! -f "$LMMS_EVAL_WEBUI_CANCEL_FILE" ]; do sleep 0.05; done'
    monkeypatch.setattr(server, "_build_shell_command", lambda *a, **kw: command)
    stopped = []
    monkeypatch.setattr(server, "_get_dlc_job_detail", lambda _: {"Status": "Stopped" if stopped else "Running"})
    monkeypatch.setattr(server, "_run_authenticated_dlc_command", lambda args, user, **kw: stopped.append((args, user["access_key_id"])))
    job_id = start(client)
    response = client.post(f"/eval/{job_id}/stop")
    assert response.status_code == 200
    wait_until(lambda: server._jobs[job_id]["status"] == "stopped")
    assert stopped and stopped[0][0][:3] == ["stop", "job", "dlcreview"]
    assert stopped[0][1] == "fake-a"


def test_cloud_reads_use_the_authenticated_session(client, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/fake/dlc")
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"JobId": "dlcreview", "DisplayName": "eval_review"}), stderr="")
    monkeypatch.setattr(server.subprocess, "run", run)
    response = client.get("/dlc/jobs/dlcreview")
    assert response.status_code == 200, response.text
    assert "fake-a" in calls[-1]
    assert "fake-secret-a" in calls[-1]
    assert "--ignore_local_config" in calls[-1]


def test_invalid_task_python_is_rejected_before_writing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "TASKS_DIR", tmp_path)
    response = client.post("/tasks/create", json={
        "task_id": "invalid_fixture", "yaml_content": "task: invalid_fixture\n",
        "python_content": "def broken(:\n", "overwrite": False,
    })
    assert response.status_code == 400
    assert "Python" in response.json()["detail"]
    assert not (tmp_path / "invalid_fixture").exists()


def test_default_mmmu_pro_leaf_is_in_judge_config():
    task = "mmmu_pro_standard_reasoning_qwen3_official"
    request = server.EvalRequest(user="tianleniu", model="/tmp/model", tasks=[task])
    config = server._build_judge_config(request, server._build_eval_config(request))
    assert config is not None
    assert config["eval"]["tasks"] == task
