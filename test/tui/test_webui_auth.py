import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmms_eval.tui import server


@pytest.fixture(autouse=True)
def auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "webui_users.json"
    path.write_text(
        json.dumps(
            {
                "admins": [
                    {
                        "username": "admin",
                        "display_name": "Admin User",
                        "access_key_id": "admin-ak",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(server.AUTH_FILE_ENV, str(path))
    monkeypatch.setenv(server.AUTH_SESSION_TTL_ENV, str(server.DEFAULT_AUTH_SESSION_TTL_SECONDS))
    monkeypatch.setattr(server, "_validate_auth_credentials", lambda access_key_id, secret_access_key: secret_access_key != "wrong-secret")
    monkeypatch.setattr(server, "_load_authenticated_aliyun_identity", lambda access_key_id, _secret_access_key: {"aliyun_user_id": f"user-{access_key_id}"})
    server._auth_sessions.clear()
    server._dlc_jobs_cache.clear()
    server._dlc_pool_usage_cache = None
    yield path
    server._auth_sessions.clear()
    server._dlc_jobs_cache.clear()
    server._dlc_pool_usage_cache = None


def _client() -> TestClient:
    return TestClient(server.app)


def _login(client: TestClient, access_key_id: str = "normal-ak", secret_access_key: str = "normal-secret"):
    return client.post(
        "/auth/login",
        json={
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
        },
    )


def _dlc_config() -> dict:
    return {
        "dlc": {
            "submit": True,
            "job_name": "eval_auth_test",
            "binary": "/tmp/dlc",
            "run_script": "/tmp/qwen35_worker.sh",
            "workers": 1,
            "worker_gpu": 8,
            "worker_cpu": 16,
            "worker_memory": 128,
            "worker_shared_memory": 64,
            "worker_image": "registry.example/image:latest",
            "data_source_uris": f"cpfs://example/::/mnt/cpfsB,{server.REQUIRED_NAS_MOUNT_URI},oss://example/::/mnt/oss",
            "resource_id": server.DEFAULT_DLC_RESOURCE_ID,
            "workspace_id": server.DEFAULT_DLC_WORKSPACE_ID,
            "vpc_id": "vpc-id",
            "switch_id": "switch-id",
            "security_group_id": "sg-id",
            "extended_cidrs": "0.0.0.0/0",
        }
    }


def _eval_payload() -> dict:
    return {
        "user": "",
        "job_name": "eval_auth_test",
        "eval_inference_mode": "ckpt",
        "model": "/tmp/model",
        "api_url": server.DEFAULT_API_EVAL_URL,
        "api_key": "",
        "dlc_path": "/tmp/dlc",
        "model_args": "",
        "tasks": ["ai2d"],
        "judge_backend": server.DEFAULT_JUDGE_BACKEND,
        "judge_api_url": "",
        "judge_api_key": "",
        "env_vars": "",
        "batch_size": 1,
        "limit": 1,
        "output_path": "/tmp/lmms-eval-results",
        "log_samples": True,
        "verbosity": "INFO",
        "device": None,
        "env_setup": "",
        "run_mode": "dlc",
        "dlc_config": _dlc_config(),
        "model_tp": 1,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 16,
        "base_port": 9000,
        "concurrency": 2,
        "gen_kwargs": "",
        "enable_thinking": False,
        "debug": False,
    }


def test_protected_api_requires_login():
    response = _client().get("/defaults")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_login_me_and_logout():
    client = _client()

    login_response = _login(client)

    assert login_response.status_code == 200
    assert login_response.json()["username"] == "normal-ak"
    assert login_response.json()["role"] == "user"
    assert "secret_access_key" not in login_response.text

    me_response = client.get("/auth/me")

    assert me_response.status_code == 200
    assert me_response.json()["access_key_id"] == "normal-ak"

    logout_response = client.post("/auth/logout")

    assert logout_response.status_code == 200
    assert client.get("/auth/me").status_code == 401


def test_invalid_login_returns_401():
    response = _login(_client(), secret_access_key="wrong-secret")

    assert response.status_code == 401
    assert response.json()["detail"] == "Access Key validation failed"


def test_auth_file_rejects_local_secret(auth_file: Path):
    data = json.loads(auth_file.read_text(encoding="utf-8"))
    data["admins"][0]["secret_access_key"] = "must-not-be-local"
    auth_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(Exception, match="must not contain secret_access_key"):
        server._load_auth_admins()


def test_admin_role_comes_from_local_access_key_id():
    client = _client()

    response = _login(client, access_key_id="admin-ak", secret_access_key="admin-secret")

    assert response.status_code == 200
    assert response.json()["username"] == "admin"
    assert response.json()["display_name"] == "Admin User"
    assert response.json()["role"] == "admin"


def test_start_eval_uses_session_credentials_and_ignores_request_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["access_key"] = "forged-ak"
    payload["secret_access_key"] = "forged-secret"
    response = client.post("/eval/start", json=payload)

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    command = server._jobs[job_id]["command"]
    assert "normal-ak" in command
    assert "normal-secret" in command
    assert "forged-ak" not in command
    assert "forged-secret" not in command


def test_preview_redacts_session_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client, access_key_id="admin-ak", secret_access_key="admin-secret").status_code == 200

    response = client.post("/eval/preview", json=_eval_payload())

    assert response.status_code == 200
    command = response.json()["command"]
    assert "********" in command
    assert "admin-ak" not in command
    assert "admin-secret" not in command


def test_preview_syncs_job_name_to_dlc_log_and_eval_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["job_name"] = "eval_synced_name"
    payload["output_path"] = "/tmp/lmms-eval-results/stale_name"
    payload["dlc_config"]["dlc"]["job_name"] = "eval_stale_name"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    expected_log_dir = server._path_with_leaf(server._default_eval_config()["log"]["dir"], "eval_synced_name", field_name="log.dir")
    assert '"job_name": "eval_synced_name"' in command
    assert f'"dir": "{expected_log_dir}"' in command
    assert '"output_path": "/tmp/lmms-eval-results/eval_synced_name"' in command
    assert "eval_stale_name" not in command
    assert "stale_name" not in command


def test_defaults_leave_evaluate_user_empty_and_keep_placeholders(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LMMS_EVAL_WEBUI_USER", "configured-user")
    client = _client()
    assert _login(client).status_code == 200

    response = client.get("/defaults")

    assert response.status_code == 200
    data = response.json()
    assert data["user"] == ""
    assert data["dlc_path"] == server.DEFAULT_DLC_PATH_TEMPLATE
    assert server.USER_PLACEHOLDER in data["model"]
    assert server.USER_PLACEHOLDER in data["output_path"]
    assert server.USER_PLACEHOLDER in data["env_vars"]
    assert data["judge_backend"] == "vllm"
    assert data["judge_api_url"] == server.DEFAULT_JUDGE_API_URL
    assert data["judge_api_key"] == ""
    assert data["api_model"] == ""
    assert data["dlc_config"]["dlc"]["binary"] == server.DEFAULT_DLC_PATH_TEMPLATE
    assert data["dlc_config"]["dlc"]["priority"] == 6
    assert data["dlc_config"]["dlc"]["judge"]["priority"] == 6
    assert server.USER_PLACEHOLDER in data["dlc_config"]["dlc"]["run_script"]
    assert "configured-user" not in json.dumps(data)


def test_defaults_match_qwen35_feishu_benchmarks_exactly():
    client = _client()
    assert _login(client).status_code == 200

    response = client.get("/defaults")

    assert response.status_code == 200
    tasks = response.json()["tasks"]
    expected_tasks = [
        "ai2d",
        "ai2d_no_mask",
        "chartqa",
        "infovqa_val",
        "mmbench_en_dev",
        "mmerealworld",
        "mmerealworld_cn",
        "mmmu_pro_standard_reasoning_qwen3_official",
        "mmmu_val_qwen3_official",
        "mmstar",
        "ocrbench",
        "realworldqa",
        "seedbench_2_plus",
        "vstar_bench",
        "simplevqa",
        "EMVista",
        "sfe-en",
        "microvqa",
        "embspatial",
        "erqa",
    ]

    assert tasks == expected_tasks
    assert len(tasks) == 20
    assert len(set(tasks)) == 20
    assert server._task_requires_llm_judge("simplevqa") is True

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200
    discovered_task_ids = {task["id"] for task in tasks_response.json()}
    assert set(tasks) <= discovered_task_ids


def test_username_alias_placeholder_is_replaced_for_webui_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["user"] = "alice"
    payload["model"] = "/mnt/cpfsB/<USERNAME>/model"
    payload["output_path"] = "/mnt/cpfsB/<USERNAME>/lmms-eval/eval_result/eval_alias"
    payload["dlc_path"] = "/mnt/cpfsB/<USERNAME>/dlc"
    payload["dlc_config"]["dlc"]["binary"] = "/mnt/cpfsB/<USERNAME>/dlc"
    payload["dlc_config"]["dlc"]["run_script"] = "/mnt/cpfsB/<USERNAME>/Innovator-Tune/lmms-eval/run_scripts/qwen35_worker.sh"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert "/mnt/cpfsB/alice/dlc" in command
    assert "/mnt/cpfsB/alice/model" in command
    assert "<USERNAME>" not in command


def test_default_dlc_config_uses_qwen35_worker():
    config = server._replace_user_placeholder(server._default_dlc_config(), "tianleniu")

    assert config["dlc"]["binary"] == "/mnt/cpfsB/tianleniu/dlc"
    assert config["dlc"]["run_script"] == "/mnt/cpfsB/tianleniu/Innovator-Tune/lmms-eval/run_scripts/qwen35_worker.sh"
    assert config["dlc"]["resource_id"] == server.DEFAULT_DLC_RESOURCE_ID
    assert config["dlc"]["judge"]["resource_id"] == server.DEFAULT_DLC_RESOURCE_ID
    assert server.REQUIRED_NAS_MOUNT_URI in config["dlc"]["data_source_uris"]
    assert server.REQUIRED_NAS_MOUNT_URI in config["dlc"]["judge"]["data_source_uris"]
    assert "qwen3_vl_worker" not in config["dlc"]["run_script"]


def test_preview_rejects_missing_eval_nas_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["dlc_config"]["dlc"]["data_source_uris"] = "cpfs://example/::/mnt/cpfsB,oss://example/::/mnt/oss"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert f"dlc.data_source_uris must include {server.REQUIRED_NAS_MOUNT_URI}" in response.json()["detail"]


def test_preview_rejects_missing_judge_nas_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["dlc_config"]["dlc"]["judge"] = {
        "resource_id": server.DEFAULT_DLC_RESOURCE_ID,
        "data_source_uris": "cpfs://example/::/mnt/cpfsB,oss://example/::/mnt/oss",
    }

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert f"dlc.judge.data_source_uris must include {server.REQUIRED_NAS_MOUNT_URI}" in response.json()["detail"]


def test_preview_rejects_non_default_dlc_resource_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["dlc_config"]["dlc"]["resource_id"] = "bad-resource-id"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert f"dlc.resource_id must be {server.DEFAULT_DLC_RESOURCE_ID}" in response.json()["detail"]


def test_preview_rejects_non_default_judge_resource_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["dlc_config"]["dlc"]["judge"] = {"resource_id": "bad-resource-id"}

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert f"dlc.judge.resource_id must be {server.DEFAULT_DLC_RESOURCE_ID}" in response.json()["detail"]


def test_preview_rejects_legacy_qwen3_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["dlc_config"]["dlc"]["run_script"] = "/tmp/qwen3_vl_worker.sh"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert "qwen35_worker.sh" in response.json()["detail"]


def test_local_judge_preview_uses_qwen35_inline_defaults_without_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["tasks"] = ["ocrbench"]

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert "qwen35_submit.sh" in command
    assert '"run_script": "/tmp/qwen35_worker.sh"' in command
    assert "config_judge.json" in command
    assert '"backend": "vllm"' in command
    assert f'"model": "{server.DEFAULT_LOCAL_JUDGE_MODEL}"' in command
    assert f'"model_path": "{server.DEFAULT_LOCAL_JUDGE_MODEL_PATH}"' in command
    assert f'"tp": {server.DEFAULT_LOCAL_JUDGE_TP}' in command
    assert f'"parallel": {server.DEFAULT_LOCAL_JUDGE_PARALLEL}' in command
    assert '"key": ""' in command
    assert "qwen3_vl_worker" not in command
    assert "qwen3_vl_submit" not in command


@pytest.mark.parametrize(
    ("judge_api_url", "judge_api_key", "expected_detail"),
    [
        ("", "", "LLM API URL is required"),
        ("https://judge.example.invalid/v1", "", "LLM API key is required"),
    ],
)
def test_api_judge_still_requires_url_and_key(
    judge_api_url: str,
    judge_api_key: str,
    expected_detail: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["tasks"] = ["ocrbench"]
    payload["judge_backend"] = "api"
    payload["judge_api_url"] = judge_api_url
    payload["judge_api_key"] = judge_api_key

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


def test_preview_rejects_unknown_judge_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["tasks"] = ["ocrbench"]
    payload["judge_backend"] = "unsupported"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported judge_backend: unsupported"


def test_webui_rejects_local_run_mode_and_points_to_safe_wrapper():
    client = _client()
    assert _login(client).status_code == 200
    payload = _eval_payload()
    payload["run_mode"] = "local"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 400
    assert "WebUI only supports DLC" in response.json()["detail"]
    assert "qwen35_local_eval.sh" in response.json()["detail"]


def test_webui_rejects_imported_local_yaml():
    client = _client()
    assert _login(client).status_code == 200

    response = client.post(
        "/eval/import-yaml",
        json={"yaml_content": "run_mode: local\nmodel: /tmp/raw-checkpoint\n"},
    )

    assert response.status_code == 400
    assert "WebUI only supports DLC" in response.json()["detail"]


@pytest.mark.parametrize(
    "judge_backend",
    ["vllm", "api"],
    ids=["local-vllm-judge", "remote-openai-judge"],
)
def test_dlc_yaml_roundtrip_preserves_judge_backend(
    judge_backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["tasks"] = ["ocrbench"]
    payload["judge_backend"] = judge_backend
    if judge_backend == "api":
        payload["judge_api_url"] = "https://judge.example.invalid/v1"
        payload["judge_api_key"] = "sk-roundtrip"

    export_response = client.post("/eval/export-yaml", json=payload)

    assert export_response.status_code == 200
    yaml_content = export_response.json()["yaml_content"]
    assert f"judge_backend: {judge_backend}" in yaml_content
    assert "source_path:" in yaml_content
    assert "processor_compat: required" in yaml_content

    import_response = client.post("/eval/import-yaml", json={"yaml_content": yaml_content})

    assert import_response.status_code == 200
    imported = import_response.json()
    assert imported["judge_backend"] == judge_backend
    assert imported["tasks"] == ["ocrbench"]
    if judge_backend == "api":
        assert imported["judge_api_url"] == "https://judge.example.invalid/v1"
        assert imported["judge_api_key"] == "sk-roundtrip"
    else:
        assert imported["judge_api_url"] == ""
        assert imported["judge_api_key"] == ""


def test_dlc_yaml_roundtrip_preserves_optional_api_model_name():
    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-roundtrip"
    payload["api_model"] = "kimi-for-coding"

    export_response = client.post("/eval/export-yaml", json=payload)

    assert export_response.status_code == 200
    yaml_content = export_response.json()["yaml_content"]
    assert "api_model: kimi-for-coding" in yaml_content

    import_response = client.post("/eval/import-yaml", json={"yaml_content": yaml_content})

    assert import_response.status_code == 200
    assert import_response.json()["api_model"] == "kimi-for-coding"


def test_dlc_yaml_roundtrip_preserves_blank_optional_api_model_name():
    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-roundtrip-blank"
    payload["api_model"] = "   "

    export_response = client.post("/eval/export-yaml", json=payload)

    assert export_response.status_code == 200
    yaml_content = export_response.json()["yaml_content"]
    assert "api_model: ''" in yaml_content

    import_response = client.post("/eval/import-yaml", json={"yaml_content": yaml_content})

    assert import_response.status_code == 200
    assert import_response.json()["api_model"] == ""


def test_import_legacy_api_runtime_yaml_recovers_model_path_as_api_model():
    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-legacy-runtime"
    payload["api_model"] = "legacy-runtime-model"
    config = server._build_eval_config(server.ExportYamlRequest(**payload))
    config["run_mode"] = "dlc"

    client = _client()
    assert _login(client).status_code == 200
    response = client.post(
        "/eval/import-yaml",
        json={"yaml_content": server.yaml.safe_dump(config, sort_keys=False)},
    )

    assert response.status_code == 200
    imported = response.json()
    assert imported["eval_inference_mode"] == "api"
    assert imported["api_model"] == "legacy-runtime-model"


def test_import_runtime_yaml_prefers_source_model_over_resolved_view():
    payload = _eval_payload()
    source_model = "/mnt/cpfsB/user/checkpoint-227"
    resolved_model = "/mnt/cpfsB/user/views/checkpoint-227-video-compat"
    config = server._build_eval_config(server.ExportYamlRequest(**payload))
    config["model"]["source_path"] = source_model
    config["model"]["path"] = resolved_model
    config["model"]["resolved_path"] = resolved_model
    config["run_mode"] = "dlc"

    client = _client()
    assert _login(client).status_code == 200
    response = client.post(
        "/eval/import-yaml",
        json={"yaml_content": server.yaml.safe_dump(config, sort_keys=False)},
    )

    assert response.status_code == 200
    assert response.json()["model"] == source_model


def test_ckpt_webui_config_records_submit_time_preflight_contract():
    payload = _eval_payload()
    payload["user"] = "test-user"
    request = server.EvalRequest(**payload)

    config = server._build_eval_config(request)

    assert config["model"]["path"] == payload["model"]
    assert config["model"]["source_path"] == payload["model"]
    assert config["model"]["processor_compat"] == "required"
    assert config["model"]["view_root"] == (
        "/mnt/cpfsB/test-user/lmms_eval_views"
    )
    assert config["model"]["startup_timeout_seconds"] == 1800
    assert "resolved_path" not in config["model"]
    assert "preflight" not in config["model"]


def test_dlc_job_detail_returns_model_artifact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = tmp_path / "runtime_config.json"
    runtime_config.write_text(
        json.dumps(
            {
                "model": {
                    "source_path": "/source/checkpoint",
                    "path": "/views/resolved",
                    "resolved_path": "/views/resolved",
                    "processor_compat": "required",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "eval_model_provenance",
            "Status": "Running",
        },
    )
    monkeypatch.setattr(
        server,
        "_job_runtime_paths",
        lambda _detail: (runtime_config, None, "/logs/eval"),
    )

    client = _client()
    assert _login(client).status_code == 200
    response = client.get("/dlc/jobs/dlcprovenance")

    assert response.status_code == 200
    detail = response.json()
    assert detail["model_source_path"] == "/source/checkpoint"
    assert detail["model_resolved_path"] == "/views/resolved"
    assert detail["processor_compat"] == "required"


def test_ckpt_reasoning_task_syncs_judge_key_into_eval_env():
    payload = _eval_payload()
    payload["tasks"] = ["mathverse_testmini_reasoning"]
    payload["judge_backend"] = "api"
    payload["judge_api_url"] = "https://judge.example.invalid/v1"
    payload["judge_api_key"] = "sk-judge-secret"

    request = server.EvalRequest(**payload)
    eval_config = server._build_eval_config(request)
    judge_config = server._build_judge_config(request, eval_config)

    assert server._task_requires_llm_judge("mathverse_testmini_reasoning") is True
    assert eval_config["env"]["judge_api_key"] == "sk-judge-secret"
    assert eval_config["env"]["judge_base_url"] == "https://judge.example.invalid/v1"
    assert eval_config["env"]["openai_api_key"] == "sk-judge-secret"
    assert judge_config is not None
    assert judge_config["eval"]["tasks"] == "mathverse_testmini_reasoning"

    redacted = server._redact_eval_config(eval_config)
    assert redacted["env"]["judge_api_key"] == server.MASKED_SECRET
    assert redacted["env"]["openai_api_key"] == server.MASKED_SECRET


def test_api_eval_preview_redacts_token_and_forces_cpu_dlc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-api-secret"
    payload["dlc_config"]["dlc"]["worker_gpu"] = 8

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert '"backend": "openai"' in command
    assert '"worker_gpu": 0' in command
    assert '"worker_cpu": 16' in command
    assert "sk-api-secret" not in command
    assert server.MASKED_SECRET in command


def test_api_eval_preview_uses_optional_openai_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-api-secret"
    payload["api_model"] = "  kimi-for-coding  "

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert '"path": "kimi-for-coding"' in command


def test_api_eval_preview_keeps_legacy_model_when_optional_name_is_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)
    monkeypatch.setenv("LMMS_EVAL_WEBUI_API_MODEL", "legacy-api-model")

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-api-secret"
    payload["api_model"] = "   "

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert '"path": "legacy-api-model"' in command


def test_api_eval_judge_preview_redacts_eval_and_judge_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["tasks"] = ["simplevqa"]
    payload["judge_backend"] = "api"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "eval-api-secret-for-judge-preview"
    payload["judge_api_url"] = "https://judge.example.invalid/v1"
    payload["judge_api_key"] = "judge-api-secret-for-preview"

    response = client.post("/eval/preview", json=payload)

    assert response.status_code == 200
    command = response.json()["command"]
    assert "config_judge.json" in command
    assert "eval-api-secret-for-judge-preview" not in command
    assert "judge-api-secret-for-preview" not in command
    assert command.count(server.MASKED_SECRET) >= 2


def test_api_eval_with_local_judge_keeps_eight_gpu_dlc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    submit_script = tmp_path / "qwen35_submit.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(server, "DLC_SUBMIT_SCRIPT", submit_script)

    client = _client()
    assert _login(client).status_code == 200

    payload = _eval_payload()
    payload["eval_inference_mode"] = "api"
    payload["tasks"] = ["simplevqa"]
    payload["judge_backend"] = "vllm"
    payload["api_url"] = "https://api.example.invalid/v1"
    payload["api_key"] = "sk-api-secret"
    payload["dlc_config"]["dlc"]["worker_gpu"] = 8

    request = server.PreviewRequest(**payload)
    dlc_config = server._request_dlc_config(request)
    response = client.post("/eval/preview", json=payload)

    assert dlc_config["dlc"]["worker_gpu"] == 8
    assert response.status_code == 200
    command = response.json()["command"]
    assert '"backend": "openai"' in command
    assert '"backend": "vllm"' in command
    assert '"worker_gpu": 8' in command


def test_dlc_job_list_marks_kill_permission_for_owner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")
    monkeypatch.setattr(
        server,
        "_list_dlc_jobs_from_cli",
        lambda **_kwargs: [
            {
                "job_id": "dlcowned",
                "name": "eval_owned",
                "status": "Running",
                "user_id": "user-normal-ak",
            },
            {
                "job_id": "dlcother",
                "name": "eval_other",
                "status": "Running",
                "user_id": "user-other-ak",
            },
            {
                "job_id": "dlcdone",
                "name": "judge_done",
                "status": "Succeeded",
                "user_id": "user-normal-ak",
            },
        ],
    )

    client = _client()
    assert _login(client).status_code == 200

    response = client.get("/dlc/jobs")

    assert response.status_code == 200
    jobs = {job["job_id"]: job for job in response.json()["jobs"]}
    assert jobs["dlcowned"]["can_kill"] is True
    assert jobs["dlcowned"]["kill_disabled_reason"] == ""
    assert jobs["dlcother"]["can_kill"] is False
    assert "Only the job owner" in jobs["dlcother"]["kill_disabled_reason"]
    assert jobs["dlcdone"]["can_kill"] is False
    assert "not killable" in jobs["dlcdone"]["kill_disabled_reason"]


def test_dlc_job_list_queries_and_returns_explicit_time_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")
    captured: dict[str, object] = {}

    def fake_list_jobs(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(server, "_list_dlc_jobs_from_cli", fake_list_jobs)
    client = _client()
    assert _login(client).status_code == 200

    response = client.get(
        "/dlc/jobs",
        params={
            "start_time": "2026-06-28T12:00:00Z",
            "end_time": "2026-07-28T12:00:00Z",
        },
    )

    assert response.status_code == 200
    assert captured["start_time"] == "2026-06-28T12:00:00Z"
    assert captured["end_time"] == "2026-07-28T12:00:00Z"
    assert response.json()["start_time"] == "2026-06-28T12:00:00Z"
    assert response.json()["end_time"] == "2026-07-28T12:00:00Z"


@pytest.mark.parametrize(
    ("params", "expected_detail"),
    [
        ({"start_time": "2026-06-28T12:00:00Z"}, "provided together"),
        (
            {"start_time": "2026-07-28T12:00:00Z", "end_time": "2026-06-28T12:00:00Z"},
            "earlier than end_time",
        ),
        (
            {"start_time": "2026-06-28T12:00:00", "end_time": "2026-07-28T12:00:00Z"},
            "include a timezone",
        ),
    ],
)
def test_dlc_job_list_rejects_invalid_time_windows(params: dict[str, str], expected_detail: str):
    client = _client()
    assert _login(client).status_code == 200

    response = client.get("/dlc/jobs", params=params)

    assert response.status_code == 422
    assert expected_detail in response.json()["detail"]


def test_dlc_job_list_marks_admin_can_kill_active_jobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")
    monkeypatch.setattr(
        server,
        "_list_dlc_jobs_from_cli",
        lambda **_kwargs: [
            {
                "job_id": "dlcother",
                "name": "eval_other",
                "status": "EnvPreparing",
                "user_id": "user-other-ak",
            },
        ],
    )

    client = _client()
    assert _login(client, access_key_id="admin-ak", secret_access_key="admin-secret").status_code == 200

    response = client.get("/dlc/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["can_kill"] is True


def test_user_can_kill_own_active_dlc_job(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "eval_owned",
            "Status": "Running",
            "UserId": "user-normal-ak",
        },
    )
    calls: list[tuple[list[str], str, str]] = []

    def fake_run(args: list[str], auth_user: dict, *, timeout: int = 30) -> str:
        calls.append((args, auth_user["access_key_id"], auth_user["secret_access_key"]))
        assert timeout == server.DLC_STOP_TIMEOUT_SECONDS
        return "stopped"

    monkeypatch.setattr(server, "_run_authenticated_dlc_command", fake_run)

    client = _client()
    assert _login(client).status_code == 200

    response = client.post("/dlc/jobs/dlcowned/kill")

    assert response.status_code == 200
    assert response.json()["status"] == "kill_requested"
    assert calls == [(["stop", "job", "dlcowned", "--force", "--quiet"], "normal-ak", "normal-secret")]


def test_authenticated_dlc_command_uses_session_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command: list[str], **_kwargs) -> Completed:
        calls.append(command)
        return Completed()

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    output = server._run_authenticated_dlc_command(
        ["stop", "job", "dlcowned", "--force", "--quiet"],
        {
            "access_key_id": "normal-ak",
            "secret_access_key": "normal-secret",
        },
    )

    assert output == "ok"
    command = calls[0]
    assert command[:5] == ["/tmp/dlc", "stop", "job", "dlcowned", "--force"]
    assert "--access_id" in command
    assert "normal-ak" in command
    assert "--access_key" in command
    assert "normal-secret" in command
    assert "--ignore_local_config" in command


def test_authenticated_dlc_command_timeout_redacts_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")

    def fake_run(*_args, **_kwargs):
        raise server.subprocess.TimeoutExpired(cmd="/tmp/dlc", timeout=1)

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    with pytest.raises(server.HTTPException) as exc_info:
        server._run_authenticated_dlc_command(
            ["stop", "job", "dlcowned", "--force", "--quiet"],
            {
                "access_key_id": "normal-ak",
                "secret_access_key": "normal-secret",
            },
        )

    detail = str(exc_info.value.detail)
    assert "normal-ak" not in detail
    assert "normal-secret" not in detail
    assert "********" in detail


def test_authenticated_dlc_command_failure_redacts_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/dlc")

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "failed for normal-ak with normal-secret"

    monkeypatch.setattr(server.subprocess, "run", lambda *_args, **_kwargs: Completed())

    with pytest.raises(server.HTTPException) as exc_info:
        server._run_authenticated_dlc_command(
            ["stop", "job", "dlcowned", "--force", "--quiet"],
            {
                "access_key_id": "normal-ak",
                "secret_access_key": "normal-secret",
            },
        )

    detail = str(exc_info.value.detail)
    assert "normal-ak" not in detail
    assert "normal-secret" not in detail
    assert "********" in detail


def test_user_cannot_kill_other_users_dlc_job(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "eval_other",
            "Status": "Running",
            "UserId": "user-other-ak",
        },
    )
    called = False

    def fake_run(*_args, **_kwargs) -> str:
        nonlocal called
        called = True
        return "stopped"

    monkeypatch.setattr(server, "_run_authenticated_dlc_command", fake_run)

    client = _client()
    assert _login(client).status_code == 200

    response = client.post("/dlc/jobs/dlcother/kill")

    assert response.status_code == 403
    assert "Only the job owner" in response.json()["detail"]
    assert called is False


def test_admin_can_kill_other_users_dlc_job(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "judge_other",
            "Status": "Queuing",
            "UserId": "user-other-ak",
        },
    )
    calls: list[list[str]] = []

    def fake_run(args: list[str], _auth_user: dict, *, timeout: int = 30) -> str:
        calls.append(args)
        return "stopped"

    monkeypatch.setattr(server, "_run_authenticated_dlc_command", fake_run)

    client = _client()
    assert _login(client, access_key_id="admin-ak", secret_access_key="admin-secret").status_code == 200

    response = client.post("/dlc/jobs/dlcother/kill")

    assert response.status_code == 200
    assert calls == [["stop", "job", "dlcother", "--force", "--quiet"]]


def test_kill_inactive_dlc_job_returns_409(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "eval_done",
            "Status": "Succeeded",
            "UserId": "user-normal-ak",
        },
    )
    monkeypatch.setattr(server, "_run_authenticated_dlc_command", lambda *_args, **_kwargs: pytest.fail("kill command should not run"))

    client = _client()
    assert _login(client).status_code == 200

    response = client.post("/dlc/jobs/dlcdone/kill")

    assert response.status_code == 409
    assert "not killable" in response.json()["detail"]


def test_kill_non_view_log_job_returns_400(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        server,
        "_get_dlc_job_detail",
        lambda job_id: {
            "JobId": job_id,
            "DisplayName": "train_other",
            "Status": "Running",
            "UserId": "user-normal-ak",
        },
    )
    monkeypatch.setattr(server, "_run_authenticated_dlc_command", lambda *_args, **_kwargs: pytest.fail("kill command should not run"))

    client = _client()
    assert _login(client).status_code == 200

    response = client.post("/dlc/jobs/dlctrain/kill")

    assert response.status_code == 400
    assert "must start with" in response.json()["detail"]
