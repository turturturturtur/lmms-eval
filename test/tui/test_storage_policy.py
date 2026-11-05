import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import HTTPException

from lmms_eval import storage_policy as storage
from lmms_eval.tui import server

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("extra", [
    "nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB",
    # A B filesystem must also be rejected when disguised under an A path.
    "nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasA/other",
    "cpfs://legacy/::/mnt/cpfsB",
    "bmcpfs://unknown/::/mnt/cpfsA/other",
])
def test_webui_rejects_extra_mount_even_with_both_required_mounts(extra):
    with pytest.raises(HTTPException) as error:
        server._require_dlc_nas_mount(storage.DEFAULT_MOUNTS + "," + extra, field="dlc.data_source_uris")
    assert error.value.status_code == 400


def test_paths_reject_aliases_and_traversal_to_b(tmp_path):
    alias = tmp_path / "checkpoint"
    alias.symlink_to("/mnt/nasB/old-model")
    with pytest.raises(ValueError):
        storage.validate_paths(str(alias))
    with pytest.raises(ValueError):
        storage.validate_paths("/mnt/nasA/../nasB/old-model")


def test_historical_result_paths_are_mapped_exactly_to_nas_a():
    old = "/mnt/cpfsB/tianleniu/Innovator-Tune/lmms-eval/eval_result/run-123/tasks/ai2d/result.json"
    expected = "/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_result/run-123/tasks/ai2d/result.json"
    assert str(storage.readable_path(old)) == expected
    assert str(storage.readable_path("/mnt/nasB/tianleniu/models/example")) == "/mnt/nasA/tianleniu/models/example"


def test_create_job_preserves_access_point_and_authenticated_credentials(tmp_path, monkeypatch):
    helper = ROOT / "run_scripts/dlc_create_job.py"
    spec = importlib.util.spec_from_file_location("dlc_create_job", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dlc = json.loads((ROOT / "run_scripts/config_dlc.json").read_text())["dlc"]
    args = ["submit", "pytorchjob", "--name", "eval_storage_test", "--command", "true"]
    for key in ("worker_image", "worker_cpu", "worker_gpu", "worker_memory", "worker_shared_memory",
                "data_source_uris", "resource_id", "workspace_id", "vpc_id", "switch_id",
                "security_group_id", "extended_cidrs"):
        args += ["--" + key, str(dlc[key])]
    parsed = module.parser().parse_args(args)
    body = module.build_request(parsed)
    assert body["ResourceId"] == "quota1cauvwqouvn"
    assert body["WorkspaceId"] == "247143"
    assert len(body["DataSources"]) == 2
    cpfs = next(x for x in body["DataSources"] if x["Uri"].startswith("bmcpfs://"))
    assert cpfs["AccessPointId"] == "ap-1sqxfm3pf4vkkcqec"
    assert cpfs["RoleArn"] == "acs:ram::1245621648665098:role/tianleniu-role"
    assert cpfs["MountPath"] == "/mnt/cpfsA/user/tianleniu"
    # Observe the external command without creating a real cloud job.
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"JobId":"dlctest12345678"}', "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(helper), *args, "--ignore_local_config",
                                    "--access_id", "session-id", "--access_key", "session-secret"])
    module.main()
    command = calls[0]
    assert command[command.index("--access-key-id") + 1] == "session-id"
    assert command[command.index("--access-key-secret") + 1] == "session-secret"
    assert json.loads(command[command.index("--body") + 1]) == body
    parsed.resource_id = "old-quota"
    with pytest.raises(ValueError):
        module.build_request(parsed)


def test_relocated_conda_environment_does_not_source_stale_activation(tmp_path):
    prefix = tmp_path / "conda/envs/lmm"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "bin/python").symlink_to(sys.executable)
    proc = subprocess.run(["bash", "-eu", "-c", 'source "$1"; activate_eval_runtime "$2"; command -v python',
                           "--", str(ROOT / "run_scripts/activate_runtime.sh"), str(prefix)],
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(prefix / "bin/python")
