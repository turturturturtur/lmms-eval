import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEFAULT_DLC_RESOURCE_ID = "quotaev2tl4w6aw0"
REQUIRED_NAS_MOUNT_URI = "nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB"
MOUNT_URIS = f"cpfs://example/::/mnt/cpfsB,{REQUIRED_NAS_MOUNT_URI},oss://example/::/mnt/oss"
MOUNT_URIS_WITHOUT_NAS = "cpfs://example/::/mnt/cpfsB,oss://example/::/mnt/oss"
MOUNT_URIS_WITHOUT_CPFSB = f"{REQUIRED_NAS_MOUNT_URI},oss://example/::/mnt/oss"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _configs(tmp_path: Path, lmms_root: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "checkpoint-raw"
    model_path.mkdir()
    _write_json(model_path / "config.json", {"model_type": "qwen3_5"})
    _write_json(model_path / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write_json(model_path / "tokenizer.json", {"version": "1.0"})
    _write_json(
        model_path / "model.safetensors.index.json",
        {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}},
    )
    (model_path / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    _write_json(
        model_path / "processor_config.json",
        {
            "processor_class": "Qwen3_5Processor",
            "video_processor": {
                "video_processor_type": "Qwen3VLVideoProcessor",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "merge_size": 2,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "size": {"shortest_edge": 3136, "longest_edge": 12845056},
            },
        },
    )
    fake_venv = tmp_path / "fake_venv"
    fake_venv_bin = fake_venv / "bin"
    fake_venv_bin.mkdir(parents=True)
    fake_python = fake_venv_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source_path=""
view_root=""
run_id=""
while (( $# > 0 )); do
  case "$1" in
    --source) source_path="$(readlink -f "$2")"; shift 2 ;;
    --view-root) view_root="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ -z "${source_path}" || -z "${view_root}" || -z "${run_id}" ]]; then
  echo "missing fake preflight argument" >&2
  exit 2
fi
resolved_path="${view_root}/${run_id}_fake_video_processor_compat"
mkdir -p "${resolved_path}"
printf '{}\\n' > "${resolved_path}/video_preprocessor_config.json"
printf '{"source_path":"%s","resolved_path":"%s","model_type":"qwen3_5","processor_class":"Qwen3VLVideoProcessor","compatibility":"qwen35_video_processor_view","source_manifest_sha256":"%064d","transformers_version":"4.57.6","vllm_version":"0.21.0"}\\n' \
  "${source_path}" "${resolved_path}" 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    dlc_config = {
        "dlc": {
            "submit": True,
            "job_name": "eval_submitter_judge_template",
            "binary": "/bin/true",
            "run_script": str(lmms_root / "run_scripts" / "qwen35_worker.sh"),
            "workers": 1,
            "worker_gpu": 8,
            "worker_cpu": 110,
            "worker_memory": "1500Gi",
            "worker_shared_memory": "1500Gi",
            "priority": 6,
            "job_max_running_time_minutes": 10080,
            "running_timeout": 86400,
            "worker_image": "eval-image",
            "data_source_uris": MOUNT_URIS,
            "resource_id": DEFAULT_DLC_RESOURCE_ID,
            "workspace_id": "workspace-eval",
            "vpc_id": "vpc-eval",
            "switch_id": "switch-eval",
            "security_group_id": "sg-eval",
            "extended_cidrs": "10.0.0.0/24",
            "region": "cn-wulanchabu",
            "endpoint": "pai-dlc.cn-wulanchabu.aliyuncs.com",
            "judge": {
                "workers": 1,
                "worker_gpu": 0,
                "worker_cpu": 4,
                "worker_memory": "8Gi",
                "worker_shared_memory": "2Gi",
                "priority": 6,
                "job_max_running_time_minutes": 60,
                "running_timeout": 3600,
                "worker_image": "judge-image",
                "data_source_uris": MOUNT_URIS,
                "resource_id": DEFAULT_DLC_RESOURCE_ID,
                "workspace_id": "workspace-judge",
                "vpc_id": "vpc-judge",
                "switch_id": "switch-judge",
                "security_group_id": "sg-judge",
                "extended_cidrs": "10.1.0.0/24",
            },
        }
    }
    eval_config = {
        "env": {"venv_path": str(fake_venv)},
        "log": {"dir": str(tmp_path / "logs")},
        "distributed": {},
        "model": {
            "path": str(model_path),
            "tp": 1,
            "processor_compat": "required",
            "view_root": str(tmp_path / "model_views"),
        },
        "eval": {
            "tasks": "ocrbench",
            "output_path": str(tmp_path / "results"),
            "debug": False,
        },
    }
    judge_config = {
        "env": {},
        "log": {"dir": str(tmp_path / "judge_logs")},
        "judge": {
            "backend": "api",
            "parallel": 1,
            "model": "judge-model",
            "api": {"key": "dummy", "base_url": "https://judge.invalid/v1"},
        },
        "eval": {
            "input_result_path": str(tmp_path / "results"),
            "tasks": "ocrbench",
            "output_path": str(tmp_path / "judge_results"),
            "debug": False,
        },
    }

    dlc_path = tmp_path / "config_dlc.json"
    eval_path = tmp_path / "config_eval.json"
    judge_path = tmp_path / "config_judge.json"
    _write_json(dlc_path, dlc_config)
    _write_json(eval_path, eval_config)
    _write_json(judge_path, judge_config)
    return dlc_path, eval_path, judge_path


def test_qwen35_submitter_preflight_resolves_raw_checkpoint_in_runtime_config(
    tmp_path: Path,
):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, _judge_config = _configs(tmp_path, lmms_root)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    runtime_configs = sorted(
        (tmp_path / "logs" / "eval_submitter_judge_template").glob(
            "*/runtime_config.json"
        )
    )
    assert runtime_configs
    runtime = json.loads(runtime_configs[-1].read_text(encoding="utf-8"))
    source = json.loads(eval_config.read_text(encoding="utf-8"))["model"]["path"]
    assert runtime["model"]["source_path"] == source
    assert runtime["model"]["source_input_path"] == source
    assert runtime["model"]["path"] == runtime["model"]["resolved_path"]
    assert runtime["model"]["path"] != source
    assert Path(runtime["model"]["path"], "video_preprocessor_config.json").is_file()
    assert runtime["model"]["preflight"]["processor_class"] == "Qwen3VLVideoProcessor"
    assert isinstance(runtime["model"]["preflight"]["lmms_eval_git_dirty"], bool)
    assert (
        len(runtime["model"]["preflight"]["lmms_eval_tree_state_sha256"]) == 64
    )


def test_qwen35_submitter_rejects_unsafe_served_model_name(tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, _judge_config = _configs(tmp_path, lmms_root)
    eval_payload = json.loads(eval_config.read_text(encoding="utf-8"))
    eval_payload["model"]["served_model_name"] = "unsafe,model"
    _write_json(eval_config, eval_payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert "served_model_name must match" in proc.stdout


def test_qwen35_submitter_rejects_missing_explicit_view_root(tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, _judge_config = _configs(tmp_path, lmms_root)
    eval_payload = json.loads(eval_config.read_text(encoding="utf-8"))
    del eval_payload["model"]["view_root"]
    _write_json(eval_config, eval_payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert "model.view_root must be an explicit absolute shared path" in proc.stdout


@pytest.mark.parametrize("script_name", ["qwen35_submit.sh", "qwen3_vl_submit.sh"])
def test_judge_dlc_uses_cpu_only_template_resource(script_name: str, tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / script_name),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    lines = proc.stdout.splitlines()
    eval_line = next(line for line in lines if line.startswith("[DRY_RUN][eval]"))
    judge_line = next(line for line in lines if line.startswith("[DRY_RUN][judge]"))

    assert f"--resource_id={DEFAULT_DLC_RESOURCE_ID}" in eval_line
    assert "--workspace_id=workspace-eval" in eval_line
    assert "--worker_gpu=8" in eval_line
    assert "--worker_image=eval-image" in eval_line
    assert "--priority=6" in eval_line
    assert REQUIRED_NAS_MOUNT_URI in eval_line

    assert f"--resource_id={DEFAULT_DLC_RESOURCE_ID}" in judge_line
    assert "--workspace_id=workspace-judge" in judge_line
    assert "--worker_gpu=0" in judge_line
    assert "--worker_cpu=4" in judge_line
    assert "--worker_memory=8Gi" in judge_line
    assert "--worker_shared_memory=2Gi" in judge_line
    assert "--worker_image=judge-image" in judge_line
    assert "--priority=6" in judge_line
    assert REQUIRED_NAS_MOUNT_URI in judge_line


def test_qwen35_local_vllm_dry_run_uses_one_priority_nine_eval_job_with_inline_judge_config(tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    dlc_payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    dlc_payload["dlc"]["priority"] = 9
    _write_json(dlc_config, dlc_payload)
    judge_payload = json.loads(judge_config.read_text(encoding="utf-8"))
    judge_payload["judge"]["backend"] = "vllm"
    judge_payload["judge"]["model"] = "Qwen3.5-9B"
    judge_payload["judge"]["api"] = {"key": "", "base_url": ""}
    judge_payload["judge"]["vllm"] = {
        "model_path": "/mnt/cpfsB/tianleniu/Innovator-Tune/models/Qwen3.5-9B",
        "tp": 8,
        "max_model_len": 40960,
        "gpu_memory_utilization": "0.88",
        "max_num_seqs": 192,
        "port": 8002,
    }
    _write_json(judge_config, judge_payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    dry_run_lines = [line for line in proc.stdout.splitlines() if line.startswith("[DRY_RUN]")]
    assert len(dry_run_lines) == 1
    assert dry_run_lines[0].startswith("[DRY_RUN][eval]")
    assert "--priority=9" in dry_run_lines[0]
    assert "qwen35_worker.sh" in dry_run_lines[0]
    assert "judge_runtime_config.json" in dry_run_lines[0]
    assert "[DRY_RUN][judge]" not in proc.stdout

    judge_runtime_configs = sorted(
        (tmp_path / "logs" / "eval_submitter_judge_template").glob("*/judge_runtime_config.json")
    )
    assert judge_runtime_configs
    judge_runtime = json.loads(judge_runtime_configs[-1].read_text(encoding="utf-8"))
    assert judge_runtime["judge"]["backend"] == "vllm"
    assert judge_runtime["eval"]["input_result_path"].startswith(str(tmp_path / "results"))
    assert judge_runtime["eval"]["output_path"].endswith("/judge")


def test_qwen35_api_judge_dry_run_keeps_two_priority_nine_jobs(tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    dlc_payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    dlc_payload["dlc"]["priority"] = 9
    dlc_payload["dlc"]["judge"]["priority"] = 9
    _write_json(dlc_config, dlc_payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    dry_run_lines = [line for line in proc.stdout.splitlines() if line.startswith("[DRY_RUN]")]
    assert len(dry_run_lines) == 2
    assert dry_run_lines[0].startswith("[DRY_RUN][eval]")
    assert dry_run_lines[1].startswith("[DRY_RUN][judge]")
    assert all("--priority=9" in line for line in dry_run_lines)
    assert "--worker_gpu=8" in dry_run_lines[0]
    assert "--worker_gpu=0" in dry_run_lines[1]


def test_qwen35_submitter_accepts_cpu_only_api_eval(tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, _judge_config = _configs(tmp_path, lmms_root)
    dlc_payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    dlc_payload["dlc"]["worker_gpu"] = 0
    dlc_payload["dlc"]["worker_cpu"] = 8
    dlc_payload["dlc"]["worker_memory"] = "64Gi"
    dlc_payload["dlc"]["worker_shared_memory"] = "16Gi"
    _write_json(dlc_config, dlc_payload)

    eval_payload = json.loads(eval_config.read_text(encoding="utf-8"))
    eval_payload["env"] = {
        "api_type": "openai",
        "openai_api_key": "sk-test",
        "openai_api_url": "https://api.example.invalid/v1",
    }
    eval_payload["model"]["backend"] = "openai"
    eval_payload["model"]["path"] = "router_fs_eval"
    eval_payload["eval"]["tasks"] = "ai2d"
    eval_payload["eval"]["limit"] = 50
    _write_json(eval_config, eval_payload)

    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / "qwen35_submit.sh"),
            str(dlc_config),
            str(eval_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    eval_line = next(line for line in proc.stdout.splitlines() if line.startswith("[DRY_RUN][eval]"))
    assert "--worker_gpu=0" in eval_line
    assert "--worker_cpu=8" in eval_line
    assert "--worker_memory=64Gi" in eval_line


@pytest.mark.parametrize("script_name", ["qwen35_submit.sh", "qwen3_vl_submit.sh"])
def test_submitter_backfills_judge_credentials_into_eval_runtime(script_name: str, tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    judge_payload = json.loads(judge_config.read_text(encoding="utf-8"))
    judge_payload["judge"]["api"]["key"] = "sk-direct-judge"
    judge_payload["judge"]["api"]["base_url"] = "https://judge.example.invalid/v1"
    _write_json(judge_config, judge_payload)

    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / script_name),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    runtime_configs = sorted((tmp_path / "logs" / "eval_submitter_judge_template").glob("*/runtime_config.json"))
    assert runtime_configs
    runtime = json.loads(runtime_configs[-1].read_text(encoding="utf-8"))
    assert runtime["env"]["judge_api_key"] == "sk-direct-judge"
    assert runtime["env"]["judge_base_url"] == "https://judge.example.invalid/v1"
    assert runtime["env"]["openai_api_key"] == "sk-direct-judge"


@pytest.mark.parametrize("script_name", ["qwen35_submit.sh", "qwen3_vl_submit.sh"])
@pytest.mark.parametrize("field_path", [("dlc", "resource_id"), ("dlc", "judge", "resource_id")])
def test_submitter_rejects_non_default_resource_id(script_name: str, field_path: tuple[str, ...], tmp_path: Path):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = "bad-resource-id"
    _write_json(dlc_config, payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / script_name),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert f"must be {DEFAULT_DLC_RESOURCE_ID}" in proc.stdout


@pytest.mark.parametrize("script_name", ["qwen35_submit.sh", "qwen3_vl_submit.sh"])
@pytest.mark.parametrize("field_path", [("dlc", "data_source_uris"), ("dlc", "judge", "data_source_uris")])
def test_submitter_rejects_missing_required_nas_mount(
    script_name: str,
    field_path: tuple[str, ...],
    tmp_path: Path,
):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = MOUNT_URIS_WITHOUT_NAS
    _write_json(dlc_config, payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / script_name),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert f"must include {REQUIRED_NAS_MOUNT_URI}" in proc.stdout


@pytest.mark.parametrize("script_name", ["qwen35_submit.sh", "qwen3_vl_submit.sh"])
@pytest.mark.parametrize("field_path", [("dlc", "data_source_uris"), ("dlc", "judge", "data_source_uris")])
def test_submitter_rejects_missing_cpfsb_mount(
    script_name: str,
    field_path: tuple[str, ...],
    tmp_path: Path,
):
    if shutil.which("jq") is None:
        pytest.skip("submitter dry-run requires jq")

    lmms_root = Path(__file__).resolve().parents[2]
    dlc_config, eval_config, judge_config = _configs(tmp_path, lmms_root)
    payload = json.loads(dlc_config.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = MOUNT_URIS_WITHOUT_CPFSB
    _write_json(dlc_config, payload)
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    proc = subprocess.run(
        [
            "bash",
            str(lmms_root / "run_scripts" / script_name),
            str(dlc_config),
            str(eval_config),
            str(judge_config),
        ],
        cwd=lmms_root.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert "must include a CPFS URI mounted at /mnt/cpfsB" in proc.stdout
