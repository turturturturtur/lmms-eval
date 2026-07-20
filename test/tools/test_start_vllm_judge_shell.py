import os
import re
import subprocess
from pathlib import Path


LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
START_VLLM_JUDGE = LMMS_EVAL_ROOT / "tools" / "start_vllm_judge.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_start_vllm_judge_uses_all_eight_visible_gpus_and_qwen35_runtime_flags(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    model_dir = tmp_path / "Qwen3.5-9B"
    model_dir.mkdir()
    capture_path = tmp_path / "launch.txt"
    ready_path = tmp_path / "ready"
    log_path = tmp_path / "judge.log"

    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "for gpu in {0..7}; do echo \"GPU ${gpu}: fake\"; done\n",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        "if [[ -f \"${READY_FILE}\" ]]; then printf '200'; else printf '503'; fi\n",
    )
    _write_executable(
        fake_bin / "setsid",
        "#!/usr/bin/env bash\n"
        "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"${CUDA_VISIBLE_DEVICES:-}\" > \"${CAPTURE_FILE}\"\n"
        "printf '%s\\n' \"$*\" >> \"${CAPTURE_FILE}\"\n"
        "touch \"${READY_FILE}\"\n"
        "sleep 30\n",
    )

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_FILE": str(capture_path),
            "READY_FILE": str(ready_path),
        }
    )
    proc = subprocess.run(
        [
            "bash",
            str(START_VLLM_JUDGE),
            "--model-path",
            str(model_dir),
            "--served-model-name",
            "Qwen3.5-9B",
            "--tp",
            "8",
            "--max-model-len",
            "40960",
            "--gpu-memory-utilization",
            "0.88",
            "--max-num-seqs",
            "192",
            "--port",
            "8002",
            "--log",
            str(log_path),
        ],
        cwd=LMMS_EVAL_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )

    assert proc.returncode == 0, proc.stdout
    assert "CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7" in proc.stdout
    assert "VLLM_OWNED=1" in proc.stdout
    launched = capture_path.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in launched
    assert "--tensor-parallel-size 8" in launched
    assert "--max-model-len 40960" in launched
    assert "--gpu-memory-utilization 0.88" in launched
    assert "--max-num-seqs 192" in launched
    assert "--attention-backend FLASHINFER" in launched
    assert "--mm-encoder-tp-mode data" in launched
    assert "--enforce-eager" in launched

    pid_match = re.search(r"^VLLM_PID=(\d+)$", proc.stdout, re.MULTILINE)
    assert pid_match is not None
    os.kill(int(pid_match.group(1)), 9)


def test_start_vllm_judge_rejects_tp_larger_than_visible_gpu_count(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    model_dir = tmp_path / "Qwen3.5-9B"
    model_dir.mkdir()
    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "for gpu in {0..3}; do echo \"GPU ${gpu}: fake\"; done\n",
    )

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    proc = subprocess.run(
        [
            "bash",
            str(START_VLLM_JUDGE),
            "--model-path",
            str(model_dir),
            "--tp",
            "8",
            "--log",
            str(tmp_path / "judge.log"),
        ],
        cwd=LMMS_EVAL_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode != 0
    assert "Not enough visible GPUs for TP=8: 4 available" in proc.stdout
