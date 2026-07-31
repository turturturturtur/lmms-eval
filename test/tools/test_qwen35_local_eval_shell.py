import os
import subprocess
from pathlib import Path


LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
LOCAL_WRAPPER = LMMS_EVAL_ROOT / "run_scripts" / "qwen35_local_eval.sh"


def test_qwen35_local_wrapper_dry_run_uses_strict_cli_compatibility(tmp_path: Path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    output = tmp_path / "output"
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(LOCAL_WRAPPER),
            str(model),
            "ai2d,ocrbench",
            str(output),
            "1",
            "off",
        ],
        cwd=LMMS_EVAL_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model_processor_compat required" in result.stdout
    assert f"--model_view_root {output}/model_views" in result.stdout
    assert f"model={model}" in result.stdout
    assert "enable_thinking=False" in result.stdout
    assert "--tasks ai2d\\,ocrbench" in result.stdout


def test_qwen35_local_wrapper_rejects_invalid_thinking_mode(tmp_path: Path):
    model = tmp_path / "checkpoint"
    model.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(LOCAL_WRAPPER),
            str(model),
            "ai2d",
            str(tmp_path / "output"),
            "1",
            "maybe",
        ],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "THINKING must be on or off" in result.stderr
