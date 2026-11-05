"""The real judge script must reach scoring after a metadata-only launcher."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("timeouts,expected", [({}, "300 3 120"), ({"datapoint_timeout_seconds": 600, "datapoint_max_attempts": 2, "request_timeout_seconds": 180}, "600 2 180")])
def test_successful_local_judge_launcher_reaches_scoring(tmp_path, timeouts, expected):
    if not Path("/mnt/nasA/tianleniu/evaluation_cache/lmms_eval").is_dir():
        pytest.skip("the worker storage contract requires the NAS benchmark mount")
    for directory in ("run_scripts", "tools", "runtime/bin", "runtime/conda-meta", "bin", "model", "samples"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for script in ("run_judge.sh", "activate_runtime.sh"):
        shutil.copy2(ROOT / "run_scripts" / script, tmp_path / "run_scripts" / script)
    marker = tmp_path / "judge.marker"
    python = tmp_path / "runtime/bin/python"
    python.write_text('#!/bin/bash\nif [[ "$1" == "-m" ]]; then echo "$JUDGE_DATAPOINT_TIMEOUT $JUDGE_DATAPOINT_MAX_ATTEMPTS $JUDGE_TIMEOUT" > "$JUDGE_TEST_MARKER"; fi\nexit 0\n')
    python.chmod(0o755)
    curl = tmp_path / "bin/curl"
    curl.write_text("#!/bin/bash\nprintf 000\n")
    curl.chmod(0o755)
    # Launch a harmless owned process, so the real cleanup also gets exercised.
    launcher = tmp_path / "tools/start_vllm_judge.sh"
    launcher.write_text('#!/bin/bash\nsleep 60 >/dev/null 2>&1 &\necho "VLLM_PID=$!"\necho "VLLM_OWNED=1"\n')
    config = {
        "env": {"venv_path": str(tmp_path / "runtime"),
                "lmms_eval_datasets_cache": "/mnt/nasA/tianleniu/evaluation_cache/lmms_eval",
                "hf_datasets_cache": "/mnt/nasA/tianleniu/evaluation_cache/lmms_eval"},
        "log": {"dir": str(tmp_path / "logs")},
        "judge": {"backend": "vllm", "parallel": 1, "model": "fixture",
                  "vllm": {"model_path": str(tmp_path / "model"), "processor_compat": "off",
                           "tp": 1, "max_model_len": 128, "max_num_seqs": 1, "port": 18002}},
        "eval": {"tasks": "ai2d", "input_result_path": str(tmp_path / "samples"),
                 "output_path": str(tmp_path / "results"), "debug": False},
    }
    config_path = tmp_path / "config.json"
    config["judge"].update(timeouts)
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "JUDGE_TEST_MARKER": str(marker)}
    result = subprocess.run(["bash", str(tmp_path / "run_scripts/run_judge.sh"), str(config_path)],
                            capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
    assert marker.read_text().strip() == expected
    assert "Judge completed successfully" in result.stdout
