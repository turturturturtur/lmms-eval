"""Exercise the worker's actual main branch without starting inference/GPU servers."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prepare_rc,judge_rc", [(0, 0), (0, 7), (2, 0)])
def test_judge_retry_skips_inference_and_propagates_failures(tmp_path, prepare_rc, judge_rc):
    runtime = tmp_path / "runtime/bin"
    runtime.mkdir(parents=True)
    python = runtime / "python"
    python.write_text(f'#!/bin/bash\nif [[ "$1" == "-m" ]]; then echo prepare; exit {prepare_rc}; fi\nexit 0\n')
    python.chmod(0o755)
    worker = (ROOT / "run_scripts/qwen35_worker.sh").read_text()
    main = 'JUDGE_ONLY_SOURCE_CONFIG=' + worker.split('JUDGE_ONLY_SOURCE_CONFIG=', 1)[1]
    harness = '\n'.join([
        'set -euo pipefail',
        'cfg() { echo /source/runtime.json; }',
        'compute_api_resources() { :; }',
        "nvidia-smi() { printf 'gpu\\n%.0s' {1..8}; }",
        'setup_logging() { :; }; ensure_venv() { :; }; prepend_pythonpath_bins() { :; }; setup_native_libs() { :; }',
        f'run_inline_local_judge() {{ echo judge; return {judge_rc}; }}',
        'launch_vllm_backends() { echo UNEXPECTED_INFERENCE; exit 99; }',
        main,
    ])
    env = {**os.environ, 'VENV_PATH': str(runtime.parent), 'CONFIG': '/new/runtime.json',
           'JUDGE_CONFIG': 'judge.json', 'LOG_DIR': str(tmp_path), 'LMMS_EVAL_ROOT': str(ROOT)}
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == (prepare_rc or judge_rc), result.stdout + result.stderr
    assert 'UNEXPECTED_INFERENCE' not in result.stdout
    assert ('\njudge\n' in result.stdout) == (prepare_rc == 0)
