import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
WATCH = TOOLS / 'run_with_backend_watch.py'


def test_backend_exit_terminates_evaluation(tmp_path):
    backend = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.8)'])
    marker = tmp_path / 'incorrectly_finished'
    try:
        result = subprocess.run([sys.executable, str(WATCH), '--backend-pids', str(backend.pid), '--',
                                 sys.executable, '-c', f'import time; from pathlib import Path; time.sleep(20); Path({str(marker)!r}).touch()'],
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 75, result.stderr
        assert 'backend exited' in result.stderr
        assert not marker.exists()
    finally:
        backend.wait()


def test_watch_preserves_task_exit_status():
    backend = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        result = subprocess.run([sys.executable, str(WATCH), '--backend-pids', str(backend.pid), '--',
                                 sys.executable, '-c', 'raise SystemExit(7)'], timeout=10)
        assert result.returncode == 7
        assert backend.poll() is None
    finally:
        backend.terminate()
        backend.wait()


def test_memory_evidence_records_oom_counters(tmp_path):
    spec = importlib.util.spec_from_file_location('monitor', TOOLS / 'monitor_worker_resources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'memory.current').write_text('137438953472')
    (tmp_path / 'memory.events').write_text('oom 2\noom_kill 1\n')
    result = module.read_memory(tmp_path)
    assert result['memory.current'] == '137438953472'
    assert 'oom_kill 1' in result['memory.events']
    assert 'memory.max' not in result


def test_worker_aborts_remaining_tasks_when_backend_dies(tmp_path):
    worker = (TOOLS.parent / "run_scripts/qwen35_worker.sh").read_text()
    function = worker.split("run_lmms_eval() {", 1)[1].split("wait_for_eval_gpu_release() {", 1)[0]
    config = tmp_path / "config.json"
    config.write_text("{}")
    harness = "\n".join([
        'set -euo pipefail',
        f'OUTPUT_PATH={str(tmp_path)!r}',
        f'CONFIG={str(config)!r}',
        'MACHINE_RANK=1', 'MODEL_BACKEND=vllm',
        'split_lmms_tasks() { TASK_ARRAY=(first second); }',
        'run_lmms_eval_task() { echo "executed:$1"; return 75; }',
        'run_lmms_eval() {' + function,
        'rc=0; run_lmms_eval || rc=$?; exit "$rc"',
    ])
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert 'executed:first' in result.stdout
    assert 'executed:second' not in result.stdout
    assert 'aborting remaining tasks' in result.stderr


@pytest.mark.parametrize("release_rc,judge_rc", [(0, 7), (17, 0)])
def test_inline_judge_failure_reaches_worker_caller(tmp_path, release_rc, judge_rc):
    worker = (TOOLS.parent / 'run_scripts/qwen35_worker.sh').read_text()
    function = worker.split('run_inline_local_judge() {', 1)[1].split('\nJUDGE_ONLY_SOURCE_CONFIG=', 1)[0]
    script = tmp_path / 'run_judge.sh'
    script.write_text(f'#!/bin/bash\nexit {judge_rc}\n')
    harness = '\n'.join([
        'set -euo pipefail',
        'JUDGE_CONFIG=fixture WORLD_SIZE=1 RANK=0 MACHINE_RANK=0 MODEL_BACKEND=vllm',
        f'SCRIPT_DIR={str(tmp_path)!r}',
        "nvidia-smi() { printf 'gpu\\n%.0s' {1..8}; }",
        'validate_inline_judge_inputs() { return 0; }',
        'cleanup_vllm() { return 0; }',
        f'wait_for_eval_gpu_release() {{ return {release_rc}; }}',
        'run_inline_local_judge() {' + function,
        'if run_inline_local_judge; then exit 0; else exit $?; fi',
    ])
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True)
    assert result.returncode == (release_rc or judge_rc), result.stdout + result.stderr
    assert 'completed successfully' not in result.stdout


def test_http_clients_do_not_multiply_concurrency_by_gpu_count():
    worker = (TOOLS.parent / "run_scripts/qwen35_worker.sh").read_text()
    function = worker.split("compute_vllm_eval_resources() {", 1)[1].split("compute_api_resources() {", 1)[0]
    harness = "\n".join([
        'set -euo pipefail',
        'compute_resources() { LOCAL_GPU_NUM=8; NPROC_PER_NODE=8; NUM_BACKENDS=4; NUM_MACHINES=2; MACHINE_RANK=1; MODEL_TP=2; }',
        'compute_vllm_eval_resources() {' + function,
        'compute_vllm_eval_resources',
        'echo "$NPROC_PER_NODE $(( NUM_BACKENDS * MODEL_TP )) $NUM_MACHINES $MACHINE_RANK"',
    ])
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1 8 2 1'
