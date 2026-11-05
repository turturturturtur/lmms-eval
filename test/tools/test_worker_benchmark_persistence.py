"""Run the worker's real task loop without GPUs or remote API calls."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_each_benchmark_is_saved_before_the_next_and_survives_failure(tmp_path):
    runtime = tmp_path / "runtime/bin"
    runtime.mkdir(parents=True)
    fake_python = runtime / "python"
    fake_python.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv
task = args[args.index('--tasks') + 1]
output = Path(args[args.index('--output_path') + 1])
assert '--log_samples' in args
if task == 'second':
    first = Path(os.environ['OUTPUT_PATH']) / 'tasks/first'
    assert (first / 'run_results.json').is_file(), 'first result not yet saved'
    assert (first / 'run_samples_first.jsonl').is_file(), 'first samples not yet saved'
    (output / 'observed_first_on_disk').touch()
(output / f'run_raw_{task}_rank0.jsonl').write_text(json.dumps({'resps': ['A']}) + '\\n')
(output / f'run_samples_{task}.jsonl').write_text(json.dumps({'resps': ['A']}) + '\\n')
(output / 'run_results.json').write_text(json.dumps({'results': {task: {'accuracy': 1}}}))
sys.exit(3 if task == 'second' else 0)
''')
    fake_python.chmod(0o755)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"eval": {"tasks": "first,second"}}))
    logs = tmp_path / "logs"
    logs.mkdir()
    output = tmp_path / "output"
    worker = (ROOT / "run_scripts/qwen35_worker.sh").read_text()
    task_slug = "task_slug() {" + worker.split("task_slug() {", 1)[1].split("\n}\n", 1)[0] + "\n}"
    manifest = "write_task_manifest_row() {" + worker.split("write_task_manifest_row() {", 1)[1].split("\n}\n", 1)[0] + "\n}"
    task_loop = "run_lmms_eval_task() {" + worker.split("run_lmms_eval_task() {", 1)[1].split("wait_for_eval_gpu_release() {", 1)[0]
    harness = "\n".join([
        f"source {shlex.quote(str(ROOT / 'run_scripts/eval_common.sh'))}",
        "MACHINE_RANK=0 RANK=0 MASTER_PORT=19000 MODEL_BACKEND=openai",
        "SYSTEM_INSTRUCTION='' GEN_KWARGS='' BATCH_SIZE=1 VERBOSITY=INFO LIMIT=1",
        "TASK_TIMEOUT_SECONDS=10 TASK_TIMEOUT_KILL_AFTER_SECONDS=1",
        "split_lmms_tasks() { TASK_ARRAY=(first second); }",
        "build_openai_model_args() { echo fixture; }",
        task_slug, manifest, task_loop,
        "rc=0; run_lmms_eval || rc=$?; exit $rc",
    ])
    env = {**os.environ, "OUTPUT_PATH": str(output), "VENV_PATH": str(runtime.parent),
           "CONFIG": str(config), "LOG_DIR": str(logs)}
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (output / "tasks/second/observed_first_on_disk").exists(), result.stdout + result.stderr
    assert (output / "tasks/first/run_results.json").exists()
    assert not (output / "tasks/second/run_results.json").exists()
    for task in ("first", "second"):
        assert json.loads((output / f"tasks/{task}/run_samples_{task}.jsonl").read_text())["resps"] == ["A"]
        assert (output / f"tasks/{task}/run_raw_{task}_rank0.jsonl").exists()
    manifest_rows = (output / "task_status.tsv").read_text().splitlines()
    assert manifest_rows[1].startswith("first\tsuccess\t")
    assert manifest_rows[2].startswith("second\tfailed\t")
