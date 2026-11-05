import importlib.util
import json
import os
import subprocess
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('snapshot', ROOT / 'tools/create_runtime_snapshot.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def test_snapshot_keeps_code_when_worktree_changes_and_excludes_runtime(tmp_path):
    source, target = tmp_path / 'repo', tmp_path / 'snapshot'
    for name in ('lmms_eval', 'run_scripts', 'tools', '.venv', 'local', 'lmms_eval/tui/web/node_modules'):
        (source / name).mkdir(parents=True)
    worker = source / 'run_scripts/qwen35_worker.sh'
    worker.write_text('echo original\n')
    (source / 'lmms_eval/task.py').write_text('value = 1\n')
    (source / '.venv/secret').write_text('external')
    (source / 'local/webui_users.json').write_text('private')
    snapshot.create_snapshot(source, target)
    worker.write_text('echo changed\n')
    assert (target / 'run_scripts/qwen35_worker.sh').read_text() == 'echo original\n'
    assert not (target / '.venv').exists()
    assert not (target / 'local').exists()
    assert not (target / 'lmms_eval/tui/web').exists()
    manifest = json.loads((target / 'source_manifest.json').read_text())
    assert len(manifest['run_scripts/qwen35_worker.sh']) == 64


@pytest.mark.parametrize("script", ["qwen35_worker.sh", "run_judge.sh"])
def test_worker_python_imports_snapshot_even_when_cwd_is_live_worktree(tmp_path, script):
    live, frozen = tmp_path / "live", tmp_path / "frozen"
    live.mkdir()
    frozen.mkdir()
    for directory, value in ((live, "live"), (frozen, "frozen")):
        (directory / "snapshot_probe.py").write_text(f"print({value!r})")
    worker = (ROOT / "run_scripts" / script).read_text()
    exports = [line for line in worker.splitlines() if line.startswith("export PYTHON")]
    result = subprocess.run(
        ["bash", "-c", "\n".join(exports) + '\nexec "$SNAPSHOT_TEST_PYTHON" -m snapshot_probe'],
        cwd=live, env={**os.environ, "LMMS_EVAL_ROOT": str(frozen), "INNOVATOR_SITECUSTOMIZE": str(frozen), "SNAPSHOT_TEST_PYTHON": sys.executable},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "frozen"
