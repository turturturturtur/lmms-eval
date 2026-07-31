import subprocess
from pathlib import Path


LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
EVAL_COMMON = LMMS_EVAL_ROOT / "run_scripts" / "eval_common.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_models_curl(path: Path, model_id: str) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
output=""
while (( $# > 0 )); do
  case "$1" in
    -o) output="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\\n' '{{"data":[{{"id":"{model_id}"}}]}}' > "${{output}}"
printf '200'
""",
    )


def test_wait_for_backends_reports_dead_owned_process_without_waiting_for_timeout(
    tmp_path: Path,
):
    backend_log = tmp_path / "backend.log"
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        MACHINE_RANK=0
        MODEL_NAME=expected-model
        MODEL_STARTUP_TIMEOUT_SECONDS=1800
        BACKEND_URLS=http://127.0.0.1:9/v1
        BACKEND_LOGS=({backend_log})
        bash -c 'echo processor-initialization-failed; exit 23' > {backend_log} 2>&1 &
        PIDS=($!)
        sleep 0.1
        if wait_for_backends; then
            echo unexpected-success >&2
            exit 90
        fi
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "exited before readiness" in combined
    assert "exit_code=23" in combined
    assert "processor-initialization-failed" in combined


def test_wait_for_backends_rejects_foreign_model_identity(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_models_curl(fake_bin / "curl", "foreign-model")
    backend_log = tmp_path / "backend.log"
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        export PATH={fake_bin}:$PATH
        MACHINE_RANK=0
        MODEL_NAME=expected-model
        MODEL_STARTUP_TIMEOUT_SECONDS=30
        BACKEND_URLS=http://127.0.0.1:8941/v1
        BACKEND_LOGS=({backend_log})
        sleep 30 > {backend_log} 2>&1 &
        PIDS=($!)
        if wait_for_backends; then
            rc=90
        else
            rc=$?
        fi
        kill "${{PIDS[0]}}" 2>/dev/null || true
        wait "${{PIDS[0]}}" 2>/dev/null || true
        exit "${{rc}}"
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "model identity mismatch" in combined
    assert "expected=expected-model" in combined
    assert 'observed=["foreign-model"]' in combined


def test_wait_for_backends_accepts_exact_model_identity(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_models_curl(fake_bin / "curl", "expected-model")
    backend_log = tmp_path / "backend.log"
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        export PATH={fake_bin}:$PATH
        MACHINE_RANK=0
        MODEL_NAME=expected-model
        MODEL_STARTUP_TIMEOUT_SECONDS=30
        BACKEND_URLS=http://127.0.0.1:8941/v1
        BACKEND_LOGS=({backend_log})
        sleep 30 > {backend_log} 2>&1 &
        PIDS=($!)
        wait_for_backends
        kill "${{PIDS[0]}}" 2>/dev/null || true
        wait "${{PIDS[0]}}" 2>/dev/null || true
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 0, result.stderr
    assert "Ready: http://127.0.0.1:8941/v1 model=expected-model" in result.stdout


def test_wait_for_backends_rejects_previously_ready_backend_death(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    second_count = tmp_path / "second-count"
    curl_script = f"""#!/usr/bin/env bash
set -euo pipefail
output=""
url=""
while (( $# > 0 )); do
  case "$1" in
    -o) output="$2"; shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
if [[ "${{url}}" == *":8941/"* ]]; then
  printf '%s\\n' '{{"data":[{{"id":"expected-model"}}]}}' > "${{output}}"
  printf '200'
  exit 0
fi
count=0
[[ -f {second_count} ]] && count="$(cat {second_count})"
count=$((count + 1))
printf '%s' "${{count}}" > {second_count}
if (( count == 1 )); then
  : > "${{output}}"
  printf '503'
else
  printf '%s\\n' '{{"data":[{{"id":"expected-model"}}]}}' > "${{output}}"
  printf '200'
fi
"""
    _write_executable(fake_bin / "curl", curl_script)
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        export PATH={fake_bin}:$PATH
        export LMMS_EVAL_BACKEND_POLL_SECONDS=1
        MACHINE_RANK=0
        MODEL_NAME=expected-model
        MODEL_STARTUP_TIMEOUT_SECONDS=30
        BACKEND_URLS='http://127.0.0.1:8941/v1;http://127.0.0.1:8942/v1'
        BACKEND_LOGS=({first_log} {second_log})
        bash -c 'sleep 0.3; echo first-backend-exited' > {first_log} 2>&1 &
        first=$!
        sleep 30 > {second_log} 2>&1 &
        second=$!
        PIDS=("${{first}}" "${{second}}")
        if wait_for_backends; then
            rc=90
        else
            rc=$?
        fi
        kill "${{second}}" 2>/dev/null || true
        wait "${{second}}" 2>/dev/null || true
        exit "${{rc}}"
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "exited before readiness" in combined
    assert "first-backend-exited" in combined


def test_cleanup_stops_only_owned_backend_pid(tmp_path: Path):
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        MACHINE_RANK=0
        DEBUG=false
        LMMS_EVAL_CLEANUP_GRACE_SECONDS=0
        setsid sleep 30 &
        owned=$!
        sleep 30 &
        foreign=$!
        PIDS=("${{owned}}")
        BACKEND_LOGS=({tmp_path / "owned.log"})
        cleanup_vllm
        if kill -0 "${{foreign}}" 2>/dev/null; then
            echo foreign-alive
        else
            echo foreign-killed
            exit 91
        fi
        kill "${{foreign}}"
        wait "${{foreign}}" 2>/dev/null || true
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 0, result.stderr
    assert "foreign-alive" in result.stdout


def test_cleanup_stops_owned_backend_process_group_children(tmp_path: Path):
    child_file = tmp_path / "child.pid"
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        MACHINE_RANK=0
        DEBUG=false
        LMMS_EVAL_CLEANUP_GRACE_SECONDS=1
        setsid bash -c 'sleep 30 & child=$!; echo "$child" > {child_file}; wait' &
        owned_group=$!
        for _ in {{1..20}}; do
            [[ -s {child_file} ]] && break
            sleep 0.05
        done
        child=$(cat {child_file})
        sleep 30 &
        foreign=$!
        PIDS=("${{owned_group}}")
        BACKEND_LOGS=({tmp_path / "owned.log"})
        cleanup_vllm
        child_state="$(ps -o stat= -p "${{child}}" 2>/dev/null | tr -d ' ' || true)"
        if kill -0 "${{child}}" 2>/dev/null && [[ "${{child_state}}" != Z* ]]; then
            echo owned-child-alive
            kill "${{child}}" 2>/dev/null || true
            kill "${{foreign}}" 2>/dev/null || true
            exit 92
        fi
        if ! kill -0 "${{foreign}}" 2>/dev/null; then
            echo foreign-killed
            exit 93
        fi
        echo owned-group-dead-foreign-alive
        kill "${{foreign}}"
        wait "${{foreign}}" 2>/dev/null || true
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "owned-group-dead-foreign-alive" in result.stdout


def test_term_trap_exits_instead_of_continuing(tmp_path: Path):
    script = f"""
        set -euo pipefail
        source {EVAL_COMMON}
        MACHINE_RANK=0
        DEBUG=false
        LMMS_EVAL_CLEANUP_GRACE_SECONDS=0
        PIDS=()
        BACKEND_LOGS=()
        setup_cleanup_trap
        kill -TERM $$
        echo continued-after-term
    """

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=4,
    )

    assert result.returncode == 143
    assert "continued-after-term" not in result.stdout
