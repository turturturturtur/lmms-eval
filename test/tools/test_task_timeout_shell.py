import subprocess
from pathlib import Path


LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2]
EVAL_COMMON = LMMS_EVAL_ROOT / "run_scripts" / "eval_common.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=LMMS_EVAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_classify_lmms_eval_task_status():
    result = run_bash(
        f"""
        set -euo pipefail
        source {EVAL_COMMON}
        classify_lmms_eval_task_status 0 9
        printf '\\n'
        classify_lmms_eval_task_status 124 9
        printf '\\n'
        classify_lmms_eval_task_status 7 9
        printf '\\n'
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "success\tcompleted",
        "timeout\ttimeout_after_9s",
        "failed\texit_code_7",
    ]


def test_timeout_wrapper_continues_after_timed_out_command():
    result = run_bash(
        f"""
        set -euo pipefail
        source {EVAL_COMMON}
        TASK_TIMEOUT_SECONDS=1
        TASK_TIMEOUT_KILL_AFTER_SECONDS=1
        statuses=()
        for command in "sleep 5" "true"; do
            set +e
            timeout --signal=TERM --kill-after="${{TASK_TIMEOUT_KILL_AFTER_SECONDS}}s" "${{TASK_TIMEOUT_SECONDS}}s" bash -c "${{command}}"
            rc=$?
            set -e
            classification="$(classify_lmms_eval_task_status "${{rc}}" "${{TASK_TIMEOUT_SECONDS}}")"
            statuses+=("${{classification%%$'\\t'*}}")
        done
        printf '%s\\n' "${{statuses[@]}}"
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["timeout", "success"]


def test_load_config_rejects_missing_task_timeout(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "env": {
            "hf_home": "/tmp/hf",
            "hf_token": "",
            "venv_path": "/tmp/venv",
            "hf_datasets_offline": true,
            "transformers_offline": true
          },
          "log": {"dir": "/tmp/logs"},
          "distributed": {
            "master_addr": "127.0.0.1",
            "master_port": 23456,
            "world_size": 1,
            "rank": 0
          },
          "model": {
            "path": "/tmp/model",
            "tp": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 8,
            "base_port": 8000
          },
          "eval": {
            "tasks": "fake",
            "output_path": "/tmp/output",
            "concurrency": 1,
            "gen_kwargs": "max_new_tokens=16,max_pixels=1024",
            "limit": 1,
            "task_timeout_seconds": 10
          }
        }
        """,
        encoding="utf-8",
    )

    result = run_bash(
        f"""
        set -euo pipefail
        source {EVAL_COMMON}
        load_config {config_path} ""
        """
    )

    assert result.returncode == 2
    assert "eval.task_timeout_kill_after_seconds" in result.stderr


def test_load_config_accepts_empty_gen_kwargs_with_explicit_timeouts(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "env": {
            "hf_home": "/tmp/hf",
            "hf_token": "",
            "venv_path": "/tmp/venv",
            "hf_datasets_offline": true,
            "transformers_offline": true
          },
          "log": {"dir": "/tmp/logs"},
          "distributed": {
            "master_addr": "127.0.0.1",
            "master_port": 23456,
            "world_size": 1,
            "rank": 0
          },
          "model": {
            "path": "/tmp/model",
            "tp": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 8,
            "base_port": 8000
          },
          "eval": {
            "tasks": "fake",
            "output_path": "/tmp/output",
            "concurrency": 1,
            "gen_kwargs": "",
            "limit": 1,
            "task_timeout_seconds": 10,
            "task_timeout_kill_after_seconds": 2
          }
        }
        """,
        encoding="utf-8",
    )

    result = run_bash(
        f"""
        set -euo pipefail
        source {EVAL_COMMON}
        load_config {config_path} ""
        printf '%s %s %s %s\\n' "${{MAX_NEW_TOKENS}}" "${{MAX_PIXELS}}" "${{TASK_TIMEOUT_SECONDS}}" "${{TASK_TIMEOUT_KILL_AFTER_SECONDS}}"
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "32768 4014080 10 2"
