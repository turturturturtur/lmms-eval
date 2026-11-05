import asyncio
import os
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from lmms_eval.entrypoints.job_scheduler import JobScheduler
from lmms_eval.entrypoints.protocol import EvaluateRequest, JobStatus


class TestEvaluateRequestTimeouts(unittest.TestCase):
    def test_requires_explicit_positive_timeout_fields(self):
        with self.assertRaises(ValidationError):
            EvaluateRequest(model="fake", tasks=["task"])

        with self.assertRaises(ValidationError):
            EvaluateRequest(
                model="fake",
                tasks=["task"],
                timeout_seconds=0,
                timeout_kill_after_seconds=1,
            )

        with self.assertRaises(ValidationError):
            EvaluateRequest(
                model="fake",
                tasks=["task"],
                timeout_seconds="10",
                timeout_kill_after_seconds=1,
            )

        request = EvaluateRequest(
            model="fake",
            tasks=["task"],
            timeout_seconds=10,
            timeout_kill_after_seconds=1,
        )
        self.assertEqual(request.timeout_seconds, 10)
        self.assertEqual(request.timeout_kill_after_seconds, 1)


class TestJobSchedulerTimeouts(unittest.TestCase):
    def test_run_evaluation_times_out_and_can_run_later_job(self):
        async def run_case():
            with tempfile.TemporaryDirectory() as tmpdir:
                fake_bin = Path(tmpdir) / "bin"
                fake_bin.mkdir()
                fake_launcher = fake_bin / "fake_eval_launcher"
                fake_launcher.write_text(
                    textwrap.dedent(
                        """\
                        #!/usr/bin/env bash
                        set -euo pipefail
                        if [[ "${FAKE_LAUNCHER_MODE}" == "sleep" ]]; then
                            sleep 30
                        elif [[ "${FAKE_LAUNCHER_MODE}" == "success" ]]; then
                            mkdir -p "$1/fake"
                            echo '{}' > "$1/fake/20260101_000000_results.json"
                            exit 0
                        else
                            echo "unexpected FAKE_LAUNCHER_MODE=${FAKE_LAUNCHER_MODE}" >&2
                            exit 99
                        fi
                        """
                    ),
                    encoding="utf-8",
                )
                fake_launcher.chmod(0o755)

                old_mode = os.environ.get("FAKE_LAUNCHER_MODE")
                try:
                    scheduler = JobScheduler()
                    # _build_eval_cmd invokes `python -m lmms_eval` directly for a
                    # single GPU instead of the accelerate launcher, so pin the
                    # command shape here and exercise the timeout logic itself.
                    with patch.object(
                        JobScheduler,
                        "_build_eval_cmd",
                        staticmethod(lambda config, output_path: [str(fake_launcher), str(output_path)]),
                    ):
                        timeout_config = {
                            "model": "fake",
                            "tasks": ["hang"],
                            "num_gpus": 1,
                            "output_dir": str(Path(tmpdir) / "timeout_output"),
                            "timeout_seconds": 1,
                            "timeout_kill_after_seconds": 1,
                        }

                        os.environ["FAKE_LAUNCHER_MODE"] = "sleep"
                        started = time.monotonic()
                        with self.assertRaises(TimeoutError):
                            await scheduler._run_evaluation(timeout_config)
                        self.assertLess(time.monotonic() - started, 5)

                        success_config = dict(timeout_config)
                        success_config["output_dir"] = str(Path(tmpdir) / "success_output")
                        success_config["timeout_seconds"] = 5
                        os.environ["FAKE_LAUNCHER_MODE"] = "success"
                        self.assertIn("fake", await scheduler._run_evaluation(success_config))
                finally:
                    if old_mode is None:
                        os.environ.pop("FAKE_LAUNCHER_MODE", None)
                    else:
                        os.environ["FAKE_LAUNCHER_MODE"] = old_mode

        asyncio.run(run_case())

    def test_worker_continues_after_failed_job(self):
        class StubScheduler(JobScheduler):
            async def _run_evaluation(self, config: dict) -> dict:
                if config["model"] == "fail":
                    raise TimeoutError("forced timeout")
                return {"ok": {"results": "done"}}

        async def run_case():
            scheduler = StubScheduler()
            await scheduler.start()
            try:
                first = EvaluateRequest(
                    model="fail",
                    tasks=["first"],
                    timeout_seconds=1,
                    timeout_kill_after_seconds=1,
                )
                second = EvaluateRequest(
                    model="success",
                    tasks=["second"],
                    timeout_seconds=1,
                    timeout_kill_after_seconds=1,
                )
                first_id, _ = await scheduler.add_job(first)
                second_id, _ = await scheduler.add_job(second)
                await asyncio.wait_for(scheduler._job_queue.join(), timeout=5)

                first_job = await scheduler.get_job(first_id)
                second_job = await scheduler.get_job(second_id)
                assert first_job is not None
                assert second_job is not None
                self.assertEqual(first_job.status, JobStatus.FAILED)
                self.assertIn("forced timeout", first_job.error or "")
                self.assertEqual(second_job.status, JobStatus.COMPLETED)
            finally:
                await scheduler.stop()

        asyncio.run(run_case())


if __name__ == "__main__":
    unittest.main()
