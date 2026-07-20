import json

from lmms_eval.tui import server


WRAPPED_DLC_JOBS = """
+--------------------------------+------------------+-------------+-----------+
|              Name              |      JobId       | WorkspaceId | JobStatus |
+--------------------------------+------------------+-------------+-----------+
| eval_qwen35_9b_lllllllllllllll | dlc17isyru6cven2 | 240810      | Stopped   |
| llllllllllllllllll             |                  |             |           |
+--------------------------------+------------------+-------------+-----------+
"""


def test_parse_dlc_table_merges_wrapped_job_name_rows():
    rows = server._parse_dlc_table(WRAPPED_DLC_JOBS)

    assert len(rows) == 1
    assert rows[0]["JobId"] == "dlc17isyru6cven2"
    assert rows[0]["Name"] == "eval_qwen35_9b_lllllllllllllllllllllllllllllllll"


def test_list_dlc_jobs_from_cli_returns_full_wrapped_name(monkeypatch):
    server._dlc_jobs_cache.clear()
    monkeypatch.setattr(server, "_run_dlc_command", lambda *_args, **_kwargs: WRAPPED_DLC_JOBS)
    monkeypatch.setattr(server, "_paistudio_user_name_map", lambda: {})
    monkeypatch.setattr(server, "_aiworkspace_member_name_map", lambda: {})

    rows = server._list_dlc_jobs_from_cli(page_size=5, max_pages=1, status="", display_name="eval_")

    assert len(rows) == 1
    assert rows[0]["job_id"] == "dlc17isyru6cven2"
    assert rows[0]["name"] == "eval_qwen35_9b_lllllllllllllllllllllllllllllllll"


def test_view_logs_metric_rows_keep_ordinary_and_judge_results_together(tmp_path):
    result_root = tmp_path / "result"
    result_root.mkdir()
    ordinary_result = result_root / "20260715_120000_results.json"
    ordinary_result.write_text(
        json.dumps(
            {
                "results": {"ocrbench": {"ocrbench_accuracy,none": 0.5}},
                "n-samples": {"ocrbench": {"effective": 1, "original": 1}},
            }
        ),
        encoding="utf-8",
    )

    judge_dir = result_root / "judge"
    judge_dir.mkdir()
    judged_samples = judge_dir / "20260715_120100_samples_ocrbench.jsonl"
    judged_samples.write_text(
        json.dumps(
            {
                "doc_id": 0,
                "judge_mode": "llm_judge",
                "metrics": {
                    "ocrbench_accuracy": {
                        "score": 1,
                        "llm_judge_score": 1,
                        "llm_judge_success": True,
                    },
                    "llm_judge_score": 1,
                    "llm_judge_success": True,
                    "llm_judge_failed": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, summary_files = server._build_metric_rows(result_root)

    assert summary_files == []
    assert len(rows) == 2
    ordinary_rows = [row for row in rows if row.result_json == str(ordinary_result)]
    judge_rows = [row for row in rows if row.value_source == "judged_samples"]
    assert len(ordinary_rows) == 1
    assert ordinary_rows[0].metric_name == "ocrbench_accuracy"
    assert ordinary_rows[0].value == 0.5
    assert len(judge_rows) == 1
    assert judge_rows[0].metric_name == "ocrbench_accuracy.score"
    assert judge_rows[0].value == 1
    assert judge_rows[0].sample_jsonls == [str(judged_samples)]


def test_view_logs_lists_eval_and_judge_rows_together_and_deduplicates_job_ids(monkeypatch):
    server._dlc_jobs_cache.clear()
    requested_filters: list[str] = []

    def fake_run(args, **_kwargs):
        display_filter = args[args.index("--display_name") + 1]
        requested_filters.append(display_filter)
        return display_filter

    def fake_parse(output):
        if output == "eval_":
            return [
                {
                    "Name": "eval_regular",
                    "JobId": "dlceval",
                    "WorkspaceId": "240810",
                    "JobStatus": "Succeeded",
                },
                {
                    "Name": "eval_duplicate",
                    "JobId": "dlcshared",
                    "WorkspaceId": "240810",
                    "JobStatus": "Succeeded",
                },
            ]
        if output == "judge_":
            return [
                {
                    "Name": "judge_regular",
                    "JobId": "dlcjudge",
                    "WorkspaceId": "240810",
                    "JobStatus": "Succeeded",
                },
                {
                    "Name": "judge_duplicate",
                    "JobId": "dlcshared",
                    "WorkspaceId": "240810",
                    "JobStatus": "Succeeded",
                },
            ]
        raise AssertionError(f"Unexpected display filter: {output}")

    monkeypatch.setattr(server, "_run_dlc_command", fake_run)
    monkeypatch.setattr(server, "_parse_dlc_table", fake_parse)
    monkeypatch.setattr(server, "_paistudio_user_name_map", lambda: {})
    monkeypatch.setattr(server, "_aiworkspace_member_name_map", lambda: {})

    rows = server._list_dlc_jobs_from_cli(
        page_size=5,
        max_pages=1,
        status="",
        display_name="eval_,judge_",
    )

    assert requested_filters == ["eval_", "judge_"]
    assert [row["job_id"] for row in rows] == ["dlceval", "dlcshared", "dlcjudge"]
    assert [row["name"] for row in rows] == ["eval_regular", "eval_duplicate", "judge_regular"]
    assert [row["job_stage"] for row in rows] == ["eval", "eval", "judge"]
