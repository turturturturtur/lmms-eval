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
