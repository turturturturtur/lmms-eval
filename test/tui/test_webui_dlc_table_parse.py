import json

from lmms_eval.tui import server


WRAPPED_DLC_JOBS = """
+--------------------------------+------------------+-------------+-----------+
|              Name              |      JobId       | WorkspaceId | JobStatus |
+--------------------------------+------------------+-------------+-----------+
| eval_qwen35_9b_lllllllllllllll | dlc17isyru6cven2 | 247143      | Stopped   |
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


def test_list_dlc_jobs_from_cli_passes_explicit_history_window(monkeypatch):
    server._dlc_jobs_cache.clear()
    commands: list[list[str]] = []

    def fake_run(args, **_kwargs):
        commands.append(args)
        return WRAPPED_DLC_JOBS

    monkeypatch.setattr(server, "_run_dlc_command", fake_run)
    monkeypatch.setattr(server, "_paistudio_user_name_map", lambda: {})
    monkeypatch.setattr(server, "_aiworkspace_member_name_map", lambda: {})

    server._list_dlc_jobs_from_cli(
        page_size=5,
        max_pages=1,
        status="",
        display_name="eval_",
        start_time="2026-06-28T12:00:00Z",
        end_time="2026-07-28T12:00:00Z",
    )

    assert len(commands) == 1
    args = commands[0]
    assert args[args.index("--start_time") + 1] == "2026-06-28T12:00:00Z"
    assert args[args.index("--end_time") + 1] == "2026-07-28T12:00:00Z"


def test_view_logs_judge_result_replaces_provisional_eval_score(tmp_path):
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
    assert len(rows) == 1
    ordinary_rows = [row for row in rows if row.result_json == str(ordinary_result)]
    judge_rows = [row for row in rows if row.value_source == "judged_samples"]
    assert ordinary_rows == []
    assert len(judge_rows) == 1
    assert judge_rows[0].metric_name == "ocrbench_accuracy.score"
    assert judge_rows[0].value is None
    assert judge_rows[0].status == "pending_judge"
    assert judge_rows[0].sample_jsonls == [str(judged_samples)]


def test_generation_placeholder_waits_for_real_judge_score(tmp_path):
    task = "mmmu_pro_standard_reasoning_qwen3_official"
    original = tmp_path / "20261005_010101_results.json"
    original.write_text(json.dumps({"results": {task: {"mmmu_pro_acc_official,none": {
        "mode": "generation_only", "overall_accuracy": None}}}}))
    rows, _ = server._build_metric_rows(tmp_path)
    assert rows[0].status == "pending_judge"
    assert rows[0].value is None

    judged = tmp_path / "judge"
    judged.mkdir()
    (judged / "20261005_020202_results.json").write_text(json.dumps({
        "results": {task: {"llm_judge_success": 1.0, "total_acc": 0.5}},
        "judge": {"mode": "judge"},
    }))
    rows, _ = server._build_metric_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].status == "success"
    assert rows[0].metric_name == "total_acc"
    assert rows[0].value == 0.5


def test_marked_generation_score_is_not_a_final_zero(tmp_path):
    path = tmp_path / "20261005_010101_results.json"
    path.write_text(json.dumps({"execution_stage": "generation",
        "results": {"mmbench_en_dev": {"gpt_eval_score,none": 0.0}}}))
    rows, _ = server._build_metric_rows(tmp_path)
    assert rows[0].status == "pending_judge"
    assert rows[0].value is None


def test_judge_tasks_with_empty_generation_metrics_stay_visible(tmp_path):
    path = tmp_path / "20261005_010101_results.json"
    path.write_text(json.dumps({"execution_stage": "generation", "results": {
        "simplevqa": {"simplevqa_judge,none": []},
        "sfe-en": {"exact_match,none": []},
    }}))
    rows, _ = server._build_metric_rows(tmp_path)
    assert {row.lmms_tasks for row in rows} == {"simplevqa", "sfe-en"}
    assert all(row.status == "pending_judge" and row.value is None for row in rows)


def test_metric_link_stays_bound_to_its_task_when_new_results_arrive(tmp_path):
    (tmp_path / "20261005_020202_results.json").write_text(json.dumps({
        "results": {"chartqa": {"accuracy,none": 0.5}}}))
    before, _ = server._build_metric_rows(tmp_path)
    (tmp_path / "20261005_010101_results.json").write_text(json.dumps({
        "results": {"ai2d": {"accuracy,none": 1.0}}}))
    after, _ = server._build_metric_rows(tmp_path)
    selected = next(row for row in after if row.metric_id == before[0].metric_id)
    assert selected.lmms_tasks == "chartqa"


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
                    "WorkspaceId": "247143",
                    "JobStatus": "Succeeded",
                },
                {
                    "Name": "eval_duplicate",
                    "JobId": "dlcshared",
                    "WorkspaceId": "247143",
                    "JobStatus": "Succeeded",
                },
            ]
        if output == "judge_":
            return [
                {
                    "Name": "judge_regular",
                    "JobId": "dlcjudge",
                    "WorkspaceId": "247143",
                    "JobStatus": "Succeeded",
                },
                {
                    "Name": "judge_duplicate",
                    "JobId": "dlcshared",
                    "WorkspaceId": "247143",
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


def test_failed_judge_samples_cannot_appear_as_a_complete_score(tmp_path):
    (tmp_path / '20261005_results.json').write_text(json.dumps({
        'results': {'mmbench_en_dev': {'total_acc': 0.5}},
        'judge': {'failed_samples': 1},
    }))
    rows, _ = server._build_metric_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].status == 'judge_failed'
    assert rows[0].value is None
