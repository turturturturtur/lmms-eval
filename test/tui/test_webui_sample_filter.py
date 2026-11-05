import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from lmms_eval.tui import server


def test_chartqa_only_wrong_uses_selected_relaxed_overall_metric(tmp_path, monkeypatch):
    result_root = tmp_path / "chartqa"
    result_root.mkdir()
    result_file = result_root / "20260716_results.json"
    sample_file = result_root / "20260716_samples_chartqa.jsonl"

    result_file.write_text(
        json.dumps(
            {
                "results": {
                    "chartqa": {
                        "relaxed_overall,none": 0.5,
                        "relaxed_overall_stderr,none": 0.5,
                    }
                },
                "n-samples": {"chartqa": {"original": 2, "effective": 2}},
            }
        ),
        encoding="utf-8",
    )
    samples = [
        {"doc_id": 0, "target": "100", "filtered_resps": ["200"], "relaxed_overall": 0.0},
        {"doc_id": 1, "target": "100", "filtered_resps": ["100"], "relaxed_overall": 1.0},
    ]
    sample_file.write_text(
        "".join(f"{json.dumps(sample)}\n" for sample in samples),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "_get_dlc_job_detail", lambda _job_id: {"DisplayName": "eval_chartqa_fixture"})
    monkeypatch.setattr(server, "_job_runtime_paths", lambda _detail: (None, result_root, None))

    response = asyncio.run(
        server.get_dlc_metric_samples(
            job_id="dlc-chartqa-fixture",
            metric_id=server._build_metric_rows(result_root)[0][0].metric_id,
            offset=0,
            limit=50,
            only_wrong=True,
        )
    )

    assert response.total == 1
    assert [row["doc_id"] for row in response.rows] == [0]
    assert response.answer_stats.wrong_total == 1
    assert response.answer_stats.unknown_correctness_total == 0


@pytest.mark.parametrize("sample_filter,expected", [
    ("all", [0, 1, 2, 3, 4, None, None]),
    ("wrong", [1]),
    ("unknown", [2, 3, 4, None, None]),
])
def test_sample_filter_separates_wrong_and_unparseable(tmp_path, sample_filter, expected):
    sample_file = tmp_path / "samples.jsonl"
    samples = [
        {"doc_id": 0, "target": "A", "filtered_resps": "A", "exact_match": 1},
        {"doc_id": 1, "target": "A", "filtered_resps": "B", "exact_match": 0},
        {"doc_id": 2, "target": "A", "filtered_resps": "cannot extract", "exact_match": 0},
        {"doc_id": 3, "target": "free text", "filtered_resps": "unscored"},
        {"doc_id": 4, "target": "A", "filtered_resps": "B", "exact_match": 0,
         "judge_mode": "error", "metrics": {"judge_failed": True}},
    ]
    sample_file.write_text("".join(json.dumps(sample) + "\n" for sample in samples) + '{broken json\n[]\n')
    kwargs = dict(job_id="test", metric_id="test", metric_name="exact_match", only_wrong=False, sample_filter=sample_filter)
    rows, total, stats = server._read_sample_jsonls([str(sample_file)], offset=0, limit=50, **kwargs)
    assert [row.get("doc_id") for row in rows] == expected
    assert total == len(expected)
    assert stats.total == 7
    assert stats.wrong_total == 1
    assert stats.unknown_correctness_total == 5
    page, page_total, _ = server._read_sample_jsonls([str(sample_file)], offset=1, limit=2, **kwargs)
    assert [row.get("doc_id") for row in page] == expected[1:3]
    assert page_total == total


def test_judged_extraction_failure_is_unknown_and_success_overrides_eval():
    original = {"target": "A", "filtered_resps": "The answer is A.", "exact_match": 0,
                "mmmu_acc_official": {"parsed_pred": None}}
    for success, prediction, hit, expected in [(False, "Z", 0, None), (True, "A", 1, True), (True, "B", 0, False)]:
        sample = {**original, "judge_mode": "llm_judge", "metrics": {
            "mmmu_acc_official": {"parsed_pred": [prediction], "extraction_success": success, "hit": hit},
        }}
        extracted = server._choice_prediction(sample)
        assert extracted == prediction
        assert server._sample_correctness(sample, "exact_match", "A", extracted) is expected


def test_unknown_filter_is_exposed_by_samples_api(tmp_path, monkeypatch):
    (tmp_path / "run_results.json").write_text(json.dumps({"results": {"ai2d": {"accuracy": 0.5}}}))
    (tmp_path / "run_samples_ai2d.jsonl").write_text(
        json.dumps({"doc_id": 0, "target": "A", "filtered_resps": "B"}) + "\n" +
        json.dumps({"doc_id": 1, "target": "A", "filtered_resps": "unparseable"}) + "\n"
    )
    monkeypatch.setattr(server, "_get_dlc_job_detail", lambda _: {"DisplayName": "eval_fixture"})
    monkeypatch.setattr(server, "_job_runtime_paths", lambda _: (None, tmp_path, None))
    session, _ = server._create_auth_session({"username": "fixture", "role": "user", "access_key_id": "fake"})
    metric = server._build_metric_rows(tmp_path)[0][0]
    with TestClient(server.app) as client:
        client.cookies.set(server.AUTH_COOKIE_NAME, session)
        url = f"/dlc/jobs/fixture/metrics/{metric.metric_id}/samples"
        response = client.get(url, params={"sample_filter": "unknown"})
        assert response.status_code == 200, response.text
        assert [row["doc_id"] for row in response.json()["rows"]] == [1]
        assert client.get(url, params={"sample_filter": "invalid"}).status_code == 422
