import asyncio
import json

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
            metric_id="0",
            offset=0,
            limit=50,
            only_wrong=True,
        )
    )

    assert response.total == 1
    assert [row["doc_id"] for row in response.rows] == [0]
    assert response.answer_stats.wrong_total == 1
    assert response.answer_stats.unknown_correctness_total == 0
