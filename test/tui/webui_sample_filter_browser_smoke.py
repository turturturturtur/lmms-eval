"""Offline browser acceptance using the built UI and real sample filtering API."""

import json
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse, parse_qs

from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright, expect

from lmms_eval.tui import server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path(server.__file__).parent / "web/dist")
    dist = parser.parse_args().dist
    errors = []
    requests = []
    with TemporaryDirectory() as directory, TestClient(server.app) as client:
        root = Path(directory)
        (root / "run_results.json").write_text(json.dumps({"results": {
            "ai2d": {"accuracy": 0.5}, "ocrbench": {"accuracy": 0.777},
        }}))
        samples = [{"doc_id": i, "input": "question", "target": "A",
                    "resps": ["raw answer"], "filtered_resps": "A" if i == 0 else "B" if i == 1 else "unparsed"}
                   for i in range(57)]
        (root / "run_samples_ai2d.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        judge_dir = root / "judge"
        judge_dir.mkdir()
        (judge_dir / "custom.jsonl").write_text(json.dumps({
            "doc_id": 0, "target": "A", "filtered_resps": "A", "resps": ["raw answer"],
            "judge_mode": "llm_judge", "metrics": {"llm_judge_score": 0, "llm_judge_success": True},
        }) + "\n")
        (judge_dir / "run_results.json").write_text(json.dumps({
            "results": {"mmbench_en_dev": {"total_acc": 0.9691, "accuracy": 84.27835051546391}},
            "judge": {"model": "Qwen3.5-9B", "samples_file": "custom.jsonl", "failed_samples": 0},
        }))
        (judge_dir / "failed_results.json").write_text(json.dumps({
            "results": {"mmmu_val_qwen3_official": {"total_acc": 0.4522, "accuracy": -1}},
            "judge": {"mode": "judge"},
        }))
        (judge_dir / "partial_samples_sfe-en.jsonl").write_text(json.dumps({
            "judge_mode": "llm_judge", "metrics": {"exact_match": 1},
        }) + "\n")
        name = "eval_judge_retry_dlcsource_abcd1234"
        server._get_dlc_job_detail = lambda _: {"DisplayName": name}
        server._job_runtime_paths = lambda _: (None, root, None)
        session, _ = server._create_auth_session({"username": "fixture", "role": "user", "access_key_id": "fake"})
        client.cookies.set(server.AUTH_COOKIE_NAME, session)
        job = server.DlcJobSummary(job_id="fixture", name=name, status="Succeeded",
                                   job_stage=server._job_stage_from_name(name), create_time="2026-10-06T00:00:00Z").model_dump()

        def route_request(route):
            url = urlparse(route.request.url)
            path = url.path
            status = 200
            if path == "/" or path.startswith("/assets/") or path == "/favicon.svg":
                route.fulfill(path=str(dist / ("index.html" if path == "/" else path.lstrip("/"))))
                return
            if path == "/auth/me":
                payload = {"username": "fixture", "display_name": "Fixture", "role": "user", "access_key_id": "fake"}
            elif path in ("/health", "/defaults", "/storage"):
                payload = client.get(path).json()
            elif path in ("/tasks", "/models"):
                payload = []
            elif path == "/eval/preview":
                payload = {"command": "fixture"}
            elif path == "/dlc/pool-usage":
                status, payload = 503, {"detail": "offline fixture"}
            elif path == "/dlc/jobs":
                payload = {"jobs": [job], "total": 1, "start_time": "2026-09-06T00:00:00Z",
                           "end_time": "2026-10-06T00:00:00Z", "source": "fixture", "fetched_at": "now"}
            elif path == "/dlc/jobs/fixture":
                payload = {"job": {}, "result_status": "has_results"}
            elif path.endswith("/metrics"):
                payload = {"job_id": "fixture", "metrics": [row.model_dump() for row in server._build_metric_rows(root)[0]], "summary_files": []}
            elif path.endswith("/samples"):
                requests.append(parse_qs(url.query))
                response = client.get(path + "?" + url.query)
                status, payload = response.status_code, response.json()
            else:
                payload = {}
            route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path="/root/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome",
                headless=True, args=["--no-sandbox"],
            )
            page = browser.new_page(viewport={"width": 1800, "height": 1000})
            page.on("pageerror", lambda error: (errors.append(str(error)), print(f"Browser error: {error}", flush=True)))
            page.route("**/*", route_request)
            page.goto("http://webui.test/")
            page.get_by_role("button", name="View Logs", exact=True).click()
            expect(page.get_by_text("Pending judge", exact=True)).to_be_visible()
            expect(page.get_by_text("0.777", exact=True)).to_have_count(0)
            expect(page.get_by_text("84.2784", exact=True)).to_be_visible()
            expect(page.get_by_text("0.9691", exact=True)).to_have_count(0)
            expect(page.get_by_role("cell", name="Judge Qwen3.5-9B")).to_be_visible()
            expect(page.get_by_text("Judge failed", exact=True)).to_be_visible()
            expect(page.get_by_text("Awaiting aggregate", exact=True)).to_be_visible()
            page.get_by_role("cell", name="mmbench_en_dev", exact=True).click()
            page.get_by_role("button", name="Wrong only", exact=True).click()
            expect(page.get_by_text("1-1 / 1", exact=True)).to_be_visible()
            page.get_by_role("button", name="Back to job", exact=True).click()
            page.get_by_role("cell", name="ai2d", exact=True).click()
            slider = page.get_by_role("slider", name="Sample filter")
            expect(slider).to_have_value("0")
            expect(page.locator("th").filter(has=page.get_by_text("resps", exact=True))).to_be_visible()
            page.get_by_role("button", name="Next", exact=True).click()
            expect(page.get_by_text("51-57 / 57", exact=True)).to_be_visible()
            page.get_by_role("button", name="Unknown", exact=True).click()
            expect(page.get_by_text("1-50 / 55", exact=True)).to_be_visible()
            assert requests[-1]["sample_filter"] == ["unknown"]
            assert requests[-1]["offset"] == ["0"]
            expect(slider).to_have_value("2")
            slider.focus()
            slider.press("ArrowLeft")
            expect(page.get_by_text("1-1 / 1", exact=True)).to_be_visible()
            assert requests[-1]["sample_filter"] == ["wrong"]
            slider.press("ArrowLeft")
            expect(page.get_by_text("1-50 / 57", exact=True)).to_be_visible()
            assert requests[-1]["sample_filter"] == ["all"]
            assert errors == [], errors
            browser.close()
    print("PASS: judge source/model, native score, failed/partial aggregates, custom sample links; raw responses and All/Wrong/Unknown pagination")


if __name__ == "__main__":
    main()
