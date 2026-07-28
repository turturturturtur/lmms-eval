from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Request, Route, sync_playwright


def _timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(job_id: str, name: str, create_time: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "name": name,
        "status": "Succeeded",
        "workspace_id": "240810",
        "resource_id": "quota",
        "job_type": "PyTorchJob",
        "priority": "6",
        "user_name": "Smoke",
        "user_id": "smoke",
        "job_stage": "eval",
        "lmms_tasks": [],
        "llm_judge_tasks": [],
        "requires_llm_judge": False,
        "create_time": create_time,
        "submitted_time": create_time,
        "running_time": create_time,
        "finish_time": create_time,
        "duration_seconds": "1",
        "result_root": None,
        "has_results": False,
        "can_kill": False,
        "kill_disabled_reason": "DLC job status is not killable: succeeded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--chromium-executable", required=True)
    args = parser.parse_args()

    requests: list[dict[str, str]] = []

    now = datetime.now(timezone.utc).replace(microsecond=0)
    boundary = now - timedelta(days=30)

    def route_jobs(route: Route, request: Request) -> None:
        query = parse_qs(urlparse(request.url).query)
        start_time = query["start_time"][0]
        end_time = query["end_time"][0]
        requests.append({"start_time": start_time, "end_time": end_time})
        request_index = len(requests)
        jobs = (
            [
                _job("dlc-new", "eval_new", _timestamp(now - timedelta(days=1))),
                *[
                    _job(f"dlc-filler-{index}", f"eval_filler_{index}", _timestamp(now - timedelta(days=index + 2)))
                    for index in range(35)
                ],
                _job("dlc-boundary", "eval_boundary", _timestamp(boundary)),
            ]
            if request_index == 1
            else [
                _job("dlc-boundary", "eval_boundary", _timestamp(boundary)),
                _job("dlc-old", "eval_old", _timestamp(boundary - timedelta(days=5))),
            ]
        )
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "jobs": jobs,
                    "total": len(jobs),
                    "start_time": start_time,
                    "end_time": end_time,
                    "fetched_at": _timestamp(now),
                    "source": "browser-smoke",
                }
            ),
        )

    def route_detail(route: Route) -> None:
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"job": {}, "result_status": "not_found"}))

    def route_metrics(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "smoke", "metrics": [], "summary_files": [], "message": ""}),
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=args.chromium_executable)
        page = browser.new_page()
        page.on("console", lambda message: print(f"console[{message.type}] {message.text}", flush=True))
        page.on("pageerror", lambda error: print(f"pageerror {error}", flush=True))
        page.route(
            "**/auth/me",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"username": "smoke", "display_name": "Smoke", "role": "user", "access_key_id": "smoke", "expires_at": 0}),
            ),
        )
        page.route(
            "**/defaults",
            lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"job_name": "eval_smoke", "tasks": [], "dlc_config": {}})),
        )
        page.route("**/tasks", lambda route: route.fulfill(status=200, content_type="application/json", body="[]"))
        page.route(
            "**/dlc/pool-usage",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"workspace_id": "240810", "resource_id": "quota", "resource_name": "smoke", "active_statuses": [], "gpu": {"used": 0, "total": 1, "percent": 0, "capacity_source": "smoke"}, "cpu": {"used": 0, "total": 1, "percent": 0, "capacity_source": "smoke"}, "jobs": [], "errors": [], "fetched_at": _timestamp(now), "source": "smoke"}),
            ),
        )
        page.route("**/eval/preview", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"command": "smoke"})))
        page.route("**/dlc/jobs?*", route_jobs)
        page.route("**/dlc/jobs/*/metrics", lambda route: route_metrics(route))
        page.route("**/dlc/jobs/*", lambda route: route_detail(route))
        page.goto(args.url, wait_until="domcontentloaded")
        view_logs = page.get_by_role("button", name="View Logs")
        view_logs.click()
        scroll = page.get_by_test_id("viewlog-job-list-scroll")
        scroll.get_by_text("eval_new", exact=True).wait_for()

        first = requests[0]
        first_span = datetime.fromisoformat(first["end_time"].replace("Z", "+00:00")) - datetime.fromisoformat(first["start_time"].replace("Z", "+00:00"))
        assert first_span == timedelta(days=30), first

        scroll.evaluate("element => { element.scrollTop = element.scrollHeight; element.dispatchEvent(new Event('scroll')) }")
        scroll.get_by_text("eval_old", exact=True).wait_for()

        assert len(requests) == 2, requests
        second = requests[1]
        second_span = datetime.fromisoformat(second["end_time"].replace("Z", "+00:00")) - datetime.fromisoformat(second["start_time"].replace("Z", "+00:00"))
        assert second_span == timedelta(days=15), second
        assert second["end_time"] == first["start_time"], requests
        assert scroll.get_by_text("eval_boundary", exact=True).count() == 1
        assert scroll.get_by_text("eval_old", exact=True).count() == 1
        browser.close()

    print(json.dumps({"requests": requests, "jobs": ["dlc-new", "dlc-boundary", "dlc-old"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
