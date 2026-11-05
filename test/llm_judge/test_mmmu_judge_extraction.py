"""A valid no-match judgment must not turn into retries or a random answer."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lmms_eval.llm_judge.standalone import JudgeRunner


@pytest.mark.parametrize("package", ["mmmu", "mmmu_pro_qwen3_official"])
@pytest.mark.parametrize("answer", ["Z", "A"])
def test_valid_judge_option_returns_after_one_request(package, answer, monkeypatch):
    module = importlib.import_module(f"lmms_eval.tasks.{package}.utils_qwen3_official")
    server = Mock()
    server.evaluate.return_value = SimpleNamespace(content=answer)
    monkeypatch.setattr(module, "get_judge_server", lambda: (server, module.ServerConfig(model_name="fixture"), "fixture"))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    option, _, success = module.call_judge_for_extraction(
        {"question": "Which?"}, "Unclear answer", {"A": "Alpha", "B": "Beta"}
    )
    assert success is True
    assert option == answer
    assert server.evaluate.call_count == 1
    assert server.evaluate.call_args.args[0].allowed_outputs == ["A", "B", "Z"]


@pytest.mark.parametrize("method,expected_failure", [("random", True), ("rule", False)])
def test_random_extraction_failure_is_visible_without_rejecting_rule_refusals(method, expected_failure):
    metrics = {"mmmu_acc": {"hit": False, "extraction_method": method, "extraction_success": False}}
    runner = JudgeRunner(judge_mode="rule")
    result = runner._judge_sample(
        {"doc_id": 0, "doc": {"a": 1, "b": 2, "c": 3}, "filtered_resps": ["Z"]},
        SimpleNamespace(), lambda doc, responses: metrics,
    )
    assert bool(result["metrics"].get("judge_failed")) == expected_failure
