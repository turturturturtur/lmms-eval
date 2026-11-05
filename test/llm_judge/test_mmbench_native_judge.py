from unittest.mock import Mock

import pytest

from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator


@pytest.fixture
def item():
    return {"index": 42, "question": "Which option matches?", "prediction": "Unparseable reasoning",
            "A": "cat", "B": "dog", "C": float("nan"), "D": float("nan")}


@pytest.mark.parametrize("response", ["ambiguous reasoning", "Failed to obtain answer via API"])
def test_exhausted_native_judge_never_randomly_guesses(item, monkeypatch, response):
    evaluator = MMBench_Evaluator()
    call = Mock(return_value=response)
    monkeypatch.setattr(evaluator, "get_chat_response", call)
    with pytest.raises(RuntimeError, match="42.*3 attempts"):
        evaluator.extract_answer_from_item(item)
    assert call.call_count == 3
    assert all(c.kwargs["patience"] == 1 for c in call.call_args_list)


def test_valid_no_match_e_is_accepted_without_retry(item, monkeypatch):
    evaluator = MMBench_Evaluator()
    call = Mock(return_value="E")
    monkeypatch.setattr(evaluator, "get_chat_response", call)
    assert evaluator.extract_answer_from_item(item) == ("E", "E")
    call.assert_called_once()


def test_transport_failure_has_three_total_requests(item, monkeypatch):
    post = Mock(side_effect=TimeoutError("fixture timeout"))
    monkeypatch.setattr("lmms_eval.tasks.mmbench.mmbench_evals.requests.post", post)
    monkeypatch.setattr("lmms_eval.tasks.mmbench.mmbench_evals.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="42.*3 attempts"):
        MMBench_Evaluator().extract_answer_from_item(item)
    assert post.call_count == 3


@pytest.mark.parametrize("local", [True, False])
def test_native_request_honors_local_constraints_and_timeout(item, monkeypatch, local):
    monkeypatch.setenv("JUDGE_STRUCTURED_OUTPUTS", "1" if local else "0")
    monkeypatch.setenv("JUDGE_DISABLE_THINKING", "1" if local else "0")
    monkeypatch.setenv("JUDGE_TIMEOUT", "120")
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "E"}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr("lmms_eval.tasks.mmbench.mmbench_evals.requests.post", post)
    evaluator = MMBench_Evaluator(API_URL="http://localhost:8002/v1/chat/completions")
    assert evaluator.extract_answer_from_item(item) == ("E", "E")
    payload = post.call_args.kwargs["json"]
    assert post.call_args.kwargs["timeout"] == 120
    if local:
        assert payload["structured_outputs"] == {"choice": ["A", "B", "E"]}
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert "structured_outputs" not in payload
        assert "chat_template_kwargs" not in payload
