from copy import deepcopy
from types import SimpleNamespace

import pytest

from lmms_eval.models.chat import vllm_backend as backend_module
from lmms_eval.models.chat.vllm_backend import VLLMBackend
from lmms_eval.utils import simple_parse_args_string


def _request(*, doc_id=0, until=None, max_new_tokens=16):
    generation_kwargs = {
        "until": until,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    request = SimpleNamespace(
        args=(
            "context",
            lambda doc: [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": doc["question"]}],
                }
            ],
            generation_kwargs,
            doc_id,
            "task",
            "test",
        )
    )
    return request, generation_kwargs


def _backend(monkeypatch, **kwargs):
    fake_accelerator = SimpleNamespace(
        num_processes=1,
        local_process_index=0,
        device="cpu",
    )
    monkeypatch.setattr(backend_module, "Accelerator", lambda: fake_accelerator)
    monkeypatch.setattr(backend_module, "is_budget_exceeded", lambda: False)
    monkeypatch.setattr(backend_module, "log_usage", lambda **kwargs: None)
    monkeypatch.setattr(backend_module, "log_metrics", lambda **kwargs: None)
    monkeypatch.setattr(backend_module, "get_running_totals", lambda: {"total_tokens": 0})

    backend = VLLMBackend(
        model="model-under-test",
        num_concurrent=2,
        adaptive_max_concurrency=2,
        max_retries=1,
        prefix_aware_queue=False,
        **kwargs,
    )
    backend._rank = 1
    backend.task_dict = {
        "task": {
            "test": [
                {"question": "question-0"},
                {"question": "question-1"},
            ]
        }
    }
    return backend


def _capture_payloads(backend, requests):
    captured = []

    def fake_request(payload, url):
        captured.append(deepcopy(payload))
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    backend._make_request = fake_request
    results = backend.generate_until(requests)
    assert [result.text for result in results] == ["answer"] * len(requests)
    return captured


def test_generate_until_forwards_task_stop_and_model_stop_token_ids(monkeypatch):
    backend = _backend(monkeypatch, stop_token_ids="[248046]")
    request, generation_kwargs = _request(until=["\n\n"])
    original_generation_kwargs = deepcopy(generation_kwargs)

    payloads = _capture_payloads(backend, [request])

    assert payloads[0]["stop"] == ["\n\n"]
    assert payloads[0]["stop_token_ids"] == [248046]
    assert generation_kwargs == original_generation_kwargs


def test_generate_until_keeps_request_stop_conditions_isolated(monkeypatch):
    backend = _backend(monkeypatch, stop_token_ids="[248046]")
    request_0, _ = _request(doc_id=0, until="END-0")
    request_1, _ = _request(doc_id=1, until=["END-1", "END-2"])

    payloads = _capture_payloads(backend, [request_0, request_1])
    payloads_by_question = {payload["messages"][0]["content"][0]["text"]: payload for payload in payloads}

    assert payloads_by_question["question-0"]["stop"] == "END-0"
    assert payloads_by_question["question-1"]["stop"] == ["END-1", "END-2"]
    assert all(payload["stop_token_ids"] == [248046] for payload in payloads)


@pytest.mark.parametrize("until", [None, []])
def test_generate_until_omits_absent_stop_conditions(monkeypatch, until):
    backend = _backend(monkeypatch)
    request, _ = _request(until=until)

    payloads = _capture_payloads(backend, [request])

    assert "stop" not in payloads[0]
    assert "stop_token_ids" not in payloads[0]


def test_generate_until_uses_task_native_max_new_tokens_when_enabled(monkeypatch):
    backend = _backend(
        monkeypatch,
        max_new_tokens=8,
        task_native_max_new_tokens=True,
    )
    request, _ = _request(max_new_tokens=16)

    payloads = _capture_payloads(backend, [request])

    assert payloads[0]["max_tokens"] == 16


def test_generate_until_keeps_model_level_max_new_tokens_cap_by_default(monkeypatch):
    backend = _backend(monkeypatch, max_new_tokens=8)
    request, _ = _request(max_new_tokens=16)

    payloads = _capture_payloads(backend, [request])

    assert payloads[0]["max_tokens"] == 8


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "16"])
def test_task_native_max_new_tokens_rejects_missing_or_invalid_values(monkeypatch, value):
    backend = _backend(monkeypatch, task_native_max_new_tokens=True)
    request, generation_kwargs = _request()
    if value is None:
        generation_kwargs.pop("max_new_tokens")
    else:
        generation_kwargs["max_new_tokens"] = value

    with pytest.raises(ValueError, match="max_new_tokens"):
        backend.generate_until([request])


def test_model_args_parser_preserves_json_stop_token_ids(monkeypatch):
    parsed = simple_parse_args_string("model=model-under-test,stop_token_ids=[248046]")

    assert parsed["stop_token_ids"] == "[248046]"
    backend = _backend(monkeypatch, stop_token_ids=parsed["stop_token_ids"])
    assert backend.stop_token_ids == [248046]


def test_model_args_parser_enables_task_native_max_new_tokens(monkeypatch):
    parsed = simple_parse_args_string(
        "model=model-under-test,task_native_max_new_tokens=true"
    )

    backend = _backend(
        monkeypatch,
        task_native_max_new_tokens=parsed["task_native_max_new_tokens"],
    )
    assert backend.task_native_max_new_tokens is True


@pytest.mark.parametrize(
    "until",
    ["", ["valid", ""], ["valid", 1], {"not": "valid"}],
)
def test_generate_until_rejects_invalid_task_stop_conditions(monkeypatch, until):
    backend = _backend(monkeypatch)
    request, _ = _request(until=until)

    with pytest.raises(ValueError, match="until"):
        backend.generate_until([request])


@pytest.mark.parametrize(
    "stop_token_ids",
    ["", "not-json", "[]", "[248046, true]", "[248046, -1]", "[248046, 1.5]"],
)
def test_backend_rejects_invalid_stop_token_ids(monkeypatch, stop_token_ids):
    with pytest.raises(ValueError, match="stop_token_ids"):
        _backend(monkeypatch, stop_token_ids=stop_token_ids)
