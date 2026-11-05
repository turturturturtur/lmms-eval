import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmms_eval.llm_judge.protocol import Request, ServerConfig
from lmms_eval.llm_judge.providers.openai import OpenAIProvider


@pytest.mark.parametrize("sdk", [True, False])
@pytest.mark.parametrize("structured", [True, False])
def test_local_option_constraint_reaches_both_transports(sdk, structured, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("OPENAI_API_URL", "http://localhost:8002/v1")
    monkeypatch.setenv("JUDGE_DISABLE_THINKING", "1")
    monkeypatch.setenv("JUDGE_STRUCTURED_OUTPUTS", "1" if structured else "0")
    provider = OpenAIProvider(ServerConfig(model_name="fixture"))
    provider.use_client = sdk
    if sdk:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Z"))], model="fixture", usage=None)
        provider._client_cycle = itertools.cycle([client])
    else:
        request = Mock(return_value={"choices": [{"message": {"content": "Z"}}], "model": "fixture"})
        monkeypatch.setattr(provider, "_make_request", request)
    result = provider.evaluate(Request(messages=[{"role": "user", "content": "Match answer to A, B or Z"}], allowed_outputs=["A", "B", "Z"]))
    assert result.content == "Z"
    payload = client.chat.completions.create.call_args.kwargs["extra_body"] if sdk else request.call_args.args[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    if structured:
        assert payload["structured_outputs"] == {"choice": ["A", "B", "Z"]}
    else:
        assert "structured_outputs" not in payload
