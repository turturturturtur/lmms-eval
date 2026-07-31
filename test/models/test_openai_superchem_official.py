import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmms_eval"
    / "models"
    / "model_utils"
    / "superchem_official.py"
)
SPEC = importlib.util.spec_from_file_location("superchem_official_protocol", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(protocol)


def test_official_request_payload_is_strict_and_explicit():
    payload, options = protocol.build_request(
        model="kimi-k3",
        messages=[{"role": "user", "content": "question"}],
        generation_kwargs={
            "official_request": "superchem",
            "max_new_tokens": 32768,
            "temperature": 1.0,
            "reasoning_effort": "high",
            "stream": True,
            "stream_options": {"include_usage": True},
            "request_timeout": 600,
            "request_max_retries": 5,
            "request_concurrency": 10,
        },
    )

    assert payload == {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "question"}],
        "temperature": 1.0,
        "max_tokens": 32768,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "high",
    }
    assert options == protocol.RequestOptions(
        timeout=600,
        max_retries=5,
        concurrency=10,
        require_boxed=True,
        retry_initial_wait=5.0,
        retry_increment=0.5,
    )


def test_official_request_rejects_non_official_or_wrong_budget():
    with pytest.raises(ValueError, match="official_request"):
        protocol.build_request(
            model="kimi-k3",
            messages=[],
            generation_kwargs={"max_new_tokens": 32768},
        )

    with pytest.raises(ValueError, match="32768"):
        protocol.build_request(
            model="kimi-k3",
            messages=[],
            generation_kwargs={
                "official_request": "superchem",
                "max_new_tokens": 128,
                "temperature": 1.0,
                "reasoning_effort": "high",
                "stream": True,
                "stream_options": {"include_usage": True},
                "request_timeout": 600,
                "request_max_retries": 5,
                "request_concurrency": 10,
            },
        )


def test_stream_parser_separates_reasoning_and_content_and_usage():
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content="step 1"),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=r"\boxed{B}"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
            ),
        ),
    ]

    parsed = protocol.parse_stream(chunks)

    assert parsed.content == r"\boxed{B}"
    assert parsed.reasoning_content == "step 1"
    assert parsed.finish_reason == "stop"
    assert parsed.prompt_tokens == 11
    assert parsed.completion_tokens == 7
    assert parsed.reasoning_tokens == 5


def test_stream_parser_extracts_think_tags_when_reasoning_field_is_absent():
    parsed = protocol.parse_stream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="<think>hidden steps</think>\\boxed{C}"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        ]
    )

    assert parsed.reasoning_content == "hidden steps"
    assert parsed.content == r"\boxed{C}"
