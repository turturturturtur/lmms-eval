import re
from dataclasses import dataclass
from typing import Any, Iterable


OFFICIAL_TASK_NAME = "superchem"
OFFICIAL_MAX_NEW_TOKENS = 32768
OFFICIAL_TEMPERATURE = 1.0
OFFICIAL_REASONING_EFFORTS = {"low", "medium", "high"}
OFFICIAL_TIMEOUT = 600
OFFICIAL_MAX_RETRIES = 5
# SUPERChem's official launcher uses N_PROCS=4 and N_THREADS=10.  lmms-eval
# applies this setting per distributed process, so each rank must use ten
# concurrent requests (4 x 10 = 40 requests globally).
OFFICIAL_CONCURRENCY = 10


@dataclass(frozen=True)
class RequestOptions:
    timeout: int
    max_retries: int
    concurrency: int
    require_boxed: bool
    retry_initial_wait: float
    retry_increment: float


@dataclass(frozen=True)
class StreamResponse:
    content: str
    reasoning_content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


def build_request(model: str, messages: list[dict[str, Any]], generation_kwargs: dict):
    if generation_kwargs.get("official_request") != OFFICIAL_TASK_NAME:
        raise ValueError(
            "superchem official requests must set official_request='superchem'."
        )

    max_new_tokens = int(generation_kwargs.get("max_new_tokens", -1))
    if max_new_tokens != OFFICIAL_MAX_NEW_TOKENS:
        raise ValueError(
            "SUPERChem official requires max_new_tokens=32768; "
            f"got {max_new_tokens}."
        )

    temperature = float(generation_kwargs.get("temperature", -1))
    if temperature != OFFICIAL_TEMPERATURE:
        raise ValueError(
            "SUPERChem official requires temperature=1.0; "
            f"got {temperature}."
        )

    reasoning_effort = generation_kwargs.get("reasoning_effort")
    if reasoning_effort not in OFFICIAL_REASONING_EFFORTS:
        raise ValueError(
            "SUPERChem official requires reasoning_effort in "
            f"{sorted(OFFICIAL_REASONING_EFFORTS)}; got {reasoning_effort!r}."
        )

    if generation_kwargs.get("stream") is not True:
        raise ValueError("SUPERChem official requires stream=True.")
    if generation_kwargs.get("stream_options") != {"include_usage": True}:
        raise ValueError(
            "SUPERChem official requires stream_options={'include_usage': True}."
        )

    timeout = int(generation_kwargs.get("request_timeout", -1))
    if timeout != OFFICIAL_TIMEOUT:
        raise ValueError(
            "SUPERChem official requires request_timeout=600; "
            f"got {timeout}."
        )
    max_retries = int(generation_kwargs.get("request_max_retries", -1))
    if max_retries != OFFICIAL_MAX_RETRIES:
        raise ValueError(
            "SUPERChem official requires request_max_retries=5; "
            f"got {max_retries}."
        )
    concurrency = int(generation_kwargs.get("request_concurrency", -1))
    if concurrency != OFFICIAL_CONCURRENCY:
        raise ValueError(
            "SUPERChem official requires request_concurrency=10 "
            "(N_THREADS=10 per rank; N_PROCS=4 globally); "
            f"got {concurrency}."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": reasoning_effort,
    }
    return payload, RequestOptions(
        timeout=timeout,
        max_retries=max_retries,
        concurrency=concurrency,
        require_boxed=True,
        # Matches SUPERChem eval/eval.py's single_task retry schedule.
        retry_initial_wait=5.0,
        retry_increment=0.5,
    )


def _get(value: Any, key: str, default: Any = None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_stream(chunks: Iterable[Any]) -> StreamResponse:
    content_parts = []
    reasoning_parts = []
    finish_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0

    for chunk in chunks:
        choices = _get(chunk, "choices", []) or []
        if choices:
            choice = choices[0]
            delta = _get(choice, "delta")
            reasoning = _get(delta, "reasoning_content")
            content = _get(delta, "content")
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)
            candidate_finish_reason = _get(choice, "finish_reason")
            if candidate_finish_reason:
                finish_reason = candidate_finish_reason

        usage = _get(chunk, "usage")
        if usage:
            prompt_tokens = int(_get(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(_get(usage, "completion_tokens", 0) or 0)
            details = _get(usage, "completion_tokens_details")
            reasoning_tokens = int(
                (_get(details, "reasoning_tokens", 0) or 0) if details else 0
            )

    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    if not reasoning_content and "<think>" in content:
        matches = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
        if matches:
            reasoning_content = "\n\n---\n\n".join(matches)
            content = re.sub(
                r"<think>.*?</think>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

    return StreamResponse(
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def has_boxed_answer(content: str) -> bool:
    return bool(re.search(r"\\boxed\{(.+?)\}", content, re.DOTALL))
