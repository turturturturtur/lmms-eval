from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image


METRIC_NAME = "webui_local_vqa_demo_exact_match"
REQUIRED_FIELDS = ("image_path", "question", "answer", "accepted_answers")


def _require_fields(doc: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in doc]
    if missing:
        raise KeyError(f"webui_local_vqa_demo sample is missing required fields: {missing}")


def _normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def webui_local_vqa_demo_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    _require_fields(doc)
    image_path = Path(str(doc["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"webui_local_vqa_demo image does not exist: {image_path}")
    return [Image.open(image_path).convert("RGB")]


def webui_local_vqa_demo_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    _require_fields(doc)
    kwargs = lmms_eval_specific_kwargs or {}
    question = str(doc["question"]).strip()
    if not question:
        raise ValueError("webui_local_vqa_demo question must be non-empty")
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", ""))
    return f"{pre_prompt}{question}{post_prompt}"


def webui_local_vqa_demo_doc_to_target(doc: dict[str, Any]) -> str:
    _require_fields(doc)
    answer = str(doc["answer"]).strip()
    if not answer:
        raise ValueError("webui_local_vqa_demo answer must be non-empty")
    return answer


def webui_local_vqa_demo_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    _require_fields(doc)
    if not results:
        raise ValueError("webui_local_vqa_demo expected at least one model result")
    accepted_answers = doc["accepted_answers"]
    if not isinstance(accepted_answers, list) or not accepted_answers:
        raise TypeError("webui_local_vqa_demo accepted_answers must be a non-empty list")
    normalized_prediction = _normalize_answer(results[0])
    normalized_answers = {_normalize_answer(answer) for answer in accepted_answers}
    score = float(normalized_prediction in normalized_answers)
    return {METRIC_NAME: score}
