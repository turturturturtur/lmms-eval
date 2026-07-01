import base64
import io
import re

from PIL import Image


def simplevqa_doc_to_visual(doc):
    image = doc.get("image")
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    if not image:
        raise ValueError(f"SimpleVQA sample is missing image: question={doc.get('question', '')[:80]!r}")

    try:
        image_bytes = base64.b64decode(image)
        decoded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return [decoded_image]
    except Exception as exc:
        raise RuntimeError(f"Failed to decode SimpleVQA image: question={doc.get('question', '')[:80]!r}") from exc


def simplevqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def simplevqa_process_results(doc, result):
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    prediction = result[0].strip()
    answer = str(doc["answer"]).strip()
    question = simplevqa_doc_to_text(doc)
    return {
        "simplevqa_judge": {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "judge_score": None,
            "judge_success": False,
        },
        "needs_llm_judge": True,
        "question": question,
        "answer": answer,
        "raw_output": prediction,
    }


def get_judge_prompt(doc, prediction, target=None):
    question = doc.get("question") or doc.get("__sample_context__", {}).get("question", "")
    answer = str(doc.get("answer") or target or "").strip()
    return f"""You are an expert evaluator for SimpleVQA factuality scoring.

Given a visual question, the reference short answer, and a model prediction, decide whether the prediction correctly answers the question.

Question:
{question}

Reference answer:
{answer}

Model prediction:
{prediction}

Scoring rules:
- Output 1 if the prediction is semantically equivalent to the reference answer.
- Output 1 for harmless wording, casing, pluralization, punctuation, or formatting differences.
- Output 1 if the prediction contains the correct short answer with no contradictory information.
- Output 0 if the prediction is wrong, too vague, contradicts the reference, or adds incorrect facts.
- Output only a single digit, 1 or 0."""


def parse_judge_response(response):
    text = str(response).strip()
    match = re.search(r"(?<!\d)([01])(?!\d)", text)
    if match:
        return int(match.group(1))
    return None


def update_metrics_from_judge(doc, results, fallback_metrics, parsed, raw_response, model_used):
    prediction = results[0].strip() if results else ""
    answer = str(doc.get("answer", fallback_metrics.get("answer", ""))).strip()
    question = doc.get("question") or fallback_metrics.get("question", "")
    score = int(parsed) if parsed in (0, 1, True, False) else 0
    metrics = fallback_metrics.copy()
    metrics["simplevqa_judge"] = {
        "question": question,
        "answer": answer,
        "prediction": prediction,
        "judge_score": score,
        "judge_success": parsed in (0, 1, True, False),
        "judge_raw": raw_response,
        "judge_model": model_used,
    }
    metrics["exact_match"] = float(score)
    metrics["llm_judge_score"] = score
    metrics["llm_judge_raw"] = raw_response
    metrics["llm_judge_model"] = model_used
    metrics["llm_judge_success"] = parsed in (0, 1, True, False)
    metrics["llm_judge_failed"] = parsed not in (0, 1, True, False)
    metrics["needs_llm_judge"] = False
    return metrics


def simplevqa_standalone_aggregate(extracted_data):
    if not extracted_data:
        return 0.0
    total = 0
    correct = 0
    failed = 0
    for item in extracted_data:
        score = item.get("judge_score")
        if score in (0, 1, True, False):
            correct += int(score)
        else:
            failed += 1
        total += 1
    if failed:
        raise ValueError(f"SimpleVQA judge failed for {failed}/{total} samples; inspect judged samples before reporting.")
    return correct / total * 100.0
