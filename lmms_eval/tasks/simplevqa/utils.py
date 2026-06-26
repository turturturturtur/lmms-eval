import base64
import io
import re

from PIL import Image

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

EVAL_AI_PROCESSOR = EvalAIAnswerProcessor()
FORCE_REPROCESS_FROM_SAMPLE = False


def _prediction_text(result):
    if isinstance(result, list):
        return result[0] if result else ""
    return result or ""


def _sample_context(doc):
    if isinstance(doc, dict):
        return doc.get("__sample_context__", {}) or {}
    return {}


def _question_text(doc):
    context = _sample_context(doc)
    question = ""
    if isinstance(doc, dict):
        question = doc.get("question") or doc.get("input") or ""
    question = question or context.get("question") or context.get("input") or context.get("prompt") or ""
    question = str(question)
    for suffix in (
        "\nAnswer the question using a short phrase.",
        "\nAnswer with a short phrase only.",
    ):
        question = question.replace(suffix, "")
    if question.startswith("Question: "):
        question = question[len("Question: "):]
    return question.strip()


def _answer_text(doc, target=None):
    if target is not None and target != "":
        return str(target)
    context = _sample_context(doc)
    if isinstance(doc, dict):
        answer = doc.get("answer") or doc.get("target") or ""
        if answer:
            return str(answer)
    return str(context.get("target") or context.get("answer") or "")


def simplevqa_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]

    image_bytes = base64.b64decode(image)
    decoded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return [decoded_image]


def simplevqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def simplevqa_process_results(doc, result):
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    prediction = EVAL_AI_PROCESSOR(_prediction_text(result))
    reference = EVAL_AI_PROCESSOR(_answer_text(doc))
    exact_match = float(prediction == reference)
    return {
        "exact_match": exact_match,
        "simplevqa_strict_exact_match": exact_match,
        "needs_llm_judge": exact_match == 0.0,
    }


def get_judge_prompt(doc, prediction, target=None):
    question = _question_text(doc)
    answer = _answer_text(doc, target)
    prediction = _prediction_text(prediction)
    return f"""You are a strict answer equivalence judge for a short-answer visual QA benchmark.

Decide whether the Model Prediction should be counted as correct for the Question and Ground Truth Answer. You do not need to inspect the image; only judge answer equivalence against the provided ground truth.

Question:
```
{question}
```

Ground Truth Answer:
```
{answer}
```

Model Prediction:
```
{prediction}
```

Rules:
- Output "correct" only when the prediction is semantically equivalent to the ground truth answer.
- Treat any alternative separated by "<OR>" as acceptable.
- Ignore harmless formatting differences: case, English articles, extra spaces, OCR spacing, terminal punctuation, Chinese punctuation, brackets, quotes, currency symbols when the value is unchanged, and number/date suffixes such as "年" when the meaning is unchanged.
- Accept obvious bilingual aliases or full-name/short-name variants only when they unambiguously refer to the same answer in the question context.
- Mark "incorrect" if the prediction is only a vague substring, misses required specificity, reverses the meaning, has the wrong number/unit/entity, or adds conflicting information.

Strict output format: correct or incorrect"""


def parse_judge_response(response):
    text = (response or "").strip().lower()
    if not text:
        return None
    first_token_match = re.search(r"[a-zA-Z]+|[\u4e00-\u9fff]+|\d+", text)
    first_token = first_token_match.group(0) if first_token_match else text
    if first_token in {"incorrect", "wrong", "false", "no", "0", "不正确", "错误", "错"}:
        return False
    if first_token in {"correct", "right", "true", "yes", "1", "正确", "对"}:
        return True
    if "incorrect" in text or "wrong" in text or "false" in text:
        return False
    if "correct" in text or "true" in text:
        return True
    return None


def update_metrics_from_judge(doc, results, metrics, parsed, raw_response, model):
    updated = dict(metrics)
    strict_exact = float(updated.get("simplevqa_strict_exact_match", updated.get("exact_match", 0.0)))
    updated["simplevqa_strict_exact_match"] = strict_exact
    updated["llm_judge_raw"] = raw_response
    updated["llm_judge_model"] = model
    updated["llm_judge_success"] = parsed is not None
    updated["llm_judge_failed"] = parsed is None
    if parsed is None:
        updated["simplevqa_judged_exact_match"] = strict_exact
        updated["exact_match"] = strict_exact
        return updated

    judged_exact = float(bool(parsed))
    updated["llm_judge_score"] = int(bool(parsed))
    updated["simplevqa_judged_exact_match"] = judged_exact
    updated["exact_match"] = judged_exact
    updated["needs_llm_judge"] = False
    return updated
