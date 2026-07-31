import ast
import io
import re

from PIL import Image

from lmms_eval.api.metrics import levenshtein_distance


def _as_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def chartqapro_doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    if isinstance(image, dict) and image.get("bytes") is not None:
        image = image["bytes"]
    if isinstance(image, (bytes, bytearray)):
        return [Image.open(io.BytesIO(image)).convert("RGB")]
    raise TypeError(f"Unsupported ChartQAPro image type: {type(image)!r}")


def _direct_prompt(questions, answers, question_type):
    final_question = questions[-1]
    common = (
        "Answer using only the final answer, without explanation. "
        "If the answer cannot be determined from the chart, answer 'unanswerable'. "
    )

    if question_type == "Conversational":
        history = []
        for question, answer in zip(questions[:-1], answers[:-1]):
            history.append(f"Question: {question}\nAnswer: {answer}")
        context = "\n".join(history)
        return (
            "Answer the final question using the chart and conversation history. "
            f"{common}\nConversation history:\n{context}\nFinal question: {final_question}"
        )
    if question_type == "Multi Choice":
        return (
            "Select the correct option from the chart. Return only the option letter "
            f"(a, b, c, or d). {common}\nQuestion: {final_question}"
        )
    if question_type == "Fact Checking":
        return f"Determine whether the statement is true or false. {common}\nStatement: {final_question}"
    if question_type == "Hypothetical":
        return (
            "Answer the hypothetical question from the chart. Use the chart's exact notation "
            f"for numerical units. {common}\nQuestion: {final_question}"
        )
    return (
        "Answer the factoid question from the chart. Do not add units unless they are required; "
        f"when required, use the chart's exact notation. {common}\nQuestion: {final_question}"
    )


def chartqapro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    questions = _as_list(doc["Question"])
    answers = _as_list(doc["Answer"])
    question_type = str(doc["Question Type"])
    paragraph = str(doc.get("Paragraph") or "").strip()
    prompt = _direct_prompt(questions, answers, question_type)
    if paragraph:
        prompt = f"Context paragraph:\n{paragraph}\n\n{prompt}"
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def chartqapro_doc_to_target(doc):
    return _as_list(doc["Answer"])[-1]


def _parse_answer_list(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip(" '") for item in parsed]


def _to_float(value):
    try:
        return float(value.strip().strip("%"))
    except (AttributeError, ValueError):
        return None


def _anls(target, prediction, threshold=0.5):
    target = target.lower()
    prediction = prediction.lower()
    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0
    score = 1.0 - levenshtein_distance(target, prediction) / max(len(target), len(prediction))
    return score if score >= threshold else 0.0


def _score_single(target, prediction, max_relative_change=0.05):
    target = target.strip().strip("%").strip()
    prediction = prediction.strip().strip("%").strip()
    target_float = _to_float(target)
    prediction_float = _to_float(prediction)
    if target_float is not None and prediction_float is not None:
        if target_float == 0.0:
            return float(prediction_float == 0.0)
        return float(abs(prediction_float - target_float) / abs(target_float) <= max_relative_change)
    return _anls(target, prediction)


def _relaxed_correctness(target, prediction, year_flags):
    targets = _parse_answer_list(target) or [target]
    predictions = _parse_answer_list(prediction) or [prediction]
    if len(year_flags) < len(targets):
        year_flags = year_flags * len(targets)

    scores = []
    for index in range(max(len(targets), len(predictions))):
        if index >= len(targets) or index >= len(predictions):
            scores.append(0.0)
            continue
        if str(year_flags[index]).upper() == "YES":
            scores.append(float(targets[index].strip().lower() == predictions[index].strip().lower()))
        else:
            scores.append(_score_single(targets[index], predictions[index]))
    return sum(scores) / len(scores) if scores else 0.0


def chartqapro_process_results(doc, results):
    prediction = str(results[0]).strip(".\n ")
    target = chartqapro_doc_to_target(doc).strip(".\n ")
    year_flags = _as_list(doc["Year"])
    if str(doc["Question Type"]) == "Conversational":
        year_flags = year_flags[-1:]
    return {"relaxed_overall": _relaxed_correctness(target, prediction, year_flags)}
