import json
import re


def seed_doc_to_visual(doc):
    image = doc.get("image")
    if image is None:
        raise ValueError(f"SEED-Bench-2-Plus sample is missing image: question_id={doc.get('question_id')!r}")
    return [image.convert("RGB")]


def parse_choice_img(choice: str, img_token: str):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_text(doc, model_specific_kwargs=None):
    if model_specific_kwargs is None:
        raise ValueError("SEED-Bench-2-Plus requires model_specific_kwargs with img_token and post_prompt")
    question = doc["question"]
    question = question.replace("<img>", model_specific_kwargs["img_token"])
    question += "\n" + f"A. {parse_choice_img(doc['choice_A'], model_specific_kwargs['img_token'])}\n"
    question += f"B. {parse_choice_img(doc['choice_B'], model_specific_kwargs['img_token'])}\n"
    question += f"C. {parse_choice_img(doc['choice_C'], model_specific_kwargs['img_token'])}\n"
    question += f"D. {parse_choice_img(doc['choice_D'], model_specific_kwargs['img_token'])}"

    return f"{question}\n{model_specific_kwargs['post_prompt']}"


def extract_seed_answer_letter(response: str) -> str:
    response = response.strip()
    patterns = [
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\]\.]?\s*$",
        r"(?:answer|option|choice)(?:\s+is)?\s*[:：]?\s*[\(\[]?\s*([A-D])\s*[\)\]\.]?",
        r"^\s*([A-D])[\.\)]\s+",
        r"\(([A-D])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return response[:1].upper() if len(response) == 1 and response.upper() in "ABCD" else ""


def seed_process_result(doc, result):
    pred = extract_seed_answer_letter(result[0])
    answer = doc["answer"]
    data_type = doc["question_image_type"].capitalize()

    return {f"seedbench_2_plus_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, "seedbench_2_plus_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count if total_count != 0 else 0


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score
