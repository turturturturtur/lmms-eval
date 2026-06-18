import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file, sanitize_for_excel
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

with open(Path(__file__).parent / "mmbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

def _resolve_api_url():
    url = os.getenv("OPENAI_API_URL")
    if url:
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        return url
    judge_base = os.getenv("JUDGE_BASE_URL")
    if judge_base:
        judge_base = judge_base.split(";")[0].strip()
        if not judge_base.endswith("/chat/completions"):
            judge_base = judge_base.rstrip("/") + "/chat/completions"
        return judge_base
    return ""


GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION") or os.getenv("JUDGE_MODEL", "gpt-4o-2024-11-20")
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = _resolve_api_url()
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("JUDGE_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY") or os.getenv("JUDGE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = os.getenv("JUDGE_API_KEY", "YOUR_API_KEY")

mmbench_evaluator = MMBench_Evaluator(sys_prompt=config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)


def _refresh_mmbench_evaluator():
    global mmbench_evaluator
    mmbench_evaluator.API_TYPE = os.getenv("API_TYPE", "openai")
    mmbench_evaluator.model_version = os.getenv("MODEL_VERSION") or os.getenv("JUDGE_MODEL", "gpt-4o-2024-11-20")
    mmbench_evaluator.API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("JUDGE_API_KEY", "YOUR_API_KEY")
    mmbench_evaluator.API_URL = _resolve_api_url()


def mmbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmbench_cn_cc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "source": doc["source"],
    }

    query_prompt = f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt


def mmbench_cn_cc_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "source": doc["source"],
            "category": doc["category"],
        },
        "submission": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "source": doc["source"],
            "category": doc["category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_cn_cc_aggregate_dev_results_eval_standalone(results):
    """Standalone wrapper for mmbench_cn_cc_aggregate_dev_results_eval (no args required)."""
    class _Args:
        output_path = "."
    _refresh_mmbench_evaluator()
    return mmbench_cn_cc_aggregate_dev_results_eval(results, _Args())


def mmbench_cn_cc_aggregate_results_standalone(results):
    """Standalone wrapper for mmbench_cn_cc_aggregate_results (no args required)."""
    class _Args:
        output_path = "."
    _refresh_mmbench_evaluator()
    return mmbench_cn_cc_aggregate_results(results, _Args())


def mmbench_cn_cc_aggregate_dev_results_eval(results, args):
    if os.getenv("SKIP_MMBENCH_DEV_JUDGE", "0") == "1":
        eval_logger.info("SKIP_MMBENCH_DEV_JUDGE=1, skipping GPT-based MMBench dev evaluation during generation.")
        return 0.0
    print("============= MMBench-CN(CC) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    file = generate_submission_file("mmbench_cn_cc_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100


def mmbench_cn_cc_aggregate_results(results, args):
    df = pd.DataFrame(results)
    df = df.map(sanitize_for_excel)
    file = generate_submission_file("mmbench_cn_cc_results.xlsx", args)
    with pd.ExcelWriter(file) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {file}")
