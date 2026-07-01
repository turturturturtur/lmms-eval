import re
from typing import Dict, Any, List
from PIL import Image

PROMPT_TEMPLATE = """\
The following is a multiple choice question (with answers). 
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{question}

Options:
{choices}
"""

PROMPT_TEMPLATE_NO_IMAGE = """\
The following is a multiple choice question (with answers).
If an image is mentioned ignore this information and try your best to answer the question.
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{question}

Options:
{choices}
"""

PROMPT_TEMPLATE_NO_IMAGE_V2 = """\
We have a multiple choice question (with answers) and images.
However I will not give you the question text or the images, I will only give you the choices, so please try your best to answer the question.
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{choices}
"""


# REGEX_PATTERN = r"answer is \*?\*?\(?([0-9])\)?\*?\*?"
REGEX_PATTERN = r"[Tt]he answer is[:：\s]*\*?\*?[\(\（]?([A-Za-z0-9])[\)\）]?\*?\*?"




def _format_choices(choices: List[str]) -> str:
    s = ""
    for j, ch in enumerate(choices):
        # chr(65) 是 'A', chr(66) 是 'B'...
        letter = chr(65 + j)
        s += f"  ({letter}): {ch}\n"
    return s

def _get_last_sentence_regex(text: str) -> str:
    sentence_pattern = r'[^.!?]+[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    return sentences[-1].strip() if sentences else text.strip()


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    mode = lmms_eval_specific_kwargs.get("mode", "default")

    if mode == "stage1":
        question = doc["question_1"]
        choices = doc["choices_1"]
    else:
        question = doc["question"]
        choices = doc["choices"]

    choices_str = _format_choices(choices)

    if mode == "noimage_v2":
        return PROMPT_TEMPLATE_NO_IMAGE_V2.format(
            choices=choices_str
        )

    if mode == "noimage_v3":
        question = _get_last_sentence_regex(question)
        return PROMPT_TEMPLATE_NO_IMAGE.format(
            question=question,
            choices=choices_str
        )

    if mode == "noimage":
        return PROMPT_TEMPLATE_NO_IMAGE.format(
            question=question,
            choices=choices_str
        )

    # default
    return PROMPT_TEMPLATE.format(
        question=question,
        choices=choices_str
    )


def doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    mode = lmms_eval_specific_kwargs.get("mode", "default")

    if mode.startswith("noimage"):
        return None

    images = doc.get("images_list")
    if images is None:
        raise ValueError(f"MicroVQA sample is missing images_list: question={doc.get('question', '')[:80]!r}")

    visual = []
    for idx, img in enumerate(images):
        if img is None:
            raise ValueError(f"MicroVQA sample has null image at index={idx}; question={doc.get('question', '')[:80]!r}")
        visual.append(img.convert("RGB"))
    return visual


def doc_to_target(doc: Dict[str, Any], model_specific_target_kwargs=None) -> int:
    model_specific_target_kwargs = model_specific_target_kwargs or {}
    mode = model_specific_target_kwargs.get("mode", "default")

    if mode == "stage1":
        return doc["correct_index_1"]

    return doc["correct_index"]

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0]
    gt = doc_to_target(doc)  # gt 是索引 0, 1, 2...

    # 1. 尝试使用正则匹配标准格式 "The answer is (A)"
    match = re.search(REGEX_PATTERN, response)
    
    if match is not None:
        token = match.group(1).upper()
    else:
        # 2. 如果正则没匹配到，看一眼整个输出去掉空格后是不是一个单字符或单数字
        # 比如模型只回了 "A" 或 "1"
        res_clean = response.strip()
        if len(res_clean) == 1:
            token = res_clean.upper()
        else:
            token = None # 还是没法处理

    # 3. 根据提取到的 token 计算索引和最终显示的答案
    if token is not None:
        if token.isdigit():
            # 数字转索引 (如 "1" -> 0)
            pred = int(token) - 1
            filtered_answer = chr(65 + pred) if 0 <= pred < 26 else token
        elif 'A' <= token <= 'Z':
            # 字母转索引 (如 "A" -> 0)
            pred = ord(token) - ord('A')
            filtered_answer = token
        else:
            # 其他单字符（如标点）
            pred = -1
            filtered_answer = token
    else:
        # 彻底匹配失败，保留完整输出用于观察
        pred = -1
        filtered_answer = response.strip()

    return {
        "accuracy": float(pred == gt),
        "filtered_answer": filtered_answer,
        "gt_index": gt
    }

# def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
#     response = results[0]
#     gt = doc_to_target(doc)  # gt 是索引 0, 1, 2...

#     # 1. 尝试匹配
#     match = re.search(REGEX_PATTERN, response)
    
#     if match is not None:
#         token = match.group(1).upper()  # 拿到提取到的原始字符并转大写
        
#         if token.isdigit():
#             # 如果模型回的是数字 "1"，我们把它转成对应的索引 0
#             pred = int(token) - 1
#             # 为了保存的一致性，把数字转回字母展示：0 -> A, 1 -> B
#             filtered_answer = chr(65 + pred) if 0 <= pred < 26 else token
#         else:
#             # 如果模型回的是字母 "A"，转成索引 0
#             pred = ord(token) - ord('A')
#             filtered_answer = token
#     else:
#         # 2. 如果正则匹配失败
#         pred = -1
#         # 记录为 "N/A" 或者你可以记录 response 的前 20 个字符方便调试
#         filtered_answer = "N/A"

#     # 3. 返回字典
#     # 除了 accuracy，我们新增了 filtered_answer 字段
#     return {
#         "accuracy": float(pred == gt),
#         "filtered_answer": filtered_answer,  # 这里会保存 A, B, C 或 N/A
#         "gt_index": gt                       # 可选：把正确答案索引也存下来方便对比
#     }

def aggregation(results: List[float]) -> float:
    if not results:
        return 0.0
    return sum(results) / len(results)
