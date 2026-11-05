"""
Official Qwen3-VL MMMU Evaluation Implementation

This module replicates the evaluation logic from Qwen3-VL official repository:
https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu

Key features:
1. Rule-based answer extraction (can_infer_option + can_infer_text)
2. GPT Judge fallback when rule-based fails
3. Separate judge prompts for multi-choice and open-ended questions
4. Split-based aggregation (dev/validation)
"""

import ast
import copy
import json
import os
import random
import re
import string
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.protocol import Request

# Load task config
with open(Path(__file__).parent / "mmmu_val_qwen3_official.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

# ============================================================================
# API Configuration (Compatible with Qwen3-VL official repository)
# ============================================================================
# Supported API types and their environment variables:
#   - 'compatible': OPENAI_API_KEY + OPENAI_API_URL (yunwu.ai, etc.)
#   - 'dash': CHATGPT_DASHSCOPE_API_KEY + DASHSCOPE_API_BASE (Aliyun)
#   - 'openai': OPENAI_API_KEY + OPENAI_API_BASE
#   - 'kimi': KIMI_API_KEY + KIMI_API_BASE
#   - 'mit': MIT_SPIDER_TOKEN + MIT_SPIDER_URL
#
# Usage:
#   source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh
#   API_TYPE=compatible python -m lmms_eval ...
# ============================================================================

def get_api_config():
    """
    Get API configuration based on API_TYPE (compatible with Qwen3-VL official).
    
    Priority:
    1. Official API configuration (yunwu.ai, dashscope, etc.) - if API key is set
    2. JUDGE_BASE_URL (from start_vllm_judge_and_run.sh) - use local vLLM as fallback
    3. Raise error if neither is available
    """
    model_version = os.getenv("JUDGE_MODEL", os.getenv("MODEL_VERSION", "gpt-4o-mini"))
    api_type = os.getenv("API_TYPE", "compatible").lower()
    
    # Check if official API key is configured (Priority 1)
    api_key = None
    api_base = None
    
    if api_type == 'compatible':
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_URL", "https://yunwu.ai/v1/chat/completions")
    elif api_type == 'dash':
        api_key = os.getenv("CHATGPT_DASHSCOPE_API_KEY", "")
        api_base = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    elif api_type == 'openai':
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    elif api_type == 'kimi':
        api_key = os.getenv("KIMI_API_KEY", "")
        api_base = os.getenv("KIMI_API_BASE", "https://api.kimi.com/coding/v1/chat/completions")
    elif api_type == 'mit':
        api_key = os.getenv("MIT_SPIDER_TOKEN", "")
        api_base = os.getenv("MIT_SPIDER_URL", "")
    
    # Use official API if key is available (Priority 1)
    if api_key:
        eval_logger.info(f"Using official API ({api_type}): {api_base}")
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_URL"] = api_base.replace('/chat/completions', '') if api_base else ""
        return "openai", model_version, api_key, api_base
    
    # Check if JUDGE_BASE_URL is set (Priority 2 - fallback to local vLLM)
    judge_base_url = os.getenv("JUDGE_BASE_URL", "")
    if judge_base_url:
        eval_logger.info(f"No API key found. Using local vLLM judge backend: {judge_base_url}")
        api_key = os.getenv("JUDGE_API_KEY", "dummy-key")
        # Use first URL if multiple are provided (semicolon separated)
        api_base = judge_base_url.split(';')[0].strip()
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_URL"] = api_base
        return "openai", model_version, api_key, api_base
    
    # No configuration available (Priority 3 - error)
    raise ValueError(
        f"No API configuration found!\n\n"
        f"Please configure one of the following:\n\n"
        f"1. Official API (recommended):\n"
        f"   source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh\n\n"
        f"2. Or manually set environment variables:\n"
        f"   export API_TYPE=compatible\n"
        f"   export OPENAI_API_KEY=your-key\n"
        f"   export OPENAI_API_URL=https://yunwu.ai/v1/chat/completions\n\n"
        f"3. Or use local vLLM backend:\n"
        f"   bash examples/judge_process/start_vllm_judge_and_run.sh\n\n"
        f"Current API_TYPE: {api_type}"
    )


# Judge server configuration (initialized lazily)
_server = None
_server_config = None
_model_version = None
_server_lock = threading.Lock()


def get_judge_server():
    """Get or initialize the judge server (lazy initialization, thread-safe)."""
    global _server, _server_config, _model_version
    
    # Fast path: already initialized
    if _server is not None:
        return _server, _server_config, _model_version
    
    # Slow path: need to initialize (with lock)
    with _server_lock:
        # Double-check after acquiring lock
        if _server is None:
            api_type, model_version, api_key, api_base = get_api_config()
            _model_version = model_version
            
            if not api_key:
                raise ValueError(
                    f"API key not found for API_TYPE={api_type}.\n"
                    f"Please run: source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh\n"
                    f"Or set the appropriate environment variables."
                )
            
            _server_config = ServerConfig(
                model_name=model_version,
                temperature=0.0,
                max_tokens=16,
                timeout=int(os.getenv("JUDGE_TIMEOUT", "60")),
            )
            _server = get_server(server_name=api_type, config=_server_config)
            eval_logger.info(f"Initialized judge server: model={model_version}, api_type={api_type}")
    
    return _server, _server_config, _model_version


# ==================== Official Prompts from Qwen3-VL repo ====================

OFFICIAL_MULTI_CHOICE_JUDGE_PROMPT = """You are an AI assistant who will help me to match an answer with several options of a single-choice question. You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. If the meaning of all options are significantly different from the answer, output Z. Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. 
Example 1: 
Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A
Example 2: 
Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: Z
Example 3: 
Question: {question}?\nOptions: {options}\nAnswer: {prediction}\nYour output: """


OFFICIAL_OPEN_ENDED_JUDGE_PROMPT = """You are an expert evaluator for open-ended questions. Determine if the predicted answer contains the correct answer.

Question: {question}
Ground Truth Answer: {target}
Predicted Answer: {prediction}

Instructions:
1. Extract the final answer from the prediction (may be in reasoning like "The answer is X")
2. Check numerical equivalence ("24/7" ≈ "3.429", "100" = "100.0")
3. Check semantic equivalence (same meaning, different wording)
4. Format differences ("$100" vs "100") are acceptable
5. Respond with ONLY "CORRECT" or "INCORRECT"

Examples:
- Truth: "4", Prediction: "After calculation, the answer is 4." -> CORRECT
- Truth: "5", Prediction: "I think the answer is 6." -> INCORRECT

Your response:"""


# ==================== Document Processing Functions ====================

def replace_images_tokens(input_string: str) -> str:
    """Replace <image 1>, <image 2> etc with <image>."""
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options: List[str]) -> str:
    """Format options as A. option1\nB. option2 etc."""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Build prompt for Qwen3-VL format."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    open_ended_prompt = lmms_eval_specific_kwargs.get("open_ended_prompt", "")
    
    question = doc["question"]
    options = parse_options(ast.literal_eval(doc["options"]))
    question_type = doc["question_type"]
    
    if question_type == "multiple-choice":
        prompt = f"{pre_prompt}{question}\nOptions:\n{options}\n{post_prompt}"
    else:
        # open ended question - note: official still includes options
        prompt = f"{pre_prompt}{question}\nOptions:\n{options}\n{open_ended_prompt}"
    
    return prompt


def mmmu_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Convert document to text prompt."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    
    prompt = construct_prompt(doc, lmms_eval_specific_kwargs)
    
    if config.get("metadata", {}).get("interleaved_format", False):
        prompt = replace_images_tokens(prompt)
    
    return prompt


def mmmu_doc_to_visual(doc: Dict) -> List:
    """Extract visual elements from document."""
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and swap space as _
    image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


def mmmu_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Convert document to message format with official Qwen3-VL settings."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    
    # Official image resolution settings (match Qwen3-VL official repo)
    min_pixels = lmms_eval_specific_kwargs.get("min_pixels", 1280 * 28 * 28)  # ~1M pixels
    max_pixels = lmms_eval_specific_kwargs.get("max_pixels", 5120 * 28 * 28)  # ~4M pixels
    
    # If you use doc to messages, the interleaved format is always used
    prompt = mmmu_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = mmmu_doc_to_visual(doc)
    
    messages = [{"role": "user", "content": []}]
    interleaved_content = prompt.split("<image>")
    
    for i, (image, text) in enumerate(zip(visuals, interleaved_content)):
        if text.strip() != "":
            messages[0]["content"].append({"type": "text", "text": text.strip()})
        # Add image with official resolution settings
        messages[0]["content"].append({
            "type": "image",
            "url": image,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels
        })
    
    # There will be one more text part after the last image
    if len(interleaved_content) > 0:
        messages[0]["content"].append({"type": "text", "text": interleaved_content[-1].strip()})
    
    return messages


# ==================== Official Rule-Based Extraction ====================

def can_infer_option(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Official implementation: Extract option letter from answer.
    Returns the option letter if exactly one is found, None otherwise.
    Special return 'Z' indicates refusal to answer.
    """
    if 'Failed to obtain answer via API' in answer:
        return None
    
    # Rejection patterns
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'
    
    def count_choice(splits: List[str], choices: Dict[str, str], prefix: str = '', suffix: str = '') -> int:
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt
    
    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')
    
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    
    if count == 1:
        for ch in choices:
            if ch in splits:
                # Guard against 'A' being a quantifier, only when matching 'A'
                if ch == 'A' and len(splits) > 3:
                    return False
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    
    return False


def can_infer_text(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Official implementation: Extract option by matching text content.
    Returns the option letter if exactly one option text matches.
    """
    answer_lower = answer.lower()
    
    # Guard: skip if answer is much longer than all options combined
    total_option_len = sum(len(str(v)) for v in choices.values())
    if len(answer) > 2 * total_option_len:
        return None
    
    cands = []
    for k, v in choices.items():
        if str(v).lower() in answer_lower:
            cands.append(k)
    
    if len(cands) == 1:
        return cands[0]
    return None


def can_infer(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """Combined approach to infer answer choice (official implementation)."""
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    # Note: 'Z' is a valid return (truthy) indicating refusal to answer
    # False/None are falsy and will trigger can_infer_text
    return copt if copt else can_infer_text(answer, choices)


def build_choices(doc: Dict) -> Dict[str, str]:
    """Build choices dictionary from document."""
    ret = {}
    
    # Try to get from cached options first
    cached = doc.get('mmmu_acc_official') or doc.get('mmmu_acc')
    if cached and isinstance(cached, dict) and 'options' in cached:
        return cached['options']
    
    # Build from doc fields
    for ch in string.ascii_uppercase:
        if ch in doc and doc[ch] is not None:
            val = doc[ch]
            # Handle pandas/numpy types
            if hasattr(val, 'item'):
                val = val.item()
            if val is not None and str(val) != 'nan' and str(val) != '':
                ret[ch] = str(val)
    
    # If no choices found, try to infer from input text
    if not ret:
        input_text = doc.get('input', '')
        # Look for patterns like "A. option\nB. option"
        for ch in string.ascii_uppercase[:8]:  # A-H
            pattern = rf'\n{ch}\.\s*(.+?)(?=\n[A-H]\.|\n\n|$)'
            match = re.search(pattern, input_text)
            if match:
                ret[ch] = match.group(1).strip()
    
    return ret


def build_option_str(option_dict: Dict[str, str]) -> str:
    """Build option string for judge prompt."""
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if content and str(content) != 'nan':
            s += f'{c}. {content}\n'
    return s


def build_official_judge_prompt(question: str, options: str, prediction: str) -> str:
    """Build the official multi-choice judge prompt."""
    return OFFICIAL_MULTI_CHOICE_JUDGE_PROMPT.format(
        question=question,
        options=options,
        prediction=prediction
    )


def build_official_open_ended_judge_prompt(question: str, target: str, prediction: str) -> str:
    """Build the official open-ended judge prompt."""
    return OFFICIAL_OPEN_ENDED_JUDGE_PROMPT.format(
        question=question,
        target=target,
        prediction=prediction
    )


# ==================== GPT Judge Integration ====================

def call_judge_for_extraction(
    doc: Dict,
    prediction: str,
    choices: Dict[str, str],
    max_retries: int = 5,   # Reduced to avoid overwhelming the API
    wait_time: float = 3.0
) -> Tuple[str, str, bool]:
    """
    Call GPT judge to extract answer when rule-based fails.
    Returns: (extracted_option, log_message, success_flag)
    """
    server, srv_config, model_version = get_judge_server()
    options_str = build_option_str(choices)
    
    # Get question from doc - handle different formats
    question = doc.get('question', '')
    if not question:
        # Try to extract from input field
        input_text = doc.get('input', '')
        # Extract question part (before options)
        if 'Options:' in input_text:
            question = input_text.split('Options:')[0].strip()
        elif '\nA.' in input_text:
            question = input_text.split('\nA.')[0].strip()
        else:
            question = input_text[:200]  # Use first 200 chars
    
    if not question:
        # Fallback: use prediction as context
        question = "Multiple choice question"
    
    prompt = build_official_judge_prompt(question, options_str, prediction)
    
    retry = max_retries
    while retry > 0:
        try:
            # Use the judge server via Request/Response protocol
            request = Request(
                messages=[{"role": "user", "content": prompt}],
                config=srv_config,
                allowed_outputs=list(dict.fromkeys([*choices, 'Z'])),
            )
            response = server.evaluate(request)
            ans = response.content.strip()
            
            if not ans or 'Failed to obtain answer' in ans:
                eval_logger.warning(f'Judge API failed to answer.')
                retry -= 1
                time.sleep(wait_time * (2 ** (max_retries - retry)) + random.random())
                continue
            
            # Try to extract option from judge response
            ret = can_infer(ans, choices)
            # Z is the prompt's valid "no matching option" result.
            if ret:
                log = f'Judge {model_version} extract Succeed. {model_version}:{ans}\n'
                return ret, log, True
            else:
                eval_logger.warning(
                    f"Invalid judge extraction for sample {doc.get('doc_id', doc.get('id', 'unknown'))}, "
                    f"attempt {max_retries - retry + 1}/{max_retries}, response_chars={len(ans)}, "
                    f"usage={getattr(response, 'usage', None)}, response={ans[:160]!r} ... {ans[-160:]!r}"
                )
                retry -= 1
                time.sleep(wait_time * (2 ** (max_retries - retry)) + random.random())
                
        except Exception as e:
            eval_logger.error(f'Error calling judge: {e}')
            retry -= 1
            time.sleep(wait_time * (2 ** (max_retries - retry)) + random.random())
    
    # All retries failed, random guess
    options = list(choices.keys()) + ['Z'] if 'Z' not in choices else list(choices.keys())
    ret = random.choice(options)
    log = f'Judge {model_version} extract failed after {max_retries} retries. Randomly generate one.'
    return ret, log, False


def call_judge_for_open_ended(
    doc: Dict,
    prediction: str,
    max_retries: int = 3,
    wait_time: float = 3.0
) -> Tuple[Optional[bool], str, bool]:
    """
    Call GPT judge for open-ended question.
    Returns: (is_correct, log_message, success_flag)
    """
    server, srv_config, model_version = get_judge_server()
    
    # Get question and answer from doc - handle different formats
    question = doc.get('question', '')
    if not question:
        input_text = doc.get('input', '')
        question = input_text[:200] if input_text else "Open-ended question"
    
    answer = doc.get('answer', doc.get('target', ''))
    
    prompt = build_official_open_ended_judge_prompt(
        question,
        answer,
        prediction
    )
    
    retry = max_retries
    while retry > 0:
        try:
            # Use the judge server via Request/Response protocol
            request = Request(
                messages=[{"role": "user", "content": prompt}],
                config=srv_config
            )
            response = server.evaluate(request)
            ans = response.content.strip().upper()
            
            if 'CORRECT' in ans and 'INCORRECT' not in ans:
                return True, f'Judge: CORRECT', True
            elif 'INCORRECT' in ans:
                return False, f'Judge: INCORRECT', True
            else:
                eval_logger.debug(f'Judge returned unclear response: {ans}')
                retry -= 1
                time.sleep(wait_time * (2 ** (max_retries - retry)) + random.random())
                
        except Exception as e:
            eval_logger.error(f'Error calling judge: {e}')
            retry -= 1
            time.sleep(wait_time * (2 ** (max_retries - retry)) + random.random())
    
    # Fallback to rule-based
    return None, 'Judge failed, fallback to rule-based', False


# ==================== Official Process Results ====================

def extract_answer_official(doc: Dict, prediction: str) -> Dict[str, Any]:
    """
    Official implementation of answer extraction.
    Two-stage: rule-based -> GPT judge fallback
    """
    choices = build_choices(doc)
    
    # Stage 1: Rule-based extraction
    ret = can_infer(prediction, choices)
    
    if ret:
        if ret == 'Z':
            return {
                'opt': ret,
                'log': f"Rule extract failed with rule result: {ret} prediction: {prediction}",
                'extract_model': 'rule',
                'extract_flag': False
            }
        else:
            return {
                'opt': ret,
                'log': f"Rule extract success with rule result: {ret} prediction: {prediction}",
                'extract_model': 'rule',
                'extract_flag': True
            }
    
    # Stage 2: GPT judge fallback
    doc_id = doc.get('id', doc.get('index', doc.get('doc_id', 'unknown')))
    eval_logger.debug(f"Rule extract failed. Use model-based extraction for {doc_id}")
    
    opt, log, success = call_judge_for_extraction(doc, prediction, choices)
    
    # Get model version for logging
    _, _, model_version = get_judge_server()
    
    return {
        'opt': opt,
        'log': log,
        'extract_model': model_version if success else 'random',
        'extract_flag': success
    }


def _extract_doc_from_sample(sample: Dict) -> Dict:
    """
    Extract/build complete doc information from JSONL sample.
    This ensures we have all necessary fields for official evaluation.
    """
    doc = {}
    
    # Basic identifiers
    doc['id'] = sample.get('doc_id', sample.get('id', 'unknown'))
    doc['index'] = sample.get('doc_id', sample.get('id', 'unknown'))
    doc['doc_id'] = sample.get('doc_id', sample.get('id', 'unknown'))
    
    # Answer (ground truth)
    doc['answer'] = sample.get('target', sample.get('answer', ''))
    doc['target'] = sample.get('target', sample.get('answer', ''))
    
    # Extract question and options from input field
    input_text = sample.get('input', '')
    
    # Parse input to extract question and options
    # Format: "<image> Question text\nA. option1\nB. option2...\n\nAnswer with..."
    if input_text:
        # Remove image tags
        clean_input = re.sub(r'<image\s*\d*>', '', input_text).strip()
        
        # Remove trailing prompt instructions
        prompt_endings = [
            "\n\nAnswer with the option's letter",
            "\n\nAnswer with the option letter",
            "\n\nPlease select the correct answer",
            "\n\nAnswer the question",
        ]
        for ending in prompt_endings:
            if ending in clean_input:
                clean_input = clean_input.split(ending)[0].strip()
        
        # Extract question (before options)
        if '\nA.' in clean_input:
            parts = clean_input.split('\nA.', 1)
            question_part = parts[0].strip()
            doc['question'] = question_part
            
            # Extract options
            options_text = '\nA.' + parts[1]
            for ch in string.ascii_uppercase[:8]:  # A-H
                next_ch = chr(ord(ch) + 1)
                if next_ch == 'I':
                    # Last option - match to end
                    pattern = rf'\n{ch}\.\s*(.+)$'
                else:
                    pattern = rf'\n{ch}\.\s*(.+?)(?=\n{next_ch}\.|$)'
                match = re.search(pattern, options_text, re.DOTALL)
                if match:
                    doc[ch] = match.group(1).strip()
        else:
            doc['question'] = clean_input[:500]  # Use first 500 chars as question
    
    # Determine question type
    has_options = any(ch in doc for ch in string.ascii_uppercase[:4])
    doc['question_type'] = 'multiple-choice' if has_options else 'open-ended'
    
    # Try to get split from id or sample
    doc_id = doc['id']
    
    # First check if sample has explicit split info
    if 'split' in sample:
        doc['split'] = sample['split']
    elif isinstance(doc_id, str):
        if doc_id.startswith('dev_'):
            doc['split'] = 'dev'
        elif doc_id.startswith('validation_'):
            doc['split'] = 'validation'
        else:
            doc['split'] = 'unknown'
    else:
        # Try to infer from mmmu_acc cached data
        cached = sample.get('mmmu_acc', {})
        if isinstance(cached, dict) and 'id' in cached:
            cached_id = cached['id']
            if isinstance(cached_id, str):
                if cached_id.startswith('dev_'):
                    doc['split'] = 'dev'
                elif cached_id.startswith('validation_'):
                    doc['split'] = 'validation'
                else:
                    doc['split'] = 'unknown'
            else:
                doc['split'] = 'unknown'
        else:
            doc['split'] = 'unknown'
    
    return doc


def mmmu_process_results_official(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Official implementation of result processing - GENERATION ONLY.
    
    This function ONLY saves raw generation results. NO judge is called.
    Judging must be done separately using: 
        python -m lmms_eval judge --input_result <result.jsonl> --task mmmu_val_qwen3_official
    
    This separation allows:
    1. Pure generation without API dependencies
    2. Reproducible judging with different judge models
    3. Clear separation of concerns
    """
    # Check if doc contains sample context from standalone.py (judge mode)
    if doc and "__sample_context__" in doc:
        # In judge mode, perform full evaluation
        sample = doc["__sample_context__"]
        doc = _extract_doc_from_sample(sample)
        return _mmmu_process_results_with_judge(doc, results)
    
    # Generation mode: only save raw results, NO judging
    gt_answer = doc.get('answer', doc.get('target', ''))
    raw_pred = results[0] if results else ""
    
    # Build minimal result dictionary with raw prediction only
    result_data = {
        'id': doc.get('id', doc.get('index', doc.get('doc_id', 'unknown'))),
        'split': doc.get('split', 'unknown'),
        'question_type': doc.get('question_type', 'unknown'),
        'answer': gt_answer,
        'raw_pred': raw_pred,  # Original model output, NOT extracted
        'parsed_pred': None,   # Will be filled by judge later
        'extraction_method': 'pending_judge',
        'extraction_success': False,
        'hit': None,  # Will be calculated by judge later
    }
    
    return {
        'mmmu_acc_official': result_data,
        'submission': {str(doc.get('id', 'unknown')): raw_pred}
    }


def _mmmu_process_results_with_judge(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Internal function: performs full evaluation WITH judge.
    Called only during separate judge phase (when __sample_context__ is present).
    """
    # Full evaluation with official logic
    parsed_preds = []
    extraction_logs = []
    extraction_methods = []
    extraction_success = []
    
    for pred in results:
        result = extract_answer_official(doc, pred)
        parsed_preds.append(result['opt'])
        extraction_logs.append(result['log'])
        extraction_methods.append(result['extract_model'])
        extraction_success.append(result['extract_flag'])
    
    gt_answer = doc.get('answer', doc.get('target', ''))
    
    result_data = {
        'id': doc.get('id', doc.get('index', doc.get('doc_id', 'unknown'))),
        'split': doc.get('split', 'unknown'),
        'question_type': doc.get('question_type', 'unknown'),
        'question': doc.get('question', ''),
        'answer': gt_answer,
        'parsed_pred': parsed_preds,
        'extraction_log': extraction_logs[0] if extraction_logs else '',
        'extraction_method': extraction_methods[0] if extraction_methods else 'unknown',
        'extraction_success': extraction_success[0] if extraction_success else False,
        'options': build_choices(doc)
    }
    
    hit = 1 if parsed_preds[0] == gt_answer else 0
    result_data['hit'] = hit
    
    return {
        'mmmu_acc_official': result_data,
        'submission': {str(doc.get('id', 'unknown')): parsed_preds[0]}
    }


# ==================== Official Aggregation ====================

def mmmu_aggregate_results_official(results: List[Dict]) -> Dict[str, Any]:
    """
    Official implementation of result aggregation.
    
    In GENERATION mode: only counts samples, skips accuracy calculation.
    In JUDGE mode: calculates accuracy by split.
    """
    if not results:
        return {'overall_accuracy': 0.0}
    
    # Check if we're in generation mode (hit is None) or judge mode
    sample_hit = results[0].get('hit')
    is_generation_mode = sample_hit is None
    
    # Group by split
    results_by_split = defaultdict(list)
    for result in results:
        split = result.get('split', 'unknown')
        results_by_split[split].append(result)
    
    if is_generation_mode:
        # Generation mode: only report counts, no accuracy
        eval_logger.info("=" * 50)
        eval_logger.info("MMMU Generation Complete (Judge Pending)")
        eval_logger.info("=" * 50)
        for split, split_results in sorted(results_by_split.items()):
            eval_logger.info(f"Split '{split}': {len(split_results)} samples generated")
        eval_logger.info(f"Total: {len(results)} samples generated")
        eval_logger.info("-" * 50)
        eval_logger.info("To evaluate results, run:")
        eval_logger.info(f"  python -m lmms_eval judge --input_result <path/to/result.jsonl> --task mmmu_val_qwen3_official")
        eval_logger.info("=" * 50)
        
        return {
            'overall_accuracy': None,  # Not calculated in generation mode
            'accuracy_by_split': {},
            'total_correct': None,
            'total_samples': len(results),
            'mode': 'generation_only'
        }
    
    # Judge mode: calculate accuracy per split
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        total_hits = sum(r.get('hit', 0) for r in split_results)
        total_samples = len(split_results)
        accuracy = total_hits / total_samples if total_samples > 0 else 0.0
        accuracy_by_split[split] = {
            'accuracy': round(accuracy, 4),
            'correct': total_hits,
            'total': total_samples
        }
    
    # Calculate overall accuracy
    total_hits = sum(r.get('hit', 0) for r in results)
    total_samples = len(results)
    overall_accuracy = total_hits / total_samples if total_samples > 0 else 0.0
    
    # Print results (matching official format)
    eval_logger.info("=" * 50)
    eval_logger.info("Official MMMU Evaluation Results:")
    eval_logger.info("=" * 50)
    for split, acc_data in sorted(accuracy_by_split.items()):
        eval_logger.info(f"Accuracy for {split} split: {acc_data['accuracy']:.4f} ({acc_data['correct']}/{acc_data['total']})")
    eval_logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({total_hits}/{total_samples})")
    eval_logger.info("=" * 50)
    
    return {
        'overall_accuracy': round(overall_accuracy, 4),
        'accuracy_by_split': accuracy_by_split,
        'total_correct': total_hits,
        'total_samples': total_samples,
        'mode': 'judged'
    }


# ==================== Standalone Judge Mode Support ====================

def mmmu_qwen3_official_aggregate_accuracy(extracted_data: List[Dict]) -> float:
    """
    Adapter function for aggregator.
    
    Args:
        extracted_data: List of mmmu_acc_official dicts from samples
        
    Returns:
        Accuracy as percentage (0-100), or -1.0 if in generation mode
    """
    if not extracted_data:
        return 0.0
    
    # Check if in generation mode (hit is None)
    sample_hit = extracted_data[0].get('hit')
    if sample_hit is None:
        eval_logger.info("Generation mode: accuracy not calculated yet")
        return -1.0  # Signal that judging is needed
    
    total_hits = sum(d.get('hit', 0) for d in extracted_data)
    total_samples = len(extracted_data)
    accuracy = (total_hits / total_samples) * 100 if total_samples > 0 else 0.0
    
    # Also log split-based accuracy for debugging
    split_counts = {}
    for d in extracted_data:
        split = d.get('split', 'unknown')
        if split not in split_counts:
            split_counts[split] = {'hits': 0, 'total': 0}
        split_counts[split]['hits'] += d.get('hit', 0)
        split_counts[split]['total'] += 1
    
    for split, counts in sorted(split_counts.items()):
        split_acc = (counts['hits'] / counts['total']) * 100 if counts['total'] > 0 else 0
        eval_logger.info(f"  {split}: {split_acc:.2f}% ({counts['hits']}/{counts['total']})")
    
    return accuracy


def run_official_judge_on_file(
    input_file: str,
    output_file: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Standalone function to run official judge on a result file.
    Compatible with lmms-eval judge command.
    """
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_judged_official.jsonl')
    
    # Load results
    eval_logger.info(f"Loading results from: {input_file}")
    with open(input_file, 'r') as f:
        data_list = [json.loads(line.strip()) for line in f]
    
    eval_logger.info(f"Total {len(data_list)} samples to evaluate")
    eval_logger.info(f"Judge model: {judge_model}")
    
    # Process each sample
    processed_results = []
    for data in data_list:
        doc = data.get('doc', data.get('annotation', {}))
        predictions = data.get('filtered_resps', [data.get('result', {}).get('gen', '')])
        
        result = mmmu_process_results_official(doc, predictions)
        processed_results.append(result['mmmu_acc_official'])
    
    # Aggregate
    agg_result = mmmu_aggregate_results_official(processed_results)
    
    # Save results
    eval_logger.info(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for res in processed_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    # Save summary
    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(agg_result, f, indent=2)
    
    eval_logger.info(f"Summary saved to: {summary_file}")
    
    return agg_result
