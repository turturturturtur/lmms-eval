"""
VLLM Backend Model - Native HTTP API support for vLLM

This model uses native HTTP requests to directly access vLLM's OpenAI-compatible API,
supporting all vLLM-specific parameters including:
- top_k (not supported by standard OpenAI API)
- repetition_penalty
- min_p
- skip_special_tokens
- And other vLLM-specific sampling parameters

Usage:
    --model vllm-backend \
    --model_args "base_url=http://localhost:8000/v1,model=Qwen3-VL-8B-Instruct,num_concurrent=128" \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.1,max_new_tokens=49152"
"""

import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Optional, Union

import requests
from requests.exceptions import HTTPError
import torch
from accelerate import Accelerator, DistributedType

from dotenv import load_dotenv
from loguru import logger as eval_logger

from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


class VLLMBackendRequestError(RuntimeError):
    pass


def _normalize_until(until):
    if until is None:
        return None
    if isinstance(until, str):
        if not until:
            raise ValueError("gen_kwargs['until'] must not be an empty string")
        return until
    if not isinstance(until, list):
        raise ValueError(
            "gen_kwargs['until'] must be None, a non-empty string, or a list of non-empty strings; "
            f"got {type(until).__name__}"
        )
    if any(not isinstance(item, str) or not item for item in until):
        raise ValueError("gen_kwargs['until'] must contain only non-empty strings")
    return list(until)


def _parse_stop_token_ids(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError("stop_token_ids must be a non-empty JSON array of non-negative integers")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("stop_token_ids must be valid JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("stop_token_ids must be a non-empty JSON array of non-negative integers")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in value):
        raise ValueError("stop_token_ids must contain only non-negative integers")
    return list(value)


@register_model("vllm_backend")
class VLLMBackend(lmms):
    """
    VLLM Backend model using native HTTP requests.
    
    Supports all vLLM-specific sampling parameters that are not available
    in the standard OpenAI API, such as:
    - top_k: Integer to limit the top-k tokens to sample from
    - repetition_penalty: Penalty for repeating tokens
    - min_p: Minimum probability for nucleus sampling
    - skip_special_tokens: Whether to skip special tokens in output
    
    Args:
        base_url: vLLM API base URL (e.g., http://localhost:8000/v1)
        model: Model name/path (must match what vLLM was started with)
        api_key: API key (vLLM default is "EMPTY" or not required)
        timeout: Request timeout in seconds
        retry_backoff_s: Backoff time between retries
        max_retries: Maximum number of retries per request
        fail_fast_on_request_error: Raise after one request exhausts retries
        num_concurrent: Number of concurrent requests
        adaptive_concurrency: Whether to use adaptive concurrency control
        adaptive_max_concurrency: Maximum concurrency for adaptive mode
        max_new_tokens: Maximum new tokens limit (default: 4096)
        max_pixels: Maximum pixels for image processing
        min_pixels: Minimum pixels for image processing
        max_frames: Maximum frames for video processing
        video_fps: Frames per second for video processing
        max_frames_num: Maximum number of frames to extract from video
        is_qwen3_vl: Whether the model is Qwen3-VL
        enable_thinking: Whether to pass Qwen chat_template_kwargs.enable_thinking
        prefix_aware_queue: Whether to use prefix-aware queue ordering
        shuffle_requests: Whether to randomly shuffle requests before dispatch
        stop_token_ids: Optional JSON array of model-specific token IDs that stop generation
    """
    
    is_simple = False

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "",
        api_key: Optional[str] = "EMPTY",
        timeout: int = 300,
        retry_backoff_s: float = 1.0,
        max_retries: int = 5,
        fail_fast_on_request_error: bool = True,
        num_concurrent: int = 32,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        max_new_tokens: int = 4096,
        max_pixels: int = 151200,
        min_pixels: int = 28 * 28,
        max_frames: int = 768,
        video_fps: Optional[float] = None,
        max_frames_num: int = 64,
        is_qwen3_vl: bool = False,
        enable_thinking: Optional[Union[bool, str]] = None,
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        chat_template: Optional[str] = None,
        shuffle_requests: bool = False,
        stop_token_ids: Optional[Union[str, List[int]]] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Disable colors for this model instance only
        os.environ.setdefault('LOGURU_NO_COLOR', '1')
        os.environ.setdefault('NO_COLOR', '1')
        os.environ.setdefault('FORCE_COLOR', '0')
        tqdm.disable_color = True
        
        # Handle base_url - support multiple URLs separated by semicolon
        if ";" in base_url:
            self.base_urls = [url.strip() for url in base_url.split(";")]
        else:
            self.base_urls = [base_url]
        
        self.model_name = model
        self.api_key = api_key or "EMPTY"
        self.timeout = float(timeout)
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.max_retries = int(max_retries)
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout!r}")
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {max_retries!r}")
        self.fail_fast_on_request_error = parse_bool(fail_fast_on_request_error)
        self.num_concurrent = max(1, int(num_concurrent))
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.max_new_tokens = int(max_new_tokens)
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)
        self.max_frames = int(max_frames)
        self.video_fps = float(video_fps) if video_fps is not None else None
        self.max_frames_num = int(max_frames_num)
        self.is_qwen3_vl = is_qwen3_vl if not isinstance(is_qwen3_vl, str) else is_qwen3_vl.lower() == "true"
        if enable_thinking is None or str(enable_thinking).strip().lower() in {"", "null", "none"}:
            self.enable_thinking = None
        else:
            self.enable_thinking = parse_bool(enable_thinking)
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        self.chat_template = chat_template
        self.shuffle_requests = parse_bool(shuffle_requests)
        self.stop_token_ids = _parse_stop_token_ids(stop_token_ids)
        
        # Initialize session for connection pooling
        from requests.adapters import HTTPAdapter
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "EMPTY":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Initialize accelerator for distributed setup (similar to simple/openai.py)
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        
        self.device = self.accelerator.device
        
        eval_logger.info(f"VLLM Backend initialized with {len(self.base_urls)} endpoint(s): {self.base_urls}")
        eval_logger.info(f"Model: {self.model_name}, Max new tokens: {self.max_new_tokens}")

    def _get_api_url(self, index: int) -> str:
        """Get API URL for given request index (round-robin)."""
        return self.base_urls[index % len(self.base_urls)]

    def apply_chat_template(self, messages: list) -> str:
        """
        Apply chat template to messages for few-shot context construction.
        
        For VLLMBackend, this returns a simple string representation of messages
        since the actual template rendering is done by the vLLM server.
        This method is primarily used for cache key generation and logging.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            String representation of the chat context
        """
        # Simple format: role: content
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Handle multimodal content (list of dicts)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image":
                        text_parts.append("<image>")
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        text_parts.append("<image>")
                content = " ".join(text_parts)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _make_request(self, payload: dict, url: str) -> dict:
        """Make HTTP request to vLLM API."""
        response = self.session.post(
            f"{url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        request_stops = []
        for request in reordered_requests:
            gen_kwargs = request.args[2]
            if not isinstance(gen_kwargs, dict):
                raise ValueError(f"generation kwargs must be a dict, got {type(gen_kwargs).__name__}")
            request_stops.append(_normalize_until(gen_kwargs.get("until")))
        _gen_config_printed = False
        
        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self._rank != 0),
            desc="VLLM Backend Responding",
        )

        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)
        total_latency = 0.0
        total_tokens = 0
        current_concurrency = min(
            self.num_concurrent,
            self.adaptive_config.max_concurrency,
        )
        
        dispatch_order = list(range(len(reordered_requests)))
        if self.shuffle_requests:
            random.shuffle(dispatch_order)
        elif self.prefix_aware_queue:
            prefix_hashes = {}
            for idx in dispatch_order:
                req = reordered_requests[idx]
                prefix_text = req.args[0] if isinstance(req.args[0], str) else ""
                if not prefix_text:
                    _, doc_to_messages, _, doc_id, task, split = req.args
                    chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                    prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
                prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
            dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
        
        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(
            1,
            self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency,
        )

        def process_single_request(local_index: int, payload: dict | None, preproc_time: float):
            if payload is None:
                return "", local_index, False, False, 0.0, 0, 0, 0
            
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"
            url = self._get_api_url(local_index)
            
            for attempt in range(self.max_retries):
                try:
                    api_start = time.time()
                    result = self._make_request(payload, url)
                    api_latency = time.time() - api_start
                    
                    eval_logger.debug(f"[Rank {self._rank}] Request {local_index}: Preprocessing={preproc_time:.3f}s, API_Inference={api_latency:.3f}s")
                    elapsed = time.time() - started_at
                    
                    # Extract response
                    response_text = result["choices"][0]["message"]["content"]
                    
                    # Extract token usage
                    usage = result.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    completion_tokens = output_tokens
                    
                    # Try to extract reasoning tokens if available
                    reasoning_tokens = 0
                    completion_tokens_details = usage.get("completion_tokens_details", {})
                    if completion_tokens_details:
                        reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
                    
                    log_usage(
                        model_name=self.model_name,
                        task_name=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        source="model",
                    )
                    
                    return (
                        response_text,
                        local_index,
                        True,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    )
                    
                except HTTPError as exc:
                    error_msg = f"HTTP {exc.response.status_code}: {exc.response.text}"
                    last_error_msg = error_msg
                    rate_limited = exc.response.status_code == 429
                    
                    eval_logger.info(f"[Rank {self._rank}] Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(self.retry_backoff_s)
                        
                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg
                    rate_limited = is_rate_limit_error(error_msg)
                    
                    eval_logger.info(f"[Rank {self._rank}] Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(self.retry_backoff_s)

            elapsed = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            if self.fail_fast_on_request_error:
                raise VLLMBackendRequestError(
                    f"vLLM request failed after {self.max_retries} retries: {error_preview}"
                )
            failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
            return failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0

        def maybe_update_concurrency(force: bool = False) -> None:
            nonlocal current_concurrency
            nonlocal failed_requests
            nonlocal rate_limited_requests
            nonlocal latencies
            nonlocal completed_since_adapt

            if not self.adaptive_concurrency:
                return

            sample_threshold = max(4, current_concurrency)
            if not force and completed_since_adapt < sample_threshold:
                return
            if completed_since_adapt <= 0:
                return

            decision = decide_next_concurrency(
                current_concurrency=current_concurrency,
                total_requests=completed_since_adapt,
                failed_requests=failed_requests,
                rate_limited_requests=rate_limited_requests,
                latencies=latencies,
                config=self.adaptive_config,
            )
            if decision.next_concurrency != decision.current_concurrency:
                eval_logger.info(
                    f"[Rank {self._rank}] Adaptive concurrency update: "
                    f"{decision.current_concurrency} -> "
                    f"{decision.next_concurrency} "
                    f"(fail_rate={decision.failure_rate:.3f}, "
                    f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                    f"p95_latency={decision.p95_latency_s:.3f}s)"
                )
            current_concurrency = decision.next_concurrency
            failed_requests = 0
            rate_limited_requests = 0
            latencies = []
            completed_since_adapt = 0

        def build_payload_for_index(global_index: int) -> dict:
            nonlocal _gen_config_printed
            req = reordered_requests[global_index]
            _, doc_to_messages, gen_kwargs, doc_id, task, split = req.args

            chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages_raw})
            request_gen_kwargs = dict(gen_kwargs)
            stop = request_stops[global_index]
            
            # Extract video kwargs
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.video_fps is not None and self.video_fps > 0:
                video_kwargs["fps"] = self.video_fps
            else:
                video_kwargs["nframes"] = self.max_frames_num
            
            if self.max_frames:
                video_kwargs["max_frames"] = self.max_frames

            # Convert to OpenAI format messages
            if self.is_qwen3_vl:
                messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs=video_kwargs)
            else:
                messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)

            # Build payload with all vLLM-supported parameters
            # Standard OpenAI API parameters
            max_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), self.max_new_tokens)
            temperature = request_gen_kwargs.get("temperature", 0)
            top_p = request_gen_kwargs.get("top_p")
            presence_penalty = request_gen_kwargs.get("presence_penalty")
            frequency_penalty = request_gen_kwargs.get("frequency_penalty")
            
            # vLLM-specific parameters (not in standard OpenAI API)
            top_k = request_gen_kwargs.get("top_k")
            repetition_penalty = request_gen_kwargs.get("repetition_penalty")
            min_p = request_gen_kwargs.get("min_p")
            skip_special_tokens = request_gen_kwargs.get("skip_special_tokens")
            
            # Build payload - include all parameters directly
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add optional parameters
            if top_p is not None:
                payload["top_p"] = top_p
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                payload["frequency_penalty"] = frequency_penalty
                
            # Add vLLM-specific parameters
            if top_k is not None:
                payload["top_k"] = top_k
            if repetition_penalty is not None:
                payload["repetition_penalty"] = repetition_penalty
            if min_p is not None:
                payload["min_p"] = min_p
            if skip_special_tokens is not None:
                payload["skip_special_tokens"] = skip_special_tokens
            if stop is not None:
                payload["stop"] = stop
            if self.stop_token_ids is not None:
                payload["stop_token_ids"] = list(self.stop_token_ids)
            if self.enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            
            # Log generation config once
            if self._rank == 0 and not _gen_config_printed:
                eval_logger.info(
                    f"[Generate Config] task={task}, max_tokens={max_tokens}, "
                    f"temperature={temperature}, top_p={top_p}, top_k={top_k}, "
                    f"repetition_penalty={repetition_penalty}, min_p={min_p}, "
                    f"presence_penalty={presence_penalty}, frequency_penalty={frequency_penalty}, "
                    f"stop={stop}, stop_token_ids={self.stop_token_ids}, "
                    f"enable_thinking={self.enable_thinking}, "
                    f"gen_kwargs={request_gen_kwargs}"
                )
                _gen_config_printed = True
                
            return payload

        def wrapped_task(local_index: int):
            pre_start = time.time()
            try:
                payload = build_payload_for_index(local_index)
                pre_time = time.time() - pre_start
                if payload is None:
                    return None, local_index, False, False, 0.0, 0, 0, 0
                return process_single_request(local_index, payload, pre_time)
            except VLLMBackendRequestError:
                raise
            except Exception as e:
                import traceback
                eval_logger.error(f"Error in preprocessing request {local_index}: {e}")
                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                return f"[PREPROC_FAILED] {e}", local_index, False, False, time.time() - pre_start, 0, 0, 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            no_progress_count = 0
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    if is_budget_exceeded():
                        responses[dispatch_order[cursor]] = GenerationResult(
                            text="[LMMS_EVAL_BUDGET_EXCEEDED]",
                            token_counts=TokenCounts()
                        )
                        pbar.update(1)
                        cursor += 1
                        continue

                    request_index = dispatch_order[cursor]
                    future = executor.submit(wrapped_task, request_index)
                    in_flight[future] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED, timeout=1.0)
                
                if not done:
                    no_progress_count += 1
                    if no_progress_count % 10 == 0:  # 每10秒打印一次
                        eval_logger.debug(
                            f"[Rank {self._rank}] Queue Status | In-flight requests: {len(in_flight)} / "
                            f"Target concurrency: {current_concurrency} | "
                            f"Processing cursor: {cursor}/{len(dispatch_order)}"
                        )
                    continue
                else:
                    no_progress_count = 0
                
                for future in done:
                    (
                        response_text,
                        local_index,
                        success,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    ) = future.result()
                    in_flight.pop(future, None)
                    
                    responses[local_index] = GenerationResult(
                        text=str(response_text) if response_text is not None else "",
                        token_counts=TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=completion_tokens,
                            reasoning_tokens=reasoning_tokens,
                        ),
                    )
                    total_latency += elapsed
                    total_tokens += completion_tokens
                    latencies.append(elapsed)
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    completed_since_adapt += 1
                    totals = get_running_totals()
                    pbar.set_postfix({"tokens": f"{totals['total_tokens']:,}"}, refresh=False)
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

        maybe_update_concurrency(force=True)

        avg_speed = total_tokens / total_latency if total_latency > 0 else 0
        log_metrics(
            total_elapsed_time=total_latency,
            total_gen_tokens=total_tokens,
            avg_speed=avg_speed,
        )

        pbar.close()
        return [
            response if response is not None else GenerationResult(text="", token_counts=TokenCounts())
            for response in responses
        ]

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for VLLMBackend")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("generate_until_multi_round not implemented for VLLMBackend")
