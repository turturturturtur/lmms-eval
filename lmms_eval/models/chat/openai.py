import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Union

from dotenv import load_dotenv
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.models.model_utils.superchem_official import (
    build_request as build_superchem_request,
    has_boxed_answer,
    parse_stream,
)
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleSimple
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


@register_model("openai")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def __init__(self, max_new_tokens: int = 4096, **kwargs):
        # Capture specific args for Qwen3-VL and media processing
        self.is_qwen3_vl = kwargs.get("is_qwen3_vl", False)
        # Handle cases where is_qwen3_vl is passed as a string
        if isinstance(self.is_qwen3_vl, str):
            self.is_qwen3_vl = self.is_qwen3_vl.lower() == "true"
            
        self.max_pixels = int(kwargs.get("max_pixels", 151200))
        self.min_pixels = int(kwargs.get("min_pixels", 28 * 28))
        self.max_frames = int(kwargs.get("max_frames", 768))
        self.video_fps = kwargs.get("video_fps", None)
        if self.video_fps is not None:
            self.video_fps = float(self.video_fps)
        self.max_frames_num = int(kwargs.get("max_frames_num", 64))
        super().__init__(max_new_tokens=max_new_tokens, **kwargs)

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        official_flags = [
            isinstance(req.args[2], dict)
            and req.args[2].get("official_request") == "superchem"
            for req in reordered_requests
        ]
        if any(official_flags) and not all(official_flags):
            raise ValueError(
                "SUPERChem official requests cannot be mixed with non-official "
                "requests in one OpenAI batch."
            )
        official_mode = bool(official_flags and all(official_flags))
        official_concurrency = None
        if official_mode:
            official_concurrency = int(
                reordered_requests[0].args[2]["request_concurrency"]
            )
        # Flag to print generation config only once
        _gen_config_printed = False
        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)
        total_latency = 0.0
        total_tokens = 0
        current_concurrency = (
            official_concurrency
            if official_concurrency is not None
            else min(
                self.num_concurrent,
                self.adaptive_config.max_concurrency,
            )
        )
        dispatch_order = list(range(len(reordered_requests)))
        if self.prefix_aware_queue:
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
                return "", local_index, False, False, 0.0, 0, 0, 0, "", None
            official_options = payload.pop("_superchem_official_options", None)
            retry_limit = (
                official_options.max_retries
                if official_options is not None
                else self.max_retries
            )
            retry_wait = (
                official_options.retry_initial_wait
                if official_options is not None
                else self.retry_backoff_s
            )
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"
            client_idx = local_index % len(self.clients)
            client = self.clients[client_idx]
            for attempt in range(retry_limit):
                try:
                    api_start = time.time()
                    if official_options is not None:
                        response = client.chat.completions.create(
                            timeout=official_options.timeout,
                            **payload,
                        )
                        parsed_response = parse_stream(response)
                        response_text = parsed_response.content
                        reasoning_content = parsed_response.reasoning_content
                        finish_reason = parsed_response.finish_reason
                        input_tokens = parsed_response.prompt_tokens
                        completion_tokens = parsed_response.completion_tokens
                        output_tokens = completion_tokens
                        reasoning_tokens = parsed_response.reasoning_tokens
                        if (
                            official_options.require_boxed
                            and not has_boxed_answer(response_text)
                        ):
                            raise ValueError(
                                "Could not find the answer in the required "
                                r"$\boxed{...}$ format."
                            )
                    else:
                        response = client.chat.completions.create(**payload)
                        response_text = (
                            response.choices[0].message.content or ""
                        )
                        reasoning_content = ""
                        finish_reason = getattr(
                            response.choices[0],
                            "finish_reason",
                            None,
                        )
                        input_tokens = 0
                        output_tokens = 0
                        reasoning_tokens = 0
                        if hasattr(response, "usage") and response.usage:
                            input_tokens = (
                                getattr(response.usage, "prompt_tokens", 0) or 0
                            )
                            output_tokens = (
                                getattr(response.usage, "completion_tokens", 0) or 0
                            )
                            if (
                                hasattr(response.usage, "completion_tokens_details")
                                and response.usage.completion_tokens_details
                            ):
                                reasoning_tokens = (
                                    getattr(
                                        response.usage.completion_tokens_details,
                                        "reasoning_tokens",
                                        0,
                                    )
                                    or 0
                                )
                            completion_tokens = output_tokens
                        else:
                            completion_tokens = len(response_text.split())
                            output_tokens = completion_tokens
                    api_latency = time.time() - api_start
                    eval_logger.debug(f"Request {local_index}: Preprocessing={preproc_time:.3f}s, API_Inference={api_latency:.3f}s")
                    elapsed = time.time() - started_at
                    log_usage(
                        model_name=self.model_version,
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
                        reasoning_content,
                        finish_reason,
                    )
                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg
                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{retry_limit} failed with error: {error_msg}")
                    if attempt == retry_limit - 1:
                        eval_logger.error(f"All {retry_limit} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(retry_wait)
                        if official_options is not None:
                            retry_wait += official_options.retry_increment * (attempt + 1)

            elapsed = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {retry_limit} retries] {error_preview}"
            return (
                failure_content,
                local_index,
                False,
                rate_limited,
                elapsed,
                0,
                0,
                0,
                "",
                None,
            )

        def maybe_update_concurrency(force: bool = False) -> None:
            nonlocal current_concurrency
            nonlocal failed_requests
            nonlocal rate_limited_requests
            nonlocal latencies
            nonlocal completed_since_adapt

            # The official launcher has a fixed N_PROCS x N_THREADS budget.
            # Never let lmms-eval's adaptive controller change that protocol.
            if official_mode or not self.adaptive_concurrency:
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
                    "Adaptive concurrency update: "
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
            is_superchem_official = (
                request_gen_kwargs.get("official_request") == "superchem"
            )
            max_new_tokens = (
                int(request_gen_kwargs["max_new_tokens"])
                if is_superchem_official
                else min(
                    request_gen_kwargs.get("max_new_tokens", 1024),
                    self.max_new_tokens,
                )
            )
            temperature = request_gen_kwargs.get("temperature", 0)
            top_p = request_gen_kwargs.get("top_p")
            top_k = request_gen_kwargs.get("top_k")
            presence_penalty = request_gen_kwargs.get("presence_penalty")
            frequency_penalty = request_gen_kwargs.get("frequency_penalty")

            video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
            if self.video_fps is not None and self.video_fps > 0:
                video_kwargs["fps"] = self.video_fps
            else:
                video_kwargs["nframes"] = self.max_frames_num
            
            if hasattr(self, "max_frames") and self.max_frames:
                video_kwargs["max_frames"] = self.max_frames

            if self.is_qwen3_vl:
                messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs=video_kwargs)
            else:
                messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)

            if is_superchem_official:
                payload, options = build_superchem_request(
                    model=self.model_version,
                    messages=messages,
                    generation_kwargs=request_gen_kwargs,
                )
                payload["_superchem_official_options"] = options
                if self.rank == 0 and not _gen_config_printed:
                    eval_logger.info(
                        "[Generate Config] "
                        f"task={task}, official_request=superchem, "
                        f"max_tokens={payload['max_tokens']}, "
                        f"temperature={payload['temperature']}, "
                        f"reasoning_effort={payload['reasoning_effort']}, "
                        f"stream={payload['stream']}, "
                        f"stream_options={payload['stream_options']}, "
                        f"request_timeout={options.timeout}, "
                        f"request_max_retries={options.max_retries}"
                    )
                    _gen_config_printed = True
                return payload

            payload = {
                "messages": messages,
                "model": self.model_version,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }

            # Add optional sampling parameters if provided
            if top_p is not None:
                payload["top_p"] = top_p
            # top_k is not supported by standard OpenAI API, but vLLM accepts it via extra_body
            if top_k is not None:
                payload.setdefault("extra_body", {})["top_k"] = top_k
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                payload["frequency_penalty"] = frequency_penalty

            if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version or "gpt-5" in self.model_version:
                payload.pop("temperature")
                payload.pop("max_tokens")
                payload["response_format"] = {"type": "text"}
                payload["max_completion_tokens"] = 5000

            if self.rank == 0 and not _gen_config_printed:
                eval_logger.info(f"[Generate Config] task={task}, max_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, presence_penalty={presence_penalty}, frequency_penalty={frequency_penalty}, gen_kwargs={request_gen_kwargs}")
                _gen_config_printed = True
            return payload

        def wrapped_task(local_index: int):
            pre_start = time.time()
            try:
                payload = build_payload_for_index(local_index)
                pre_time = time.time() - pre_start
                if payload is None:
                    return None, local_index, False, False, 0.0, 0, 0, 0, "", None
                return process_single_request(local_index, payload, pre_time)
            except Exception as e:
                eval_logger.error(f"Error in preprocessing request {local_index}: {e}")
                return (
                    f"[PREPROC_FAILED] {e}",
                    local_index,
                    False,
                    False,
                    time.time() - pre_start,
                    0,
                    0,
                    0,
                    "",
                    None,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    if is_budget_exceeded():
                        responses[dispatch_order[cursor]] = GenerationResult(text="[LMMS_EVAL_BUDGET_EXCEEDED]", token_counts=TokenCounts())
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
                
                # Check if it timed out to print queue status periodically
                if not done:
                    eval_logger.info(f"Queue Status | In-flight requests: {len(in_flight)} / Target concurrency: {current_concurrency} | Processing cursor: {cursor}/{len(dispatch_order)}")
                    continue
                
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
                        reasoning_content,
                        finish_reason,
                    ) = future.result()
                    in_flight.pop(future, None)
                    if response_text == "[LMMS_EVAL_BUDGET_EXCEEDED]" or success is False and response_text == "":
                        # Handle potential special cases or errors here if needed
                        pass
                    
                    responses[local_index] = GenerationResult(
                        text=str(response_text) if response_text is not None else "",
                        token_counts=TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=completion_tokens,
                            reasoning_tokens=reasoning_tokens,
                        ),
                        reasoning_content=reasoning_content,
                        finish_reason=finish_reason,
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
        return [response if response is not None else GenerationResult(text="", token_counts=TokenCounts()) for response in responses]
