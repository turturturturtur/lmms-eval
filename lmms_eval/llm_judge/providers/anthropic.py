"""
Anthropic-compatible API provider for lmms-eval Judge.

Supports Kimi Coding API and other Anthropic-compatible endpoints.
"""

import itertools
import os
import threading
import time
from typing import Dict, List, Optional, Union

import requests
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64
from lmms_eval.models.model_utils.usage_metrics import log_usage

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class AnthropicProvider(ServerInterface):
    """Anthropic API implementation of the Judge interface.

    Supports multiple backends via semicolon-separated URLs in ANTHROPIC_API_URL.
    """

    _in_flight = 0
    _in_flight_lock = threading.Lock()

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        raw_api_url = os.getenv("ANTHROPIC_API_URL") or os.getenv("OPENAI_API_URL") or ""
        # Strip trailing /messages so we can append it correctly
        if raw_api_url.endswith("/messages"):
            raw_api_url = raw_api_url[: -len("/messages")]
        self.api_urls = [u.strip() for u in raw_api_url.split(";") if u.strip()]

        self._url_cycle = itertools.cycle(self.api_urls) if self.api_urls else None

    def _next_url(self):
        if self._url_cycle is None:
            raise RuntimeError("No Anthropic API URLs available")
        return next(self._url_cycle)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: Request) -> Response:
        """Evaluate using Anthropic-compatible API"""
        if not self.is_available():
            raise ValueError("Anthropic API key not configured")

        with AnthropicProvider._in_flight_lock:
            AnthropicProvider._in_flight += 1
            in_flight = AnthropicProvider._in_flight

        started_at = time.time()
        try:
            config = request.config or self.config
            messages = self.prepare_messages(request)

            # Handle images if present
            if request.images:
                messages = self._add_images_to_messages(messages, request.images)

            # Anthropic format payload
            payload = {
                "model": config.model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
            }

            if config.top_p is not None:
                payload["top_p"] = config.top_p

            # Make API call with retries
            last_exception = None
            for attempt in range(config.num_retries):
                try:
                    url = self._next_url()
                    response = self._make_request(payload, config.timeout, url)
                    content = response["content"][0]["text"]
                    model_used = response.get("model", config.model_name)
                    usage_raw = response.get("usage", {})
                    usage = {
                        "prompt_tokens": usage_raw.get("input_tokens", 0),
                        "completion_tokens": usage_raw.get("output_tokens", 0),
                    }
                    raw_response = response

                    latency = time.time() - started_at
                    input_tokens = usage.get("prompt_tokens", 0) or 0
                    output_tokens = usage.get("completion_tokens", 0) or 0
                    log_usage(
                        model_name=model_used or config.model_name,
                        task_name=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=0,
                        source="judge",
                    )

                    eval_logger.debug(
                        f"[Judge] in={input_tokens}, out={output_tokens}, "
                        f"fly={in_flight}, latency={latency:.3f}s, model={model_used or config.model_name}"
                    )

                    return Response(
                        content=content.strip(),
                        model_used=model_used,
                        usage=usage,
                        raw_response=raw_response,
                    )

                except Exception as e:
                    last_exception = e
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        time.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise last_exception
        finally:
            with AnthropicProvider._in_flight_lock:
                AnthropicProvider._in_flight -= 1

    def _make_request(self, payload: Dict, timeout: int, url: Optional[str] = None) -> Dict:
        """Make HTTP request to Anthropic-compatible API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        target_url = url if url is not None else self.api_urls[0]
        # Ensure URL ends with /messages
        if not target_url.endswith("/messages"):
            target_url = target_url.rstrip("/") + "/messages"

        response = requests.post(target_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to the last user message"""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                if isinstance(messages[i]["content"], str):
                    content_list = [{"type": "text", "text": messages[i]["content"]}]
                else:
                    content_list = messages[i]["content"]

                for image in images:
                    if isinstance(image, str):
                        base64_image = self._encode_image(image)
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            }
                        })
                    elif isinstance(image, bytes):
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image.decode(),
                            }
                        })

                messages[i]["content"] = content_list
                break

        return messages

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        return encode_image_to_base64(
            image_path,
            image_format="JPEG",
            convert_rgb=True,
            quality=85,
            use_path_cache=True,
        )
