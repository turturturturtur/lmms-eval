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


class OpenAIProvider(ServerInterface):
    """OpenAI API implementation of the Judge interface.

    Supports multiple backends via semicolon-separated URLs in OPENAI_API_URL,
    e.g. http://localhost:8000/v1;http://localhost:8001/v1
    """

    _in_flight = 0
    _in_flight_lock = threading.Lock()

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY") or ""
        raw_api_url = os.getenv("OPENAI_API_URL") or ""
        # Strip trailing /chat/completions so the OpenAI client can append it correctly
        if raw_api_url.endswith("/chat/completions"):
            raw_api_url = raw_api_url[: -len("/chat/completions")]
        self.api_urls = [u.strip() for u in raw_api_url.split(";") if u.strip()]

        self.clients = []
        self.use_client = False
        try:
            from openai import OpenAI

            for url in self.api_urls:
                self.clients.append(OpenAI(api_key=self.api_key, base_url=url))
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests")

        self._client_cycle = itertools.cycle(self.clients) if self.clients else None

    def _next_client(self):
        if self._client_cycle is None:
            raise RuntimeError("No OpenAI clients available")
        return next(self._client_cycle)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: Request) -> Response:
        """Evaluate using OpenAI API"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")

        with OpenAIProvider._in_flight_lock:
            OpenAIProvider._in_flight += 1
            in_flight = OpenAIProvider._in_flight

        started_at = time.time()
        try:
            config = request.config or self.config
            messages = self.prepare_messages(request)

            # Handle images if present
            if request.images:
                messages = self._add_images_to_messages(messages, request.images)

            # Prepare payload
            payload = {
                "model": config.model_name,
                "messages": messages,
            }
            disable_thinking = os.getenv("JUDGE_DISABLE_THINKING", os.getenv("AUTOMIX_DISABLE_THINKING", "")).lower() in {"1", "true", "yes"}
            extra_body = None
            if disable_thinking:
                extra_body = {
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            if config.max_tokens is not None:
                payload["max_tokens"] = config.max_tokens

            if config.top_p is not None:
                payload["top_p"] = config.top_p

            if config.response_format == "json":
                payload["response_format"] = {"type": "json_object"}

            # Make API call with retries (across URLs for resilience)
            last_exception = None
            urls_attempted = set()
            for attempt in range(config.num_retries):
                try:
                    if self.use_client:
                        client = self._next_client()
                        client_payload = dict(payload)
                        if extra_body is not None:
                            client_payload["extra_body"] = extra_body
                        response = client.chat.completions.create(timeout=config.timeout, **client_payload)
                        message = response.choices[0].message
                        content = message.content or getattr(message, "reasoning_content", "") or ""
                        model_used = response.model
                        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                        raw_response = response
                    else:
                        if extra_body is not None:
                            payload.update(extra_body)
                        url = self.api_urls[attempt % len(self.api_urls)]
                        response = self._make_request(payload, config.timeout, url)
                        message = response["choices"][0]["message"]
                        content = message.get("content") or message.get("reasoning_content") or ""
                        model_used = response["model"]
                        usage = response.get("usage")
                        raw_response = response

                    latency = time.time() - started_at
                    input_tokens = 0
                    output_tokens = 0
                    if self.use_client and hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                        log_usage(
                            model_name=model_used or config.model_name,
                            task_name=None,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            reasoning_tokens=0,
                            source="judge",
                        )
                    elif not self.use_client and isinstance(usage, dict):
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

                    return Response(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    last_exception = e
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        time.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise last_exception
        finally:
            with OpenAIProvider._in_flight_lock:
                OpenAIProvider._in_flight -= 1

    def _make_request(self, payload: Dict, timeout: int, url: Optional[str] = None) -> Dict:
        """Make HTTP request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        target_url = url if url is not None else self.api_urls[0]
        response = requests.post(target_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to the last user message"""
        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # Convert content to list format if needed
                if isinstance(messages[i]["content"], str):
                    messages[i]["content"] = [{"type": "text", "text": messages[i]["content"]}]

                # Add images
                for image in images:
                    if isinstance(image, str):
                        # File path
                        base64_image = self._encode_image(image)
                        messages[i]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                    elif isinstance(image, bytes):
                        # Already base64 encoded
                        messages[i]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image.decode()}"}})
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
