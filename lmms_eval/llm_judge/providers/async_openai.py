import asyncio
import os
from typing import Dict, List, Optional, Union

import aiohttp
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.models.model_utils.usage_metrics import log_usage

from ..base import AsyncServerInterface
from ..protocol import Request, Response, ServerConfig
from .openai import OpenAIProvider, build_sampling_kwargs


class AsyncOpenAIProvider(AsyncServerInterface):
    """Async OpenAI API implementation of the Judge interface.

    Supports multiple backends via semicolon-separated URLs in OPENAI_API_URL,
    e.g. http://localhost:8000/v1;http://localhost:8001/v1
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY") or ""
        raw_api_url = os.getenv("OPENAI_API_URL") or ""
        # Strip trailing /chat/completions so the OpenAI client can append it correctly
        if raw_api_url.endswith("/chat/completions"):
            raw_api_url = raw_api_url[: -len("/chat/completions")]
        self.api_urls = [u.strip() for u in raw_api_url.split(";") if u.strip()]

        self.async_clients = []
        self.use_async_client = False
        try:
            from openai import AsyncOpenAI

            for url in self.api_urls:
                self.async_clients.append(AsyncOpenAI(api_key=self.api_key, base_url=url))
            self.use_async_client = True
        except ImportError:
            eval_logger.warning("AsyncOpenAI client not available, using aiohttp")

        self._client_idx = 0
        self._client_lock = asyncio.Lock()

    async def _next_client(self):
        if not self.async_clients:
            raise RuntimeError("No AsyncOpenAI clients available")
        async with self._client_lock:
            client = self.async_clients[self._client_idx]
            self._client_idx = (self._client_idx + 1) % len(self.async_clients)
            return client

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def evaluate_async(self, request: Request) -> Response:
        """Evaluate using OpenAI API asynchronously"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")

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
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens

        if config.top_p is not None:
            payload["top_p"] = config.top_p

        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        # Make API call with retries
        last_exception = None
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    if self.use_async_client:
                        client = await self._next_client()
                        response = await client.chat.completions.create(**payload)
                        content = response.choices[0].message.content
                        model_used = response.model
                        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                        raw_response = response
                    else:
                        url = self.api_urls[attempt % len(self.api_urls)]
                        response = await self._make_async_request(payload, config.timeout, url)
                        content = response["choices"][0]["message"]["content"]
                        model_used = response["model"]
                        usage = response.get("usage")
                        raw_response = response

                    # Log usage for token tracking
                    if self.use_async_client and hasattr(response, "usage") and response.usage:
                        log_usage(
                            model_name=model_used or config.model_name,
                            task_name=None,
                            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                            reasoning_tokens=0,
                            source="judge",
                        )
                    elif not self.use_async_client and isinstance(usage, dict):
                        log_usage(
                            model_name=model_used or config.model_name,
                            task_name=None,
                            input_tokens=usage.get("prompt_tokens", 0) or 0,
                            output_tokens=usage.get("completion_tokens", 0) or 0,
                            reasoning_tokens=0,
                            source="judge",
                        )

                    return Response(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    last_exception = e
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise last_exception

    async def _make_async_request(self, payload: Dict, timeout: int, url: Optional[str] = None) -> Dict:
        """Make async HTTP request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        target_url = url if url is not None else self.api_urls[0]
        async with aiohttp.ClientSession() as session:
            async with session.post(target_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes, Image.Image]]) -> List[Dict]:
        """Add images to messages - reuse from base implementation"""
        return OpenAIProvider._add_images_to_messages(self, messages, images)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 - reuse from base implementation"""
        return OpenAIProvider._encode_image(self, image_path)
