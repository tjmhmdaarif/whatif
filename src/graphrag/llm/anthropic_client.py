"""Anthropic LLM client implementation."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import LLMClient, LLMResponse
from .cache import get_cache
from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self):
        config = get_config()
        self.api_key = config.llm.anthropic.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = config.llm.anthropic.model
        self.default_max_tokens = config.llm.anthropic.max_tokens
        self.default_temperature = config.llm.anthropic.temperature
        self.cache = get_cache()

        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def model_name(self) -> str:
        return self.model

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Check cache first
        cached = self.cache.get(
            prompt, self.model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if cached:
            return cached

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }

        try:
            response = await self._client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data["content"][0]["text"]
            usage = data.get("usage", {})

            llm_response = LLMResponse(
                text=text,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                model=self.model,
                cached=False,
            )

            # Cache the response
            self.cache.set(
                prompt, self.model, llm_response,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return llm_response

        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Anthropic client error: {e}")
            raise

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nRespond with valid JSON only. No additional text."

        if schema:
            import json
            json_prompt += f"\n\nThe JSON must conform to this schema:\n{json.dumps(schema, indent=2)}"

        response = await self.complete(
            json_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        try:
            return orjson.loads(response.text)
        except orjson.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response.text[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    async def close(self):
        await self._client.aclose()


import orjson