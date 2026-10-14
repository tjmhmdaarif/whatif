"""Ollama LLM client implementation."""

from __future__ import annotations

from typing import Any

import httpx

from .base import LLMClient, LLMResponse
from .cache import get_cache
from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(self):
        config = get_config()
        self.base_url = config.llm.ollama.base_url
        self.model = config.llm.ollama.model
        self.default_max_tokens = config.llm.ollama.max_tokens
        self.default_temperature = config.llm.ollama.temperature
        self.cache = get_cache()

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,
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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or self.default_max_tokens,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "")

            # Ollama doesn't always return token counts, estimate them
            prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
            completion_tokens = len(text.split()) * 1.3

            llm_response = LLMResponse(
                text=text,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
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
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Ollama client error: {e}")
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