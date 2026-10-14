"""Mock LLM client for testing without a real LLM."""

from __future__ import annotations

import json
from typing import Any

from .base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Mock LLM client that returns predefined responses for testing."""

    def __init__(self, responses: dict[str, Any] | None = None):
        self._responses = responses or {}
        self._call_count = 0
        self.model = "mock-model"

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
        self._call_count += 1

        # Try to find a matching response
        response_text = self._responses.get(prompt[:100], "Mock answer")

        return LLMResponse(
            text=response_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
            model=self.model,
            cached=False,
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._call_count += 1

        # Try to find a matching response from the responses dict
        for key, value in self._responses.items():
            if key in prompt or prompt.startswith(key):
                if isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        pass

        # Default fallback
        if "entities" in prompt.lower() and "relations" in prompt.lower():
            return {"entities": [], "relations": []}
        return {"answer": "Mock answer"}


def create_mock_llm_client(responses: dict[str, Any] | None = None) -> MockLLMClient:
    """Create a mock LLM client for testing."""
    return MockLLMClient(responses)