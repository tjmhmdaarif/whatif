"""Base LLM client interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cached: bool = False


class LLMClient(abc.ABC):
    """Abstract base class for LLM clients."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abc.abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass

    @abc.abstractmethod
    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON completion for the given prompt."""
        pass