"""LLM client factory."""

from __future__ import annotations

from .base import LLMClient
from .anthropic_client import AnthropicClient
from .ollama_client import OllamaClient
from .mock_client import MockLLMClient, create_mock_llm_client
from .extractive_client import ExtractiveClient, create_extractive_client
from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)

# Global mock client for testing
_mock_client: MockLLMClient | None = None


def create_llm_client() -> LLMClient:
    """Create an LLM client based on configuration.

    A client installed via :func:`set_mock_client` always wins, so tests can
    inject fixtures regardless of the configured provider.

    Returns:
        The configured LLM client.
    """
    global _mock_client
    if _mock_client is not None:
        return _mock_client

    config = get_config()

    if config.llm.provider == "anthropic":
        logger.info("Creating Anthropic client")
        return AnthropicClient()
    elif config.llm.provider == "ollama":
        logger.info("Creating Ollama client")
        return OllamaClient()
    elif config.llm.provider == "extractive":
        logger.info("Creating Extractive client")
        return ExtractiveClient()
    elif config.llm.provider == "mock":
        logger.info("Creating Mock client")
        _mock_client = create_mock_llm_client()
        return _mock_client
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm.provider}")


def set_mock_client(client: MockLLMClient) -> None:
    """Set a mock client for testing."""
    global _mock_client
    _mock_client = client


__all__ = [
    "create_llm_client",
    "LLMClient",
    "MockLLMClient",
    "create_mock_llm_client",
    "set_mock_client",
    "ExtractiveClient",
    "create_extractive_client",
]