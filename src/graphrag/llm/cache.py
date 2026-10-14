"""Disk cache for LLM calls."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import orjson

from .base import LLMResponse
from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class LLMCache:
    """Disk-based cache for LLM responses."""

    def __init__(self, cache_dir: str | Path | None = None):
        config = get_config()
        self.cache_dir = Path(cache_dir or config.llm.cache.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.llm.cache.enabled

    def _make_key(self, prompt: str, model: str, **kwargs: Any) -> str:
        """Generate a cache key from prompt and parameters."""
        # Include relevant parameters in the key
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens"),
            "system_prompt": kwargs.get("system_prompt"),
        }
        key_str = orjson.dumps(key_data, option=orjson.OPT_SORT_KEYS).decode()
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse | None:
        """Get a cached response if available."""
        if not self.enabled:
            return None

        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                data = orjson.loads(f.read())

            # Verify the prompt matches (defense against hash collisions)
            if data.get("prompt") != prompt:
                logger.warning("Cache key collision detected, ignoring cached response")
                return None

            response = LLMResponse(
                text=data["text"],
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
                total_tokens=data.get("total_tokens", 0),
                model=data.get("model", model),
                cached=True,
            )
            logger.debug(f"Cache hit for key {key[:8]}...")
            return response
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def set(self, prompt: str, model: str, response: LLMResponse, **kwargs: Any) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return

        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        data = {
            "prompt": prompt,
            "model": model,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens"),
            "system_prompt": kwargs.get("system_prompt"),
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "timestamp": time.time(),
        }

        try:
            with open(cache_file, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            logger.debug(f"Cached response for key {key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def clear(self) -> None:
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass
        logger.info("Cache cleared")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_cache: LLMCache | None = None


def get_cache() -> LLMCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache