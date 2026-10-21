"""Embedding and generation clients for the local Ollama server.

Every network call in WhatIf goes through this module, and every call is directed at
the single host configured in ``whatif.toml``. No other network access exists in the
codebase.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence, runtime_checkable

from whatif.config import Config

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class ModelNotFoundError(OllamaError):
    """Raised when a configured model is not present on the Ollama server."""


@runtime_checkable
class Embedder(Protocol):
    """Callable that turns text into a vector embedding.

    Tests substitute a deterministic fake implementation so that nothing in the
    test suite requires a running Ollama server.
    """

    def __call__(self, text: str) -> list[float]:
        """Return the embedding vector for the given text."""
        ...


class OllamaClient:
    """Minimal HTTP client for a local Ollama server.

    Args:
        host: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
        model: Default model used for embeddings and generation.
        timeout: Request timeout in seconds.
    """

    def __init__(self, host: str, model: str = "llama3.1", timeout: int = 120) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- internals ---------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        """Build an absolute URL for an Ollama endpoint on the configured host."""
        return f"{self.host}{endpoint}"

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the Ollama server and return the decoded response."""
        url = self._url(endpoint)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.debug("POST %s payload_keys=%s", url, sorted(payload.keys()))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise OllamaError(f"Ollama returned HTTP {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is the server running? ({exc.reason})"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama request to {url} timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise OllamaError(f"Network error talking to Ollama at {self.host}: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON from {url}: {body[:200]}") from exc

    def _get(self, endpoint: str) -> dict[str, Any]:
        """GET JSON from the Ollama server and return the decoded response."""
        url = self._url(endpoint)
        request = urllib.request.Request(url, method="GET")
        logger.debug("GET %s", url)
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 15)) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise OllamaError(f"Ollama returned HTTP {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is the server running? ({exc.reason})"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama request to {url} timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise OllamaError(f"Network error talking to Ollama at {self.host}: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON from {url}: {body[:200]}") from exc

    # -- public API --------------------------------------------------------

    def is_reachable(self) -> bool:
        """Return True if the server answers the tags endpoint."""
        try:
            self._get("/api/tags")
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[str]:
        """Return the names of models installed on the server."""
        data = self._get("/api/tags")
        models = data.get("models", [])
        names: list[str] = []
        for model in models:
            name = model.get("name") or model.get("model")
            if name:
                names.append(name)
        return names

    def has_model(self, model: str) -> bool:
        """Return True if the given model is installed on the server."""
        names = self.list_models()
        # Ollama reports "llama3.1:latest" for a model requested as "llama3.1".
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)

    def require_model(self, model: str) -> None:
        """Raise ModelNotFoundError if the model is not installed."""
        if not self.has_model(model):
            available = ", ".join(self.list_models()) or "none"
            raise ModelNotFoundError(
                f"Model '{model}' is not installed on {self.host}. "
                f"Install it with: ollama pull {model}. Available models: {available}"
            )

    def _embed_legacy(self, text: str, model: str) -> list[float]:
        """Embed via the older single-prompt ``/api/embeddings`` endpoint."""
        data = self._post("/api/embeddings", {"model": model, "prompt": text})
        embedding = data.get("embedding")
        if not embedding:
            raise OllamaError(f"Ollama returned no embedding for model '{model}'")
        return [float(x) for x in embedding]

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embed a single string.

        Prefers the batch ``/api/embed`` endpoint and falls back to the older
        single-prompt ``/api/embeddings`` endpoint.
        """
        model = model or self.model
        try:
            data = self._post("/api/embed", {"model": model, "input": text})
            embeddings = data.get("embeddings")
            if embeddings:
                return [float(x) for x in embeddings[0]]
        except OllamaError as exc:
            logger.debug("Batch embed endpoint failed, trying legacy endpoint: %s", exc)

        return self._embed_legacy(text, model)

    def embed_batch(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        """Embed several strings, falling back to one-at-a-time on failure."""
        model = model or self.model
        if not texts:
            return []
        try:
            data = self._post("/api/embed", {"model": model, "input": list(texts)})
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return [[float(x) for x in vec] for vec in embeddings]
        except OllamaError as exc:
            logger.debug("Batch embed endpoint failed, embedding one at a time: %s", exc)

        # Go straight to the legacy endpoint: the batch one already failed.
        return [self._embed_legacy(text, model) for text in texts]

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion for a prompt.

        Args:
            prompt: The user prompt.
            model: Model override. Defaults to the client's model.
            system: Optional system prompt.
            options: Optional Ollama sampling options.

        Returns:
            The generated text.
        """
        model = model or self.model
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        data = self._post("/api/generate", payload)
        text = data.get("response")
        if text is None:
            raise OllamaError(f"Ollama returned no response field for model '{model}'")
        return str(text)


class OllamaEmbedder:
    """Adapter turning an :class:`OllamaClient` into an :class:`Embedder`."""

    def __init__(self, client: OllamaClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.model

    def __call__(self, text: str) -> list[float]:
        return self.client.embed(text, model=self.model)


def build_client(config: Config) -> OllamaClient:
    """Create an OllamaClient from configuration."""
    return OllamaClient(
        host=config.ollama.host,
        model=config.models.embed_model,
        timeout=config.ollama.timeout,
    )


def build_embedder(config: Config) -> OllamaEmbedder:
    """Create the default embedder from configuration."""
    client = build_client(config)
    return OllamaEmbedder(client, model=config.models.embed_model)
