"""Tests for embeddings.py - Ollama client and network guard."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from whatif.embeddings import (
    Embedder,
    ModelNotFoundError,
    OllamaClient,
    OllamaEmbedder,
    build_client,
    build_embedder,
)


def http_error(code: int, message: str) -> urllib.error.HTTPError:
    """Build an HTTPError with a readable body."""
    return urllib.error.HTTPError(
        "http://localhost:11434/api", code, message, {}, io.BytesIO(b"")
    )


class FakeResponse:
    """Minimal stand-in for the object returned by urlopen."""

    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class RecordingUrlopen:
    """Records every URL requested, so tests can assert on network targets."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.responses = responses or {}
        self.default: Any = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "embedding": [0.1, 0.2, 0.3],
            "response": "generated text",
            "models": [],
        }

    def __call__(self, request: Any, timeout: float | None = None) -> FakeResponse:
        url = request.full_url if hasattr(request, "full_url") else str(request)
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        payload = None
        if hasattr(request, "data") and request.data:
            payload = json.loads(request.data.decode("utf-8"))
        self.calls.append((url, method, payload))

        path = url.split("://", 1)[-1].split("/", 1)[-1]

        # Exact path match wins, so "api/embed" never shadows "api/embeddings".
        for key, value in self.responses.items():
            if path == key:
                if isinstance(value, Exception):
                    raise value
                return FakeResponse(value)

        for key, value in self.responses.items():
            if key in path:
                if isinstance(value, Exception):
                    raise value
                return FakeResponse(value)
        return FakeResponse(self.default)


class TestOllamaClientUrls:
    def test_all_calls_target_configured_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen({"api/tags": {"models": [{"name": "llama3.1"}]}})
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="llama3.1")
        client.list_models()
        client.embed("hello")
        client.generate("hi")

        assert len(recorder.calls) == 3
        for url, _, _ in recorder.calls:
            assert url.startswith("http://localhost:11434/")

    def test_custom_host_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen()
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://192.168.1.50:11434", model="m")
        client.embed("hello")

        assert recorder.calls[0][0] == "http://192.168.1.50:11434/api/embed"

    def test_trailing_slash_normalised(self) -> None:
        client = OllamaClient(host="http://localhost:11434/", model="m")
        assert client.host == "http://localhost:11434"

    def test_build_client_from_config(self, config_path) -> None:
        from whatif.config import load_config

        config = load_config(config_path)
        client = build_client(config)
        assert client.host == "http://localhost:11434"
        assert client.model == "nomic-embed-text"

    def test_build_embedder_from_config(self, config_path) -> None:
        from whatif.config import load_config

        config = load_config(config_path)
        embedder = build_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model == "nomic-embed-text"


class TestOllamaClientEmbed:
    def test_embed_uses_batch_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen({"api/embed": {"embeddings": [[1.0, 2.0, 3.0]]}})
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="nomic-embed-text")
        vector = client.embed("some text")

        assert vector == [1.0, 2.0, 3.0]
        url, method, payload = recorder.calls[0]
        assert url.endswith("/api/embed")
        assert method == "POST"
        assert payload == {"model": "nomic-embed-text", "input": "some text"}

    def test_embed_falls_back_to_legacy_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = RecordingUrlopen(
            {
                "api/embed": http_error(404, "not found"),
                "api/embeddings": {"embedding": [0.5, 0.5]},
            }
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        vector = client.embed("text")

        assert vector == [0.5, 0.5]
        assert len(recorder.calls) == 2
        assert recorder.calls[0][0].endswith("/api/embed")
        assert recorder.calls[1][0].endswith("/api/embeddings")

    def test_embed_batch_single_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"api/embed": {"embeddings": [[1.0], [2.0], [3.0]]}}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        vectors = client.embed_batch(["a", "b", "c"])

        assert vectors == [[1.0], [2.0], [3.0]]
        assert len(recorder.calls) == 1

    def test_embed_batch_empty_input(self) -> None:
        client = OllamaClient(host="http://localhost:11434", model="m")
        assert client.embed_batch([]) == []

    def test_embed_batch_falls_back_to_individual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = RecordingUrlopen(
            {
                "api/embed": http_error(500, "boom"),
                "api/embeddings": {"embedding": [0.1, 0.2]},
            }
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        vectors = client.embed_batch(["a", "b"])

        assert vectors == [[0.1, 0.2], [0.1, 0.2]]
        # 1 failed batch call + 2 legacy single calls.
        assert len(recorder.calls) == 3
        assert recorder.calls[1][0].endswith("/api/embeddings")

    def test_embed_missing_embedding_field_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {
                "api/embed": http_error(404, "nf"),
                "api/embeddings": {"nothing": []},
            }
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        with pytest.raises(Exception, match="no embedding"):
            client.embed("text")

    def test_embed_returns_floats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen({"api/embed": {"embeddings": [[1, 2, 3]]}})
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        vector = client.embed("text")
        assert all(isinstance(x, float) for x in vector)


class TestOllamaClientErrors:
    def test_server_down_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"": urllib.error.URLError("Connection refused")}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        with pytest.raises(Exception, match="Cannot reach Ollama"):
            client.list_models()

    def test_is_reachable_false_when_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"": urllib.error.URLError("Connection refused")}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        assert client.is_reachable() is False

    def test_is_reachable_true_when_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen({"api/tags": {"models": []}})
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        assert client.is_reachable() is True

    def test_invalid_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def bad_urlopen(request: Any, timeout: float | None = None) -> FakeResponse:
            class Raw:
                def read(self_inner: Any) -> bytes:
                    return b"not json"

                def __enter__(self_inner: Any) -> Any:
                    return self_inner

                def __exit__(self_inner: Any, *exc: object) -> None:
                    return None

            return Raw()

        monkeypatch.setattr(urllib.request, "urlopen", bad_urlopen)
        client = OllamaClient(host="http://localhost:11434", model="m")
        with pytest.raises(Exception, match="invalid JSON"):
            client.list_models()


class TestModelDetection:
    def test_list_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"api/tags": {"models": [{"name": "llama3.1:latest"}, {"name": "nomic-embed-text"}]}}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        assert client.list_models() == ["llama3.1:latest", "nomic-embed-text"]

    def test_has_model_ignores_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"api/tags": {"models": [{"name": "llama3.1:latest"}]}}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        assert client.has_model("llama3.1") is True
        assert client.has_model("llama3.2") is False

    def test_require_model_raises_with_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = RecordingUrlopen(
            {"api/tags": {"models": [{"name": "llama3.1:latest"}]}}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            client.require_model("nomic-embed-text")

    def test_require_model_passes_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen(
            {"api/tags": {"models": [{"name": "nomic-embed-text:latest"}]}}
        )
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        client.require_model("nomic-embed-text")


class TestOllamaEmbedder:
    def test_calls_client_embed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingUrlopen({"api/embed": {"embeddings": [[0.1, 0.2]]}})
        monkeypatch.setattr(urllib.request, "urlopen", recorder)

        client = OllamaClient(host="http://localhost:11434", model="m")
        embedder = OllamaEmbedder(client, model="nomic-embed-text")
        vector = embedder("hello")

        assert vector == [0.1, 0.2]
        assert recorder.calls[0][2]["model"] == "nomic-embed-text"

    def test_satisfies_embedder_protocol(self) -> None:
        client = OllamaClient(host="http://localhost:11434", model="m")
        embedder = OllamaEmbedder(client)
        assert isinstance(embedder, Embedder)


class TestNetworkGuard:
    """No module outside embeddings.py may open a network connection."""

    def test_only_embeddings_module_imports_urlopen(self) -> None:
        from pathlib import Path

        package_dir = Path(__file__).resolve().parent.parent / "whatif"
        offenders = []
        for py_file in package_dir.glob("*.py"):
            if py_file.name == "embeddings.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            if "urllib.request" in text or "urlopen" in text:
                offenders.append(py_file.name)
        assert not offenders, (
            f"Network access outside whatif/embeddings.py in: {offenders}. "
            "All HTTP calls must go through the Ollama client."
        )

    def test_no_cloud_hosts_in_source(self) -> None:
        from pathlib import Path

        package_dir = Path(__file__).resolve().parent.parent / "whatif"
        banned = ("api.openai.com", "api.anthropic.com", "generativelanguage", "huggingface.co")
        for py_file in package_dir.glob("*.py"):
            text = py_file.read_text(encoding="utf-8").lower()
            for host in banned:
                assert host not in text, f"{py_file.name} references cloud host {host}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
