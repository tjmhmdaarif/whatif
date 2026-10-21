"""End-to-end tests against a stand-in Ollama server.

A real ``http.server`` is bound to 127.0.0.1 so the full network path
(``urllib`` -> HTTP -> JSON) is exercised without needing Ollama installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from whatif.cli import main
from whatif.config import Config, create_default_config
from whatif.embeddings import OllamaClient
from whatif.store import DecisionStore


def fake_vector(text: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-embedding derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Minimal Ollama-compatible request handler."""

    calls: list[tuple[str, dict[str, Any]]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr logging."""
        return

    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/tags":
            self._send(
                {
                    "models": [
                        {"name": "llama3.1:latest"},
                        {"name": "nomic-embed-text:latest"},
                    ]
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8")) if raw else {}

        if self.path == "/api/embed":
            text_input = payload.get("input", "")
            texts = text_input if isinstance(text_input, list) else [text_input]
            self._send({"embeddings": [fake_vector(str(t)) for t in texts]})
        elif self.path == "/api/embeddings":
            self._send({"embedding": fake_vector(str(payload.get("prompt", "")))})
        elif self.path == "/api/generate":
            self._send({"response": "A generated answer. [ESTIMATE]"})
        else:
            self._send({"error": "not found"}, status=404)


@pytest.fixture
def fake_ollama():
    """Start a stand-in Ollama server on a free localhost port."""
    FakeOllamaHandler.calls = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"
    yield host
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def seeded_vault(temp_dir: Path) -> Path:
    """A vault populated with the 10 seed decision notes."""
    vault = temp_dir / "vault"
    (vault / "Decisions").mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "decisions"
    for note in source.glob("*.md"):
        (vault / "Decisions" / note.name).write_text(
            note.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return vault


def write_config(vault: Path, ollama_host: str) -> Path:
    """Write a whatif.toml pointing at the stand-in server."""
    config_path = vault / "whatif.toml"
    config_path.write_text(
        f"""[vault]
path = "{vault.as_posix()}"
decisions_dir = "Decisions"
briefs_dir = "Briefs"
canvases_dir = "Canvases"
whatif_dir = ".whatif"

[models]
llm_model = "llama3.1"
embed_model = "nomic-embed-text"

[ollama]
host = "{ollama_host}"
timeout = 30

[retrieval]
default_k = 5
similarity_threshold = 0.3
failed_assumption_threshold = 0.6

[brief]
max_retries = 1
legend = "Legend"

[logging]
level = "INFO"
format = "%(levelname)s | %(name)s | %(message)s"
""",
        encoding="utf-8",
    )
    return config_path


class TestAgainstStandInServer:
    def test_client_lists_models(self, fake_ollama: str) -> None:
        client = OllamaClient(host=fake_ollama, model="llama3.1")
        assert client.is_reachable() is True
        assert "llama3.1:latest" in client.list_models()
        assert client.has_model("nomic-embed-text") is True

    def test_client_embeds(self, fake_ollama: str) -> None:
        client = OllamaClient(host=fake_ollama, model="nomic-embed-text")
        vector = client.embed("some decision text")
        assert len(vector) == 32
        assert all(isinstance(x, float) for x in vector)

    def test_client_generates(self, fake_ollama: str) -> None:
        client = OllamaClient(host=fake_ollama, model="llama3.1")
        assert "ESTIMATE" in client.generate("hello")

    def test_index_command_end_to_end(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)

        exit_code = main(["index", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.err
        assert "Indexed: 10" in captured.out
        assert "Skipped: 0" in captured.out
        assert "Failed: 0" in captured.out

        store = DecisionStore(seeded_vault / ".whatif" / "index.sqlite")
        try:
            assert store.count() == 10
            assert store.count_embeddings() == 10
        finally:
            store.close()

    def test_index_command_skips_unchanged(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)

        main(["index", "--config", str(config_path)])
        capsys.readouterr()

        exit_code = main(["index", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Indexed: 0" in captured.out
        assert "Skipped: 10" in captured.out

    def test_index_command_reports_failures(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        (seeded_vault / "Decisions" / "broken.md").write_text(
            "---\nid: [unclosed\n---\n", encoding="utf-8"
        )

        exit_code = main(["index", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Failed: 1" in captured.out
        assert "broken.md" in captured.err

    def test_doctor_passes_against_stand_in(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        # Build the index so the database check passes too.
        main(["index", "--config", str(config_path)])
        capsys.readouterr()

        exit_code = main(["doctor", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.out
        assert "[OK] Ollama server" in captured.out
        assert "[OK] Model: llama3.1" in captured.out
        assert "[OK] Model: nomic-embed-text" in captured.out
        assert "[OK] Index database" in captured.out

    def test_doctor_fails_when_server_down(
        self, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        config_path = write_config(seeded_vault, f"http://127.0.0.1:{dead_port}")

        exit_code = main(["doctor", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "[FAIL] Ollama server" in captured.out

    def test_index_surfaces_unreachable_server(
        self, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A dead Ollama must produce a clear error, not a stack trace."""
        # Bind a port and release it so nothing is listening there.
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        config_path = write_config(seeded_vault, f"http://127.0.0.1:{dead_port}")

        exit_code = main(["index", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Cannot reach Ollama" in captured.err
        assert "Traceback" not in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
