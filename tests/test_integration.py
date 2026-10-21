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


def generated_brief(prompt: str) -> str:
    """A canned brief that cites a note id the prompt actually allowed."""
    allowed: list[str] = []
    marker = "=== NOTE IDS YOU MAY CITE ==="
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
        for line in tail.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                break
            allowed = [part.strip() for part in line.split(",") if part.strip()]
            break

    cited = allowed[0] if allowed else "2026-01-15-course-choice"
    relevance = f"{cited}: a past decision on a similar tradeoff" if allowed else ""

    return f"""<decision_summary>
The user is choosing between two options that differ on cost and time [ESTIMATE]
A similar tradeoff was decided before [EVIDENCE: {cited}]
</decision_summary>
<similar_notes>
{relevance}
</similar_notes>
<counterargument>
A similar decision did not pay off last time [EVIDENCE: {cited}]
</counterargument>
<open_questions>
What is the real budget? [ESTIMATE]
</open_questions>"""


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
            self._send({"response": generated_brief(str(payload.get("prompt", "")))})
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
legend = \"\"\"> **Legend**
>
> - `[EVIDENCE: note-id]` - a claim supported by a retrieved decision
> - `[ESTIMATE]` - a judgement with no supporting evidence
> - `[UNVERIFIED]` - a claim that failed validation
> - `COMPUTED` - a value calculated in code
\"\"\"

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

    def test_brief_command_end_to_end(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        main(["index", "--config", str(config_path)])
        capsys.readouterr()

        note = seeded_vault / "Decisions" / "2026-09-14-side-project-build.md"
        exit_code = main(["brief", str(note), "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.err
        brief_path = seeded_vault / "Briefs" / "2026-09-14-side-project-build.md"
        assert brief_path.exists()

        content = brief_path.read_text(encoding="utf-8")
        assert "# Decision brief: 2026-09-14-side-project-build" in content
        assert "[EVIDENCE: note-id]" in content
        assert "## Similar past decisions" in content
        assert "## Assumptions that failed before" in content
        assert "## Weighted comparison" in content
        assert "## Counterargument" in content
        assert "## Open questions" in content
        assert "Total (COMPUTED)" in content
        # The brief must not contain a percentage.
        assert "%" not in content

    def test_brief_command_needs_an_index(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        note = seeded_vault / "Decisions" / "2026-09-14-side-project-build.md"

        exit_code = main(["brief", str(note), "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "whatif index" in captured.err

    def test_brief_command_reports_a_missing_note(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        main(["index", "--config", str(config_path)])
        capsys.readouterr()

        exit_code = main(["brief", "no-such-note", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Note not found" in captured.err

    def test_brief_command_finds_a_note_by_id(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        main(["index", "--config", str(config_path)])
        capsys.readouterr()

        exit_code = main(["brief", "2026-10-01-certification", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.err
        assert (seeded_vault / "Briefs" / "2026-10-01-certification-vs-project.md").exists()

    def test_canvas_command_end_to_end(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        note = seeded_vault / "Decisions" / "2026-09-14-side-project-build.md"

        exit_code = main(["canvas", str(note), "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.err
        canvas_path = seeded_vault / "Canvases" / "2026-09-14-side-project-build.canvas"
        assert canvas_path.exists()

        payload = json.loads(canvas_path.read_text(encoding="utf-8"))
        assert "nodes" in payload
        assert "edges" in payload

        # One node per option, plus the root, plus assumptions and risks.
        option_texts = [
            n["text"] for n in payload["nodes"] if n.get("text", "").startswith("### ")
        ]
        assert len(option_texts) == 2
        assert any("Build in Python" in t for t in option_texts)
        assert any("Build in TypeScript" in t for t in option_texts)

        # The root is a file node pointing at the decision.
        roots = [n for n in payload["nodes"] if n["type"] == "file"]
        assert len(roots) == 1
        assert roots[0]["file"] == "Decisions/2026-09-14-side-project-build.md"

        # Every edge points at a node that exists.
        node_ids = {n["id"] for n in payload["nodes"]}
        for edge in payload["edges"]:
            assert edge["fromNode"] in node_ids
            assert edge["toNode"] in node_ids

    def test_canvas_command_needs_no_index(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Canvas export is pure code, so it must work without an index."""
        config_path = write_config(seeded_vault, fake_ollama)
        note = seeded_vault / "Decisions" / "2026-09-14-side-project-build.md"

        exit_code = main(["canvas", str(note), "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 0, captured.err
        assert (seeded_vault / "Canvases" / "2026-09-14-side-project-build.canvas").exists()

    def test_canvas_command_reports_a_missing_note(
        self, fake_ollama: str, seeded_vault: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = write_config(seeded_vault, fake_ollama)
        exit_code = main(["canvas", "no-such-note", "--config", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Note not found" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
