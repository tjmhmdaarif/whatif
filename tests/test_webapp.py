"""Tests for webapp.py - the local web interface."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from whatif.config import create_default_config
from whatif.webapp import WEB_ROOT, WebApp, create_server


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


@pytest.fixture
def server(seeded_vault: Path):
    """A running web server on an ephemeral port."""
    config = create_default_config(seeded_vault)
    httpd = create_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture
def offline_server(seeded_vault: Path):
    """A server whose Ollama host is a closed port, so it never answers."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    config = create_default_config(seeded_vault)
    config.ollama.host = f"http://127.0.0.1:{dead_port}"
    config.ollama.timeout = 3
    httpd = create_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def get(url: str) -> tuple[int, str, str]:
    """Fetch a URL and return (status, content type, body)."""
    with urllib.request.urlopen(url, timeout=15) as response:
        return (
            response.status,
            response.headers.get("Content-Type", ""),
            response.read().decode("utf-8"),
        )


def get_json(url: str) -> tuple[int, dict[str, Any]]:
    """Fetch a URL and decode the JSON body."""
    status, _, body = get(url)
    return status, json.loads(body)


def post_json(url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """POST a JSON body and decode the JSON response."""
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class TestStaticFiles:
    def test_index_is_served(self, server: str) -> None:
        status, content_type, body = get(f"{server}/")
        assert status == 200
        assert "text/html" in content_type
        assert "WhatIf" in body

    def test_stylesheet_is_served(self, server: str) -> None:
        status, content_type, body = get(f"{server}/static/styles.css")
        assert status == 200
        assert "text/css" in content_type
        assert "--accent" in body

    def test_script_is_served(self, server: str) -> None:
        status, content_type, body = get(f"{server}/static/app.js")
        assert status == 200
        assert "javascript" in content_type
        assert "renderBrief" in body

    def test_missing_static_file_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get(f"{server}/static/nope.css")
        assert excinfo.value.code == 404

    def test_path_traversal_is_blocked(self, server: str) -> None:
        """A crafted path must not escape the web folder."""
        for attempt in ("/static/../config.py", "/static/..%2fconfig.py"):
            try:
                status, _, body = get(f"{server}{attempt}")
            except urllib.error.HTTPError as exc:
                assert exc.code in (403, 404)
                continue
            assert "Config" not in body, "path traversal leaked a source file"

    def test_web_folder_exists(self) -> None:
        assert (WEB_ROOT / "index.html").exists()
        assert (WEB_ROOT / "styles.css").exists()
        assert (WEB_ROOT / "app.js").exists()


class TestHealth:
    def test_health_reports_the_vault(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/health")
        assert status == 200
        assert payload["note_count"] == 10
        assert payload["db_exists"] is False
        assert payload["ok"] is (payload["ollama_reachable"] and payload["db_exists"])

    def test_health_reports_ollama_down(self, offline_server: str) -> None:
        _, payload = get_json(f"{offline_server}/api/health")
        assert payload["ollama_reachable"] is False
        assert payload["ollama_error"]
        assert payload["ok"] is False

    def test_health_names_the_models(self, server: str) -> None:
        _, payload = get_json(f"{server}/api/health")
        assert payload["embed_model"]
        assert payload["llm_model"]
        assert payload["host"]


class TestNotes:
    def test_lists_the_notes(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/notes")
        assert status == 200
        assert len(payload["notes"]) == 10
        assert payload["errors"] == []

    def test_note_summary_shape(self, server: str) -> None:
        _, payload = get_json(f"{server}/api/notes")
        note = payload["notes"][0]
        assert set(note) >= {
            "id", "domain", "status", "created", "options", "headline",
            "failure_count", "has_failures",
        }

    def test_counts_failures(self, server: str) -> None:
        _, payload = get_json(f"{server}/api/notes")
        with_failures = [n for n in payload["notes"] if n["has_failures"]]
        assert len(with_failures) == 6

    def test_single_note(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/note/2026-01-15-course-choice")
        assert status == 200
        assert payload["id"] == "2026-01-15-course-choice"
        assert len(payload["options"]) == 2
        assert payload["outcome"]["result"] == "worse"
        assert len(payload["outcome"]["failed_assumptions"]) == 2

    def test_missing_note_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get(f"{server}/api/note/no-such-note")
        assert excinfo.value.code == 404


class TestScoring:
    def test_returns_the_computed_result(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/scoring/2026-01-15-course-choice")
        assert status == 200
        assert payload["result"]["leader"] == "In-person bootcamp"
        assert "COMPUTED" in payload["table"]

    def test_table_lists_both_options(self, server: str) -> None:
        _, payload = get_json(f"{server}/api/scoring/2026-01-15-course-choice")
        assert "In-person bootcamp" in payload["table"]
        assert "Self-paced online course" in payload["table"]


class TestCanvas:
    def test_builds_a_canvas(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/canvas/2026-01-15-course-choice")
        assert status == 200
        canvas = payload["canvas"]
        assert len(canvas["nodes"]) > 1
        assert len(canvas["edges"]) > 1
        ids = {n["id"] for n in canvas["nodes"]}
        for edge in canvas["edges"]:
            assert edge["fromNode"] in ids
            assert edge["toNode"] in ids

    def test_writes_the_canvas_file(self, server: str) -> None:
        _, payload = get_json(f"{server}/api/canvas/2026-01-15-course-choice")
        path = Path(payload["path"])
        assert path.exists()
        assert path.suffix == ".canvas"
        assert json.loads(path.read_text(encoding="utf-8"))["nodes"]


class TestBrief:
    def test_missing_brief_is_reported(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/brief/2026-01-15-course-choice")
        assert status == 200
        assert payload["exists"] is False

    def test_generation_without_ollama_is_a_503(self, offline_server: str) -> None:
        status, payload = post_json(
            f"{offline_server}/api/brief/2026-01-15-course-choice"
        )
        assert status == 503
        assert "Ollama" in payload["error"]


class TestEvaluation:
    def test_missing_results_are_reported(self, server: str) -> None:
        status, payload = get_json(f"{server}/api/eval")
        assert status == 200
        assert payload["exists"] is False

    def test_running_without_ollama_is_a_503(self, offline_server: str) -> None:
        status, payload = post_json(f"{offline_server}/api/eval")
        assert status == 503
        assert "Ollama" in payload["error"]


class TestIndex:
    def test_index_without_ollama_is_a_503(self, offline_server: str) -> None:
        status, payload = post_json(f"{offline_server}/api/index")
        assert status == 503
        assert "Ollama" in payload["error"]


class TestRouting:
    def test_unknown_route_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get(f"{server}/api/nope")
        assert excinfo.value.code == 404

    def test_unknown_post_route_is_404(self, server: str) -> None:
        status, payload = post_json(f"{server}/api/nope")
        assert status == 404
        assert "error" in payload


class TestWebApp:
    def test_summarises_a_note(self, seeded_vault: Path) -> None:
        app = WebApp(create_default_config(seeded_vault))
        decisions, _ = app.decisions()
        summary = app.note_summary(decisions[0])
        assert summary["id"] == decisions[0].id
        assert summary["failure_count"] >= 0

    def test_finds_a_note_by_id(self, seeded_vault: Path) -> None:
        app = WebApp(create_default_config(seeded_vault))
        assert app.decision("2026-01-15-course-choice") is not None
        assert app.decision("nope") is None

    def test_binds_to_loopback_only(self, seeded_vault: Path) -> None:
        """The server must not be reachable from the network."""
        config = create_default_config(seeded_vault)
        httpd = create_server(config, host="127.0.0.1", port=0)
        try:
            assert httpd.server_address[0] == "127.0.0.1"
        finally:
            httpd.server_close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
