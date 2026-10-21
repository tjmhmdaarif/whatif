"""A local web interface for WhatIf.

This is an addition to the original scope, which listed a web UI as a non-goal. It
is deliberately kept in one module plus a folder of static files, and it calls the
same functions the CLI calls, so it cannot drift from the command line behaviour.

The server binds to 127.0.0.1 only. It makes no outbound connections of its own;
the only network traffic it triggers is the calls to the configured Ollama server.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import traceback
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from whatif.config import Config, load_config
from whatif.embeddings import OllamaClient, OllamaError
from whatif.notes import Decision, parse_all_decisions, parse_decision_file
from whatif.retrieve import Retriever
from whatif.scoring import compute_scores, format_comparison_table, leading_option
from whatif.store import DecisionStore, index_decisions

logger = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).parent / "web"

# A lock so two browser tabs cannot start two index runs at the same time.
_index_lock = threading.Lock()


class WebApp:
    """Holds the state the HTTP handlers need.

    Args:
        config: Active configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()

    # -- helpers ----------------------------------------------------------

    def decisions(self) -> tuple[list[Decision], list[str]]:
        """Parse every decision note in the vault."""
        return parse_all_decisions(self.config)

    def decision(self, note_id: str) -> Decision | None:
        """Return one decision by id, or None."""
        decisions, _ = self.decisions()
        for decision in decisions:
            if decision.id == note_id:
                return decision
        return None

    def store(self) -> DecisionStore:
        """Open the index database."""
        return DecisionStore(self.config.vault.db_path)

    def ollama_client(self, model: str) -> OllamaClient:
        """Build an Ollama client for a given model."""
        return OllamaClient(
            host=self.config.ollama.host,
            model=model,
            timeout=self.config.ollama.timeout,
        )

    def retriever(self, store: DecisionStore) -> Retriever:
        """Build a retriever bound to the embedding model."""
        client = self.ollama_client(self.config.models.embed_model)
        return Retriever(store, embedder=client.embed)

    # -- payloads ---------------------------------------------------------

    def note_summary(self, decision: Decision) -> dict[str, Any]:
        """Summarise a note for the list view."""
        failures = decision.outcome.get("failed_assumptions") or []
        headline = ""
        for line in decision.context.split("\n"):
            line = line.strip()
            if len(line) > 24:
                headline = line
                break
        return {
            "id": decision.id,
            "domain": decision.domain,
            "status": decision.status,
            "created": decision.created.isoformat(),
            "options": list(decision.options),
            "chosen": decision.chosen,
            "headline": headline,
            "failure_count": len(failures),
            "has_failures": bool(failures),
            "criteria": [c.get("name") for c in decision.criteria],
        }

    def scoring_payload(self, decision: Decision) -> dict[str, Any]:
        """Return the computed scoring for a note."""
        scoring = compute_scores(decision)
        return {
            "result": scoring.to_dict(),
            "table": format_comparison_table(scoring),
            "leading": leading_option(decision, scoring),
            "warnings": scoring.warnings,
        }

    def retrieval_payload(self, decision: Decision, k: int = 5) -> dict[str, Any]:
        """Return the similar notes and past failures for a note.

        The failures are returned with their query similarity and a flag saying
        whether they clear the threshold a brief would apply, so the interface can
        show the ranking instead of an empty list.
        """
        threshold = self.config.retrieval.failed_assumption_threshold
        with self.store() as store:
            retriever = self.retriever(store)
            similar = retriever.similar_to_decision(
                decision,
                k=k,
                threshold=self.config.retrieval.similarity_threshold,
                exclude_self=True,
            )
            failed = retriever.failed_assumptions_for_decision(
                decision,
                k=k * 3,
                threshold=0.0,
                exclude_self=True,
            )

        failed_payload = []
        for item in failed:
            payload = item.to_dict()
            payload["above_threshold"] = item.similarity >= threshold
            failed_payload.append(payload)

        return {
            "similar": [note.to_dict() for note in similar],
            "failed": failed_payload,
            "threshold": threshold,
            "k": k,
        }

    def brief_payload(self, decision: Decision) -> dict[str, Any]:
        """Return an already generated brief, if there is one."""
        path = self.config.vault.briefs_path / f"{decision.id}.md"
        if not path.exists():
            return {"exists": False, "path": str(path)}
        return {
            "exists": True,
            "path": str(path),
            "content": path.read_text(encoding="utf-8"),
        }

    def canvas_payload(self, decision: Decision) -> dict[str, Any]:
        """Return the canvas for a note, generating it if needed."""
        from whatif.canvas import build_canvas

        with self.store() as store:
            canvas = build_canvas(decision, self.config)
        path = self.config.vault.canvases_path / f"{decision.id}.canvas"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canvas.to_json(), encoding="utf-8")
        return {"canvas": canvas.to_dict(), "path": str(path)}

    def health_payload(self) -> dict[str, Any]:
        """Report whether the tool is ready to work."""
        client = self.ollama_client(self.config.models.embed_model)
        reachable = False
        models: list[str] = []
        error = ""
        try:
            models = client.list_models()
            reachable = True
        except OllamaError as exc:
            error = str(exc)

        decisions, parse_errors = self.decisions()
        indexed = 0
        db_exists = self.config.vault.db_path.exists()
        if db_exists:
            with self.store() as store:
                indexed = store.count()

        return {
            "ok": reachable and db_exists,
            "ollama_reachable": reachable,
            "ollama_error": error,
            "host": self.config.ollama.host,
            "models": models,
            "llm_model": self.config.models.llm_model,
            "embed_model": self.config.models.embed_model,
            "vault": str(self.config.vault.path),
            "note_count": len(decisions),
            "parse_errors": len(parse_errors),
            "indexed": indexed,
            "db_exists": db_exists,
        }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


class WhatIfHandler(BaseHTTPRequestHandler):
    """Handles the API and serves the static files."""

    server_version = "WhatIf"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> WebApp:
        """The application state."""
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Log requests at debug level instead of writing to stderr."""
        logger.debug("%s - %s", self.address_string(), format % args)

    # -- plumbing ---------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        """Send a response with an explicit content length."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        """Send a JSON response."""
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        """Send a JSON error response."""
        self._send_json({"error": message, "status": status}, status=status)

    def _read_body(self) -> dict[str, Any]:
        """Read and parse a JSON request body."""
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        """Handle a GET request."""
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return
            if path == "/api/health":
                self._send_json(self.app.health_payload())
                return
            if path == "/api/notes":
                decisions, errors = self.app.decisions()
                self._send_json(
                    {
                        "notes": [self.app.note_summary(d) for d in decisions],
                        "errors": errors,
                    }
                )
                return
            if path.startswith("/api/note/"):
                self._handle_note(path[len("/api/note/") :])
                return
            if path.startswith("/api/scoring/"):
                self._handle_scoring(path[len("/api/scoring/") :])
                return
            if path.startswith("/api/retrieval/"):
                self._handle_retrieval(path[len("/api/retrieval/") :])
                return
            if path.startswith("/api/brief/"):
                self._handle_brief_get(path[len("/api/brief/") :])
                return
            if path.startswith("/api/canvas/"):
                self._handle_canvas(path[len("/api/canvas/") :])
                return
            if path == "/api/eval":
                self._handle_eval_get()
                return
            self._send_error_json(404, f"No route for {path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("GET %s failed", path)
            self._send_error_json(500, f"{exc}\n{traceback.format_exc(limit=3)}")

    def do_POST(self) -> None:  # noqa: N802
        """Handle a POST request."""
        path = unquote(urlparse(self.path).path)
        try:
            if path.startswith("/api/brief/"):
                self._handle_brief_post(path[len("/api/brief/") :])
                return
            if path.startswith("/api/canvas/"):
                self._handle_canvas(path[len("/api/canvas/") :])
                return
            if path == "/api/eval":
                self._handle_eval_post()
                return
            if path == "/api/index":
                self._handle_index()
                return
            self._send_error_json(404, f"No route for {path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", path)
            self._send_error_json(500, f"{exc}\n{traceback.format_exc(limit=3)}")

    # -- static -----------------------------------------------------------

    def _serve_static(self, relative: str) -> None:
        """Serve a file from the web folder."""
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._send_error_json(403, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self._send_error_json(404, f"Not found: {relative}")
            return
        content_type, _ = mimetypes.guess_type(str(target))
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    # -- handlers ---------------------------------------------------------

    def _require_decision(self, note_id: str) -> Decision | None:
        """Look up a decision, sending an error response if it is missing."""
        decision = self.app.decision(note_id)
        if decision is None:
            self._send_error_json(404, f"No note with id '{note_id}'")
        return decision

    def _handle_note(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return
        payload: dict[str, Any] = {
            "id": decision.id,
            "domain": decision.domain,
            "status": decision.status,
            "created": decision.created.isoformat(),
            "options": list(decision.options),
            "chosen": decision.chosen,
            "context": decision.context,
            "options_body": decision.options_body,
            "decision_body": decision.decision_body,
            "outcome_body": decision.outcome_body,
            "criteria": decision.criteria,
            "scores": decision.scores,
            "assumptions": list(decision.assumptions),
            "risks": list(decision.risks),
            "outcome": {
                "reviewed_on": str(decision.outcome.get("reviewed_on") or ""),
                "result": decision.outcome.get("result") or "",
                "failed_assumptions": list(
                    decision.outcome.get("failed_assumptions") or []
                ),
            },
            "tags": list(decision.tags),
            "summary": self.app.note_summary(decision),
        }
        self._send_json(payload)

    def _handle_scoring(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return
        self._send_json(self.app.scoring_payload(decision))

    def _handle_retrieval(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return
        body = self._read_body()
        k = int(body.get("k", 5)) if body else 5
        self._send_json(self.app.retrieval_payload(decision, k=k))

    def _handle_brief_get(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return
        self._send_json(self.app.brief_payload(decision))

    def _handle_brief_post(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return

        from whatif.brief import BriefGenerator

        body = self._read_body()
        k = int(body.get("k", 5))

        try:
            with self.app.store() as store:
                client = self.app.ollama_client(self.app.config.models.llm_model)
                retriever = self.app.retriever(store)
                generator = BriefGenerator(
                    self.app.config, client, retriever, note_bodies=store.get_note_bodies()
                )
                brief = generator.generate(decision, k=k)
        except OllamaError as exc:
            self._send_error_json(503, f"Ollama unavailable: {exc}")
            return

        path = self.app.config.vault.briefs_path / f"{decision.id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(brief.content, encoding="utf-8")

        self._send_json(
            {
                "id": decision.id,
                "path": str(path),
                "content": brief.content,
                "sections": brief.sections,
                "relevance": brief.relevance,
                "similar": [note.to_dict() for note in brief.similar],
                "failed": [item.to_dict() for item in brief.failed_assumptions],
                "scoring": brief.scoring.to_dict() if brief.scoring else {},
                "leading": brief.leading,
                "validation": brief.validation.as_dict() if brief.validation else {},
                "retries": brief.retries,
            }
        )

    def _handle_canvas(self, note_id: str) -> None:
        decision = self._require_decision(note_id)
        if decision is None:
            return
        self._send_json(self.app.canvas_payload(decision))

    def _handle_eval_get(self) -> None:
        path = self.app.config.vault.whatif_path / "eval-results.json"
        if not path.exists():
            self._send_json({"exists": False, "path": str(path)})
            return
        self._send_json({"exists": True, "path": str(path),
                         "results": json.loads(path.read_text(encoding="utf-8"))})

    def _handle_eval_post(self) -> None:
        from whatif.eval import format_report, run_eval, write_report

        body = self._read_body()
        threshold = body.get("threshold")
        try:
            client = self.app.ollama_client(self.app.config.models.embed_model)
            with self.app.store() as store:
                report = run_eval(
                    self.app.config,
                    store,
                    embedder=client.embed,
                    threshold=float(threshold) if threshold is not None else None,
                )
        except OllamaError as exc:
            self._send_error_json(503, f"Ollama unavailable: {exc}")
            return

        out_path = self.app.config.vault.whatif_path / "eval-results.json"
        write_report(report, out_path)
        self._send_json(
            {
                "path": str(out_path),
                "text": format_report(report),
                "results": report.as_dict(),
            }
        )

    def _handle_index(self) -> None:
        if not _index_lock.acquire(blocking=False):
            self._send_error_json(409, "An index run is already in progress.")
            return
        try:
            stats = index_decisions(self.app.config)
        except OllamaError as exc:
            self._send_error_json(503, f"Ollama unavailable: {exc}")
            return
        finally:
            _index_lock.release()
        self._send_json(stats.as_dict())


def create_server(config: Config, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create the HTTP server without starting it."""
    server = ThreadingHTTPServer((host, port), partial(WhatIfHandler))
    server.app = WebApp(config)  # type: ignore[attr-defined]
    server.daemon_threads = True
    return server


def serve(
    config_path: Path | str | None = None,
    vault_path: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> int:
    """Run the web interface until interrupted."""
    config = load_config(config_path, vault_path)
    try:
        server = create_server(config, host=host, port=port)
    except OSError as exc:
        print(f"Cannot start the server on {host}:{port}: {exc}", flush=True)
        return 1

    url = f"http://{host}:{server.server_port}"
    print(f"WhatIf is running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0
