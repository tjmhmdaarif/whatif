"""SQLite persistence for WhatIf.

Stores parsed decision notes and their embedding vectors in the vault's
``.whatif/`` directory. Uses a content hash so that unchanged notes are not
re-embedded on every index run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from whatif.config import Config
from whatif.embeddings import OllamaError
from whatif.notes import Decision, parse_all_decisions

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id           TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    type         TEXT NOT NULL,
    status       TEXT NOT NULL,
    created      TEXT NOT NULL,
    domain       TEXT NOT NULL,
    options      TEXT NOT NULL,
    chosen       TEXT NOT NULL,
    criteria     TEXT NOT NULL,
    scores       TEXT NOT NULL,
    assumptions  TEXT NOT NULL,
    risks        TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    tags         TEXT NOT NULL,
    body         TEXT NOT NULL,
    indexed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    note_id    TEXT PRIMARY KEY,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status);
CREATE INDEX IF NOT EXISTS idx_notes_domain ON notes(domain);
"""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class IndexStats:
    """Counts produced by an index run."""

    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "indexed": self.indexed,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": list(self.errors),
        }

    def summary(self) -> str:
        """One-line human summary."""
        return f"indexed={self.indexed} skipped={self.skipped} failed={self.failed}"


def vector_to_blob(vector: Iterable[float]) -> bytes:
    """Serialize a vector to little-endian float32 bytes."""
    array = np.asarray(list(vector), dtype=np.float32)
    return array.tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize float32 bytes back into a numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class DecisionStore:
    """SQLite-backed store for decision notes and their embeddings.

    Args:
        db_path: Path to the SQLite database file. Parent dirs are created.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and record the schema version."""
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> DecisionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- notes -------------------------------------------------------------

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        """Return a note row as a dict, or None if absent."""
        row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None

    def get_all_notes(self) -> list[dict[str, Any]]:
        """Return every note row as a list of dicts."""
        rows = self.conn.execute("SELECT * FROM notes ORDER BY created, id").fetchall()
        return [dict(row) for row in rows]

    def note_ids(self) -> list[str]:
        """Return the ids of all stored notes."""
        rows = self.conn.execute("SELECT id FROM notes ORDER BY id").fetchall()
        return [row["id"] for row in rows]

    def get_content_hash(self, note_id: str) -> str | None:
        """Return the stored content hash for a note, or None if not indexed."""
        row = self.conn.execute(
            "SELECT content_hash FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return row["content_hash"] if row else None

    def needs_embedding(self, note_id: str, content_hash: str) -> bool:
        """Return True if the note is new or its content changed since indexing.

        A note is considered unchanged only when both the content hash matches and an
        embedding row already exists.
        """
        stored_hash = self.get_content_hash(note_id)
        if stored_hash != content_hash:
            return True
        row = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE note_id = ?", (note_id,)
        ).fetchone()
        return row is None

    def upsert_note(
        self, decision: Decision, embedding: list[float] | None = None, model: str = ""
    ) -> None:
        """Insert or update a decision note and optionally its embedding."""
        if decision.file_path is None:
            raise ValueError(f"Decision {decision.id} has no file_path")

        self.conn.execute(
            """
            INSERT INTO notes (
                id, file_path, content_hash, type, status, created, domain,
                options, chosen, criteria, scores, assumptions, risks, outcome,
                tags, body, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_path = excluded.file_path,
                content_hash = excluded.content_hash,
                type = excluded.type,
                status = excluded.status,
                created = excluded.created,
                domain = excluded.domain,
                options = excluded.options,
                chosen = excluded.chosen,
                criteria = excluded.criteria,
                scores = excluded.scores,
                assumptions = excluded.assumptions,
                risks = excluded.risks,
                outcome = excluded.outcome,
                tags = excluded.tags,
                body = excluded.body,
                indexed_at = excluded.indexed_at
            """,
            (
                decision.id,
                str(decision.file_path),
                decision.content_hash,
                decision.type,
                decision.status,
                decision.created.isoformat(),
                decision.domain,
                json.dumps(decision.options, ensure_ascii=False),
                decision.chosen,
                json.dumps(decision.criteria, ensure_ascii=False),
                json.dumps(decision.scores, ensure_ascii=False),
                json.dumps(decision.assumptions, ensure_ascii=False),
                json.dumps(decision.risks, ensure_ascii=False),
                json.dumps(decision.outcome, ensure_ascii=False, default=str),
                json.dumps(decision.tags, ensure_ascii=False),
                decision.context,
                utc_now(),
            ),
        )
        self.conn.commit()

        if embedding is not None:
            self.set_embedding(decision.id, embedding, model=model)

    def delete_note(self, note_id: str) -> bool:
        """Delete a note and its embedding. Returns True if a row was removed."""
        cursor = self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # -- embeddings --------------------------------------------------------

    def set_embedding(self, note_id: str, vector: list[float], model: str = "") -> None:
        """Store the embedding vector for a note."""
        self.conn.execute(
            """
            INSERT INTO embeddings (note_id, dim, vector, model, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET
                dim = excluded.dim,
                vector = excluded.vector,
                model = excluded.model,
                created_at = excluded.created_at
            """,
            (note_id, len(list(vector)), vector_to_blob(vector), model, utc_now()),
        )
        self.conn.commit()

    def get_embedding(self, note_id: str) -> np.ndarray | None:
        """Return the embedding vector for a note, or None if absent."""
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE note_id = ?", (note_id,)
        ).fetchone()
        if row is None:
            return None
        return blob_to_vector(row["vector"])

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return every stored embedding keyed by note id."""
        rows = self.conn.execute("SELECT note_id, vector FROM embeddings").fetchall()
        return {row["note_id"]: blob_to_vector(row["vector"]) for row in rows}

    def embedding_ids(self) -> list[str]:
        """Return the ids of notes that have embeddings."""
        rows = self.conn.execute("SELECT note_id FROM embeddings ORDER BY note_id").fetchall()
        return [row["note_id"] for row in rows]

    def count(self) -> int:
        """Return the number of stored notes."""
        row = self.conn.execute("SELECT COUNT(*) AS n FROM notes").fetchone()
        return int(row["n"])

    def count_embeddings(self) -> int:
        """Return the number of stored embeddings."""
        row = self.conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        return int(row["n"])


def decision_text(decision: Decision) -> str:
    """Build the text used to represent a decision in embedding space.

    The outcome and the failed assumptions are deliberately excluded so that the note
    vector describes the *situation* rather than its resolution. Failed assumptions are
    embedded separately when they are surfaced.
    """
    parts = [
        f"Domain: {decision.domain}",
        f"Options: {', '.join(decision.options)}",
    ]
    if decision.context:
        parts.append(decision.context)
    if decision.options_body:
        parts.append(decision.options_body)
    if decision.assumptions:
        parts.append("Stated assumptions: " + "; ".join(decision.assumptions))
    if decision.risks:
        parts.append("Stated risks: " + "; ".join(decision.risks))
    return "\n".join(parts).strip()


def index_decisions(
    config: Config,
    embedder: Any | None = None,
    store: DecisionStore | None = None,
) -> IndexStats:
    """Index every decision note in the vault.

    Notes whose content hash is unchanged are skipped and not re-embedded.

    Args:
        config: Active configuration.
        embedder: Callable turning text into a vector. Defaults to the configured
            Ollama embedder.
        store: Existing store to write into. A new one is opened and closed if omitted.

    Returns:
        Counts of indexed, skipped, and failed notes plus per-file error messages.
    """
    stats = IndexStats()
    own_store = store is None
    if store is None:
        store = DecisionStore(config.vault.db_path)

    try:
        decisions, parse_errors = parse_all_decisions(config)
        for error in parse_errors:
            stats.failed += 1
            stats.errors.append(error)
            logger.error("Skipping invalid note: %s", error)

        if not decisions:
            logger.info("No decision notes found in %s", config.vault.decisions_path)
            return stats

        # Preflight: fail fast with a clear message if Ollama is unreachable, rather
        # than repeating the same connection error once per note.
        embed_model = config.models.embed_model
        if embedder is None:
            from whatif.embeddings import build_client, build_embedder

            client = build_client(config)
            if not client.is_reachable():
                raise OllamaError(
                    f"Cannot reach Ollama at {config.ollama.host}. "
                    "Start the server with 'ollama serve' and re-run."
                )
            client.require_model(embed_model)
            embedder = build_embedder(config)

        for decision in decisions:
            if not store.needs_embedding(decision.id, decision.content_hash):
                stats.skipped += 1
                logger.debug("Skipping unchanged note %s", decision.id)
                continue

            try:
                text = decision_text(decision)
                vector = embedder(text)
                if not vector:
                    raise ValueError("embedder returned an empty vector")
                store.upsert_note(decision, embedding=vector, model=embed_model)
                stats.indexed += 1
                logger.info("Indexed %s", decision.id)
            except Exception as exc:  # noqa: BLE001 - one bad note must not stop the run
                stats.failed += 1
                message = f"{decision.file_path}: {exc}"
                stats.errors.append(message)
                logger.error("Failed to index %s: %s", decision.id, exc)

        return stats
    finally:
        if own_store:
            store.close()
