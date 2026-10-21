"""Tests for store.py - SQLite persistence, content hashing, and skip logic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from whatif.config import Config, create_default_config
from whatif.notes import Decision, parse_all_decisions, parse_decision_file
from whatif.store import (
    DecisionStore,
    IndexStats,
    blob_to_vector,
    decision_text,
    index_decisions,
    vector_to_blob,
)


class FakeEmbedder:
    """Deterministic embedder that needs no Ollama server.

    Maps text to a vector built from a hash of its tokens, so similar texts get
    similar vectors and unrelated texts do not.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]


class CountingEmbedder:
    """Wraps another embedder and counts how many times it is called."""

    def __init__(self, inner: Callable[[str], list[float]]) -> None:
        self.inner = inner
        self.call_count = 0

    def __call__(self, text: str) -> list[float]:
        self.call_count += 1
        return self.inner(text)


# The seed data contains 11 failed assumptions across 6 decided notes.
FAILED_ASSUMPTION_COUNT = 11


@pytest.fixture
def store(temp_dir: Path) -> DecisionStore:
    store = DecisionStore(temp_dir / ".whatif" / "index.sqlite")
    yield store
    store.close()


@pytest.fixture
def vault_with_seeds(temp_dir: Path) -> Path:
    """A vault populated with the 10 seed decision notes."""
    vault = temp_dir / "vault"
    (vault / "Decisions").mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "decisions"
    for note in source.glob("*.md"):
        (vault / "Decisions" / note.name).write_text(
            note.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return vault


class TestVectorSerialization:
    def test_roundtrip(self) -> None:
        vector = [0.1, 0.2, 0.3, 0.4]
        restored = blob_to_vector(vector_to_blob(vector))
        assert np.allclose(restored, np.array(vector, dtype=np.float32))

    def test_blob_is_float32(self) -> None:
        blob = vector_to_blob([0.5, 0.5])
        assert len(blob) == 8  # 2 floats x 4 bytes

    def test_empty_vector(self) -> None:
        assert len(vector_to_blob([])) == 0


class TestDecisionStore:
    def test_creates_database_and_schema(self, temp_dir: Path) -> None:
        db_path = temp_dir / ".whatif" / "index.sqlite"
        store = DecisionStore(db_path)
        try:
            assert db_path.exists()
            assert store.count() == 0
            assert store.count_embeddings() == 0
        finally:
            store.close()

    def test_upsert_note_roundtrip(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[0.1, 0.2, 0.3])

        row = store.get_note(decision.id)
        assert row is not None
        assert row["id"] == decision.id
        assert row["domain"] == decision.domain
        assert row["status"] == decision.status
        assert json.loads(row["options"]) == decision.options
        assert json.loads(row["criteria"]) == decision.criteria
        assert json.loads(row["assumptions"]) == decision.assumptions
        assert json.loads(row["outcome"])["result"] == decision.outcome["result"]

    def test_upsert_replaces_existing(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[0.1, 0.2])
        store.upsert_note(decision, embedding=[0.5, 0.6])

        assert store.count() == 1
        vector = store.get_embedding(decision.id)
        assert vector is not None
        assert np.allclose(vector, np.array([0.5, 0.6], dtype=np.float32))

    def test_embedding_retrieval(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[1.0, 2.0, 3.0])

        vector = store.get_embedding(decision.id)
        assert vector is not None
        assert np.allclose(vector, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_records_embedding_model(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[1.0], model="nomic-embed-text")

        row = store.conn.execute(
            "SELECT model FROM embeddings WHERE note_id = ?", (decision.id,)
        ).fetchone()
        assert row["model"] == "nomic-embed-text"

    def test_missing_embedding_returns_none(self, store: DecisionStore) -> None:
        assert store.get_embedding("does-not-exist") is None

    def test_get_all_embeddings(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        for i, decision in enumerate(decisions[:3]):
            store.upsert_note(decision, embedding=[float(i), 1.0])

        embeddings = store.get_all_embeddings()
        assert len(embeddings) == 3
        assert set(embeddings.keys()) == {d.id for d in decisions[:3]}

    def test_delete_note_removes_embedding(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[0.1])

        assert store.delete_note(decision.id) is True
        assert store.get_note(decision.id) is None
        assert store.get_embedding(decision.id) is None
        assert store.delete_note(decision.id) is False

    def test_context_manager(self, temp_dir: Path) -> None:
        with DecisionStore(temp_dir / ".whatif" / "index.sqlite") as store:
            assert store.count() == 0


class TestContentHashSkipLogic:
    def test_new_note_needs_embedding(self, store: DecisionStore) -> None:
        assert store.needs_embedding("new-note", "hash-abc") is True

    def test_unchanged_note_is_skipped(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[0.1, 0.2])

        assert store.needs_embedding(decision.id, decision.content_hash) is False

    def test_changed_note_needs_embedding(self, store: DecisionStore, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision, embedding=[0.1, 0.2])

        assert store.needs_embedding(decision.id, "different-hash") is True

    def test_same_hash_without_embedding_needs_embedding(
        self, store: DecisionStore, vault_with_seeds: Path
    ) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = decisions[0]
        store.upsert_note(decision)  # note stored, no embedding

        assert store.needs_embedding(decision.id, decision.content_hash) is True

    def test_hash_matches_sha256_of_file(self, vault_with_seeds: Path) -> None:
        note_path = vault_with_seeds / "Decisions" / "2026-01-15-course-choice.md"
        raw = note_path.read_bytes()
        expected = hashlib.sha256(raw).hexdigest()

        result = parse_decision_file(note_path)
        assert result.decision is not None
        assert result.decision.content_hash == expected

    def test_hash_changes_when_file_changes(self, vault_with_seeds: Path) -> None:
        note_path = vault_with_seeds / "Decisions" / "2026-01-15-course-choice.md"
        before = parse_decision_file(note_path).decision.content_hash

        note_path.write_text(
            note_path.read_text(encoding="utf-8") + "\nExtra line.\n", encoding="utf-8"
        )
        after = parse_decision_file(note_path).decision.content_hash

        assert before != after


class TestIndexDecisions:
    def test_indexes_all_seed_notes(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        embedder = FakeEmbedder()

        stats = index_decisions(config, embedder=embedder)

        assert stats.indexed == 10
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.errors == []
        # 10 note embeddings + 11 failed-assumption embeddings.
        assert len(embedder.calls) == 10 + FAILED_ASSUMPTION_COUNT

    def test_second_run_skips_unchanged(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        embedder = FakeEmbedder()

        first = index_decisions(config, embedder=embedder)
        second = index_decisions(config, embedder=embedder)

        assert first.indexed == 10
        assert second.indexed == 0
        assert second.skipped == 10
        # No re-embedding on the second run.
        assert len(embedder.calls) == 10 + FAILED_ASSUMPTION_COUNT

    def test_changed_note_is_reindexed(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        embedder = FakeEmbedder()

        index_decisions(config, embedder=embedder)

        note_path = vault_with_seeds / "Decisions" / "2026-01-15-course-choice.md"
        note_path.write_text(
            note_path.read_text(encoding="utf-8") + "\nI changed my mind.\n", encoding="utf-8"
        )

        stats = index_decisions(config, embedder=embedder)
        assert stats.indexed == 1
        assert stats.skipped == 9
        # The changed note is re-embedded, along with its 2 failed assumptions.
        assert len(embedder.calls) == 10 + FAILED_ASSUMPTION_COUNT + 3

    def test_writes_to_database(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        index_decisions(config, embedder=FakeEmbedder())

        assert config.vault.db_path.exists()
        store = DecisionStore(config.vault.db_path)
        try:
            assert store.count() == 10
            assert store.count_embeddings() == 10
        finally:
            store.close()

    def test_records_configured_model_name(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        index_decisions(config, embedder=FakeEmbedder())

        store = DecisionStore(config.vault.db_path)
        try:
            rows = store.conn.execute("SELECT DISTINCT model FROM embeddings").fetchall()
            assert [r["model"] for r in rows] == ["nomic-embed-text"]
        finally:
            store.close()

    def test_bad_note_is_counted_as_failed(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)
        bad = vault_with_seeds / "Decisions" / "bad.md"
        bad.write_text("---\nid: [broken\n---\n", encoding="utf-8")

        stats = index_decisions(config, embedder=FakeEmbedder())

        assert stats.indexed == 10
        assert stats.failed == 1
        assert len(stats.errors) == 1
        assert "bad.md" in stats.errors[0]

    def test_embedder_failure_does_not_stop_run(self, vault_with_seeds: Path) -> None:
        config = create_default_config(vault_with_seeds)

        def exploding_embedder(text: str) -> list[float]:
            if "bootcamp" in text:
                raise RuntimeError("model exploded")
            return [0.1, 0.2, 0.3]

        stats = index_decisions(config, embedder=exploding_embedder)

        assert stats.failed == 1
        assert stats.indexed == 9
        assert "model exploded" in stats.errors[0]

    def test_empty_vault(self, temp_dir: Path) -> None:
        config = create_default_config(temp_dir)
        stats = index_decisions(config, embedder=FakeEmbedder())
        assert stats.indexed == 0
        assert stats.skipped == 0
        assert stats.failed == 0

    def test_stats_summary(self) -> None:
        stats = IndexStats(indexed=3, skipped=2, failed=1)
        assert "indexed=3" in stats.summary()
        assert stats.as_dict()["indexed"] == 3


class TestDecisionText:
    def test_includes_context_and_options(self, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        text = decision_text(decisions[0])

        assert "Domain:" in text
        assert "Options:" in text
        assert len(text) > 100

    def test_excludes_outcome(self, vault_with_seeds: Path) -> None:
        """The note vector must describe the situation, not its resolution."""
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        for decision in decisions:
            text = decision_text(decision)
            # The Outcome prose must never reach the embedding.
            assert decision.outcome_body.strip() not in text
            # The word "failed" must not appear: only stated assumptions are embedded.
            assert "failed" not in text.lower()

    def test_excludes_outcome_but_keeps_stated_assumptions(self, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = next(d for d in decisions if d.has_failed_assumptions())
        text = decision_text(decision)

        # A failed assumption is still a *stated* assumption, so it is embedded.
        assert decision.outcome["failed_assumptions"][0] in text
        # But the outcome prose is not.
        assert decision.outcome_body.strip() not in text

    def test_includes_assumptions_and_risks(self, vault_with_seeds: Path) -> None:
        decisions, _ = parse_all_decisions(create_default_config(vault_with_seeds))
        decision = next(d for d in decisions if d.assumptions)
        text = decision_text(decision)
        assert "Stated assumptions:" in text
        assert decision.assumptions[0] in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
