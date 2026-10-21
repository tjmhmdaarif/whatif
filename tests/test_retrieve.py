"""Tests for retrieve.py - cosine similarity and retrieval over the index."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from whatif.notes import parse_all_decisions
from whatif.config import create_default_config
from whatif.retrieve import (
    FailedAssumption,
    Retriever,
    SimilarDecision,
    build_summary,
    cosine_similarity,
    cosine_similarity_matrix,
    first_sentence,
    normalize,
)
from whatif.store import DecisionStore, index_decisions


def vector_from_text(text: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-embedding derived from the text's words.

    Words that appear in both texts pull the vectors together, which lets the
    tests control similarity without needing a real embedding model.
    """
    words = [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]
    raw = [0.0] * dim
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = digest[0] % dim
        raw[index] += digest[1] / 255.0
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0.0:
        return [1.0] + [0.0] * (dim - 1)
    return [x / norm for x in raw]


class WordEmbedder:
    """Embedder that maps texts sharing words to nearby vectors."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return vector_from_text(text, self.dim)


class ConstantEmbedder:
    """Embedder that always returns the same vector."""

    def __init__(self, dim: int = 8) -> None:
        self.vector = [1.0] + [0.0] * (dim - 1)
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.vector)


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
def indexed(seeded_vault: Path) -> tuple[DecisionStore, Retriever]:
    """A populated store plus a retriever that uses the fake embedder."""
    config = create_default_config(seeded_vault)
    index_decisions(config, embedder=WordEmbedder())
    store = DecisionStore(config.vault.db_path)
    retriever = Retriever(store, embedder=WordEmbedder())
    yield store, retriever
    store.close()


class TestNormalize:
    def test_unit_length(self) -> None:
        result = normalize(np.array([3.0, 4.0], dtype=np.float32))
        assert math.isclose(float(np.linalg.norm(result)), 1.0, rel_tol=1e-5)

    def test_zero_vector_unchanged(self) -> None:
        result = normalize(np.array([0.0, 0.0], dtype=np.float32))
        assert np.allclose(result, np.array([0.0, 0.0], dtype=np.float32))


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-5)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_scale_invariant(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = a * 10.0
        assert math.isclose(cosine_similarity(a, b), 1.0, rel_tol=1e-5)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_known_value(self) -> None:
        a = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        expected = 1.0 / math.sqrt(2.0)
        assert math.isclose(cosine_similarity(a, b), expected, rel_tol=1e-5)


class TestCosineSimilarityMatrix:
    def test_matches_individual_comparisons(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32
        )
        result = cosine_similarity_matrix(query, matrix)
        assert result.shape == (3,)
        for i in range(3):
            expected = cosine_similarity(query, matrix[i])
            assert math.isclose(float(result[i]), expected, rel_tol=1e-5)

    def test_empty_matrix(self) -> None:
        result = cosine_similarity_matrix(np.array([1.0, 0.0]), np.zeros((0, 2)))
        assert result.shape == (0,)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            cosine_similarity_matrix(np.array([1.0, 0.0]), np.ones((3, 5)))

    def test_zero_query_gives_zeros(self) -> None:
        result = cosine_similarity_matrix(np.zeros(3), np.ones((2, 3)))
        assert np.allclose(result, np.zeros(2))


class TestSummaryHelpers:
    def test_first_sentence(self) -> None:
        text = "The bootcamp cost a lot of money. The second part follows on."
        assert first_sentence(text) == "The bootcamp cost a lot of money."

    def test_first_sentence_skips_short_fragments(self) -> None:
        text = "Hi. This is a real sentence that should be returned."
        assert first_sentence(text) == "This is a real sentence that should be returned."

    def test_first_sentence_truncates(self) -> None:
        long_text = "word " * 100
        result = first_sentence(long_text, max_length=50)
        assert len(result) <= 50
        assert result.endswith("…")

    def test_first_sentence_empty(self) -> None:
        assert first_sentence("   ") == ""

    def test_build_summary_contains_key_facts(self) -> None:
        summary = build_summary(
            "2026-01-15-x", "study", "2026-01-15", "A real context sentence here.",
            ["A", "B"], "A",
        )
        assert "study" in summary
        assert "2026-01-15" in summary
        assert "options: A vs B" in summary
        assert "chose: A" in summary


class TestSimilarDecisions:
    def test_returns_top_k(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        results = retriever.similar_by_text("choosing between a course and a laptop", k=3)
        assert len(results) == 3

    def test_k_zero_returns_nothing(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        assert retriever.similar_by_text("anything", k=0) == []

    def test_excludes_requested_ids(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        results = retriever.similar_by_text(
            "course", k=5, exclude_ids=["2026-01-15-course-choice"]
        )
        assert all(r.note_id != "2026-01-15-course-choice" for r in results)

    def test_results_are_sorted_by_similarity(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        _, retriever = indexed
        results = retriever.similar_by_text("study course laptop", k=5)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_similarities_within_bounds(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        _, retriever = indexed
        for result in retriever.similar_by_text("anything at all", k=10):
            assert -1.0001 <= result.similarity <= 1.0001

    def test_threshold_filters_results(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        permissive = retriever.similar_by_text("x", k=10, threshold=-1.0)
        strict = retriever.similar_by_text("x", k=10, threshold=1.01)
        assert len(permissive) == 10
        assert len(strict) == 0

    def test_domain_filter(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        results = retriever.similar_by_text("anything", k=10, domain="study")
        assert results
        assert all(r.domain == "study" for r in results)

    def test_unknown_domain_filter_returns_nothing(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        _, retriever = indexed
        assert retriever.similar_by_text("anything", k=10, domain="nope") == []

    def test_result_carries_note_metadata(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        result = retriever.similar_by_text("course", k=5)[0]
        assert result.note_id
        assert result.domain
        assert result.status
        assert result.summary
        assert isinstance(result.options, list)
        assert isinstance(result.failed_assumptions, list)

    def test_similar_to_decision_excludes_self(
        self, indexed: tuple[DecisionStore, Retriever], seeded_vault: Path
    ) -> None:
        store, retriever = indexed
        decisions, _ = parse_all_decisions(create_default_config(seeded_vault))
        target = next(d for d in decisions if d.id == "2026-01-15-course-choice")

        results = retriever.similar_to_decision(target, k=5)
        assert all(r.note_id != target.id for r in results)

    def test_similar_to_decision_can_include_self(
        self, indexed: tuple[DecisionStore, Retriever], seeded_vault: Path
    ) -> None:
        store, retriever = indexed
        decisions, _ = parse_all_decisions(create_default_config(seeded_vault))
        target = next(d for d in decisions if d.id == "2026-01-15-course-choice")

        results = retriever.similar_to_decision(target, k=5, exclude_self=False)
        assert any(r.note_id == target.id for r in results)


class TestFailedAssumptions:
    def test_finds_failed_assumptions(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        results = retriever.failed_assumptions_by_text("time and schedule", k=10)
        assert results
        assert all(isinstance(r, FailedAssumption) for r in results)

    def test_excludes_requested_notes(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        _, retriever = indexed
        results = retriever.failed_assumptions_by_text(
            "anything", k=20, exclude_ids=["2026-03-10-club-role"]
        )
        assert all(r.source_note_id != "2026-03-10-club-role" for r in results)

    def test_only_surfaces_actual_failures(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        """Every surfaced assumption must be a real failed assumption in the index."""
        store, retriever = indexed
        results = retriever.failed_assumptions_by_text("anything", k=50)
        assert results
        for result in results:
            note = store.get_note(result.source_note_id)
            assert note is not None
            import json

            outcome = json.loads(note["outcome"])
            assert result.text in outcome["failed_assumptions"]

    def test_similarity_threshold(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        strict = retriever.failed_assumptions_by_text("anything", k=20, threshold=1.01)
        assert strict == []

    def test_sorted_by_similarity(self, indexed: tuple[DecisionStore, Retriever]) -> None:
        _, retriever = indexed
        results = retriever.failed_assumptions_by_text("time money team", k=20)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_deduplicates_same_assumption(
        self, indexed: tuple[DecisionStore, Retriever]
    ) -> None:
        _, retriever = indexed
        results = retriever.failed_assumptions_by_text("anything", k=50)
        keys = [(r.source_note_id, r.text) for r in results]
        assert len(keys) == len(set(keys))

    def test_for_decision_excludes_self(
        self, indexed: tuple[DecisionStore, Retriever], seeded_vault: Path
    ) -> None:
        store, retriever = indexed
        decisions, _ = parse_all_decisions(create_default_config(seeded_vault))
        target = next(d for d in decisions if d.id == "2026-03-10-club-role")

        results = retriever.failed_assumptions_for_decision(target, k=20)
        assert all(r.source_note_id != target.id for r in results)

    def test_empty_index_returns_nothing(self, temp_dir: Path) -> None:
        store = DecisionStore(temp_dir / ".whatif" / "index.sqlite")
        try:
            retriever = Retriever(store, embedder=WordEmbedder())
            assert retriever.failed_assumptions_by_text("anything") == []
            assert retriever.similar_by_text("anything") == []
        finally:
            store.close()


class TestRetrieve:
    def test_returns_both(self, indexed: tuple[DecisionStore, Retriever], seeded_vault: Path) -> None:
        store, retriever = indexed
        decisions, _ = parse_all_decisions(create_default_config(seeded_vault))
        target = next(d for d in decisions if d.id == "2026-09-14-side-project-build")

        similar, failed = retriever.retrieve(target, k=5, assumption_k=5)
        assert len(similar) == 5
        assert isinstance(failed, list)

    def test_empty_index(self, temp_dir: Path, seeded_vault: Path) -> None:
        store = DecisionStore(temp_dir / ".whatif" / "index.sqlite")
        try:
            decisions, _ = parse_all_decisions(create_default_config(seeded_vault))
            retriever = Retriever(store, embedder=WordEmbedder())
            similar, failed = retriever.retrieve(decisions[0], k=5)
            assert similar == []
            assert failed == []
        finally:
            store.close()


class TestNoEmbedder:
    def test_text_query_without_embedder_raises(self, temp_dir: Path) -> None:
        store = DecisionStore(temp_dir / ".whatif" / "index.sqlite")
        try:
            retriever = Retriever(store)
            with pytest.raises(RuntimeError, match="embedder"):
                retriever.similar_by_text("anything")
        finally:
            store.close()

    def test_vector_query_without_embedder_works(self, temp_dir: Path) -> None:
        store = DecisionStore(temp_dir / ".whatif" / "index.sqlite")
        try:
            retriever = Retriever(store)
            assert retriever.similar_by_vector([1.0, 0.0, 0.0], k=3) == []
        finally:
            store.close()


class TestAssumptionIndexing:
    def test_failed_assumptions_are_embedded(
        self, seeded_vault: Path
    ) -> None:
        config = create_default_config(seeded_vault)
        index_decisions(config, embedder=WordEmbedder())

        store = DecisionStore(config.vault.db_path)
        try:
            assert store.count_assumption_embeddings() > 0
            items = store.get_assumption_embeddings()
            # The seed data contains 11 failed assumptions across 6 decided notes.
            assert len(items) == 11
            assert len({item["note_id"] for item in items}) == 6
        finally:
            store.close()

    def test_open_notes_have_no_assumption_embeddings(
        self, seeded_vault: Path
    ) -> None:
        config = create_default_config(seeded_vault)
        index_decisions(config, embedder=WordEmbedder())

        store = DecisionStore(config.vault.db_path)
        try:
            items = store.get_assumption_embeddings()
            note_ids = {item["note_id"] for item in items}
            assert "2026-07-22-thesis-topic" not in note_ids
            assert "2026-09-14-side-project-build" not in note_ids
        finally:
            store.close()

    def test_reindex_replaces_assumption_embeddings(
        self, seeded_vault: Path
    ) -> None:
        config = create_default_config(seeded_vault)
        index_decisions(config, embedder=WordEmbedder())

        store = DecisionStore(config.vault.db_path)
        try:
            before = store.count_assumption_embeddings()
            store.delete_assumption_embeddings("2026-03-10-club-role")
            assert store.count_assumption_embeddings() == before - 2
        finally:
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
