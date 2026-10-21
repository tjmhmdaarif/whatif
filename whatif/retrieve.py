"""Retrieval of similar past decisions and previously failed assumptions.

Similarity is cosine similarity over vectors stored in SQLite, computed with numpy.
Two kinds of retrieval happen here:

1. **Similar decisions** - the top-k past notes whose situation resembles the new one.
2. **Failed assumptions** - assumptions that were stated in past notes and turned out
   to be wrong, matched against the new decision by the meaning of the assumption
   text rather than the topic of the note it came from.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from whatif.notes import Decision
from whatif.store import DecisionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return the unit-length version of a vector.

    A zero vector is returned unchanged rather than producing NaNs.
    """
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity of two vectors, in [-1, 1]."""
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector shapes differ: {vec_a.shape} vs {vec_b.shape}. "
            "The note was probably embedded with a different model."
        )
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities between one query vector and a matrix of vectors.

    Args:
        query: Query vector of shape ``(d,)``.
        matrix: Matrix of shape ``(n, d)``.

    Returns:
        Array of shape ``(n,)`` with the similarity of each row to the query.
    """
    query_vec = np.asarray(query, dtype=np.float32)
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.size == 0:
        return np.zeros(0, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"Query has {query_vec.shape[0]} dimensions but the index has {mat.shape[1]}. "
            "The note was probably embedded with a different model."
        )
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return np.zeros(mat.shape[0], dtype=np.float32)
    row_norms = np.linalg.norm(mat, axis=1)
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    dots = mat @ query_vec
    return (dots / (safe_norms * query_norm)).astype(np.float32)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SimilarDecision:
    """A past decision judged similar to the current one."""

    note_id: str
    similarity: float
    domain: str = ""
    status: str = ""
    created: str = ""
    summary: str = ""
    options: list[str] = field(default_factory=list)
    chosen: str = ""
    failed_assumptions: list[str] = field(default_factory=list)
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "similarity": round(self.similarity, 4),
            "domain": self.domain,
            "status": self.status,
            "created": self.created,
            "summary": self.summary,
            "options": list(self.options),
            "chosen": self.chosen,
            "failed_assumptions": list(self.failed_assumptions),
        }


@dataclass
class FailedAssumption:
    """An assumption from a past note that turned out to be wrong."""

    text: str
    source_note_id: str
    similarity: float
    source_domain: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_note_id": self.source_note_id,
            "similarity": round(self.similarity, 4),
            "source_domain": self.source_domain,
        }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str, max_length: int = 180) -> str:
    """Return the first meaningful sentence of a block of text, truncated."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        sentence = sentence.strip()
        if len(sentence) > 20:
            if len(sentence) > max_length:
                return sentence[: max_length - 1].rstrip() + "…"
            return sentence
    if len(cleaned) > max_length:
        return cleaned[: max_length - 1].rstrip() + "…"
    return cleaned


def build_summary(
    note_id: str,
    domain: str,
    created: str,
    body: str,
    options: Sequence[str] = (),
    chosen: str = "",
) -> str:
    """Build a deterministic one-line relevance summary for a note.

    The LLM can improve this in the brief, but retrieval needs something usable
    on its own. The options are left out on purpose: they make the line too long
    for a table cell, and they are shown next to the note elsewhere.
    """
    parts: list[str] = []
    if domain:
        parts.append(domain)
    if created:
        parts.append(created)
    headline = first_sentence(body)
    if headline:
        parts.append(headline)
    # ASCII separator: this text is printed to terminals as well as written into
    # Markdown, and em dashes render as mojibake on Windows consoles.
    return " - ".join(parts)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Retrieves similar decisions and failed assumptions from the index.

    Args:
        store: Open :class:`~whatif.store.DecisionStore` to read from.
        embedder: Callable turning text into a vector. Required for text queries.
    """

    def __init__(self, store: DecisionStore, embedder: Any | None = None) -> None:
        self.store = store
        self.embedder = embedder

    # -- helpers -----------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured embedder."""
        if self.embedder is None:
            raise RuntimeError(
                "This retriever needs an embedder for text queries. "
                "Build one with whatif.embeddings.build_embedder(config)."
            )
        return self.embedder(text)

    @staticmethod
    def _exclude(
        exclude_ids: Iterable[str] | None,
    ) -> set[str]:
        """Build the set of note ids to leave out of a result."""
        return {note_id for note_id in (exclude_ids or []) if note_id}

    def _note_rows(self, exclude: set[str]) -> list[dict[str, Any]]:
        """Return note rows with their vectors, minus the excluded ids."""
        rows = []
        for note_id, vector in self.store.get_all_embeddings().items():
            if note_id in exclude:
                continue
            row = self.store.get_note(note_id)
            if row is None:
                continue
            row = dict(row)
            row["vector"] = vector
            rows.append(row)
        return rows

    @staticmethod
    def _row_to_similar(row: dict[str, Any], similarity: float) -> SimilarDecision:
        """Convert a note row plus a similarity into a :class:`SimilarDecision`."""
        try:
            options = json.loads(row.get("options", "[]"))
        except json.JSONDecodeError:
            options = []
        try:
            outcome = json.loads(row.get("outcome", "{}"))
        except json.JSONDecodeError:
            outcome = {}
        failed = outcome.get("failed_assumptions") or []
        body = row.get("body", "") or ""

        return SimilarDecision(
            note_id=row["id"],
            similarity=similarity,
            domain=row.get("domain", ""),
            status=row.get("status", ""),
            created=row.get("created", ""),
            summary=build_summary(
                row["id"],
                row.get("domain", ""),
                row.get("created", ""),
                body,
                options,
                row.get("chosen", ""),
            ),
            options=list(options),
            chosen=row.get("chosen", ""),
            failed_assumptions=list(failed),
            file_path=row.get("file_path", ""),
        )

    # -- similar decisions -------------------------------------------------

    def similar_by_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        k: int = 5,
        exclude_ids: Iterable[str] | None = None,
        threshold: float = 0.0,
        domain: str | None = None,
    ) -> list[SimilarDecision]:
        """Return the k most similar past decisions for a query vector.

        Args:
            query_vector: The query embedding.
            k: Maximum number of results.
            exclude_ids: Note ids to leave out (usually the note being briefed).
            threshold: Minimum cosine similarity for a result to be kept.
            domain: Restrict results to one domain.

        Returns:
            Results ordered by descending similarity.
        """
        if k <= 0:
            return []
        exclude = self._exclude(exclude_ids)
        rows = self._note_rows(exclude)
        if not rows:
            return []

        if domain is not None:
            rows = [r for r in rows if r.get("domain") == domain]
            if not rows:
                return []

        matrix = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in rows])
        similarities = cosine_similarity_matrix(np.asarray(query_vector, dtype=np.float32), matrix)

        pairs = list(zip(similarities.tolist(), rows))
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for similarity, row in pairs[:k]:
            if similarity < threshold:
                continue
            results.append(self._row_to_similar(row, similarity))
        return results

    def similar_by_text(
        self,
        query_text: str,
        k: int = 5,
        exclude_ids: Iterable[str] | None = None,
        threshold: float = 0.0,
        domain: str | None = None,
    ) -> list[SimilarDecision]:
        """Return the k most similar past decisions for a text query."""
        vector = self._embed(query_text)
        return self.similar_by_vector(
            vector, k=k, exclude_ids=exclude_ids, threshold=threshold, domain=domain
        )

    def similar_to_decision(
        self,
        decision: Decision,
        k: int = 5,
        threshold: float = 0.0,
        exclude_self: bool = True,
        domain: str | None = None,
    ) -> list[SimilarDecision]:
        """Return past decisions similar to a given decision note.

        The note itself is excluded by default so a brief never cites itself.
        """
        from whatif.store import decision_text

        exclude = [decision.id] if exclude_self else []
        return self.similar_by_text(
            decision_text(decision),
            k=k,
            exclude_ids=exclude,
            threshold=threshold,
            domain=domain,
        )

    # -- failed assumptions ------------------------------------------------

    def failed_assumptions_by_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        k: int = 10,
        exclude_ids: Iterable[str] | None = None,
        threshold: float = 0.0,
    ) -> list[FailedAssumption]:
        """Return previously failed assumptions similar to a query vector.

        Args:
            query_vector: The query embedding.
            k: Maximum number of results.
            exclude_ids: Note ids whose assumptions must not be surfaced.
            threshold: Minimum cosine similarity for a result to be kept.

        Returns:
            Results ordered by descending similarity.
        """
        if k <= 0:
            return []
        exclude = self._exclude(exclude_ids)

        candidates = [
            item
            for item in self.store.get_assumption_embeddings()
            if item["note_id"] not in exclude
        ]
        if not candidates:
            return []

        matrix = np.stack([np.asarray(c["vector"], dtype=np.float32) for c in candidates])
        similarities = cosine_similarity_matrix(np.asarray(query_vector, dtype=np.float32), matrix)

        pairs = list(zip(similarities.tolist(), candidates))
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        results: list[FailedAssumption] = []
        seen: set[tuple[str, str]] = set()
        for similarity, candidate in pairs:
            if similarity < threshold:
                break
            key = (candidate["note_id"], candidate["assumption"])
            if key in seen:
                continue
            seen.add(key)
            row = self.store.get_note(candidate["note_id"])
            results.append(
                FailedAssumption(
                    text=candidate["assumption"],
                    source_note_id=candidate["note_id"],
                    similarity=similarity,
                    source_domain=row.get("domain", "") if row else "",
                )
            )
            if len(results) >= k:
                break
        return results

    def failed_assumptions_by_text(
        self,
        query_text: str,
        k: int = 10,
        exclude_ids: Iterable[str] | None = None,
        threshold: float = 0.0,
    ) -> list[FailedAssumption]:
        """Return previously failed assumptions similar to a text query."""
        vector = self._embed(query_text)
        return self.failed_assumptions_by_vector(
            vector, k=k, exclude_ids=exclude_ids, threshold=threshold
        )

    def failed_assumptions_for_decision(
        self,
        decision: Decision,
        k: int = 10,
        threshold: float = 0.0,
        exclude_self: bool = True,
    ) -> list[FailedAssumption]:
        """Return previously failed assumptions relevant to a decision note.

        The query is the note's own stated assumptions, because that is the text a
        past failure needs to be close to in order to be a warning.
        """
        from whatif.store import decision_text

        exclude = [decision.id] if exclude_self else []
        query = decision_text(decision)
        if decision.assumptions:
            query = "\n".join(decision.assumptions)
        return self.failed_assumptions_by_text(
            query, k=k, exclude_ids=exclude, threshold=threshold
        )

    # -- combined ----------------------------------------------------------

    def retrieve(
        self,
        decision: Decision,
        k: int = 5,
        note_threshold: float = 0.0,
        assumption_threshold: float = 0.0,
        assumption_k: int = 10,
    ) -> tuple[list[SimilarDecision], list[FailedAssumption]]:
        """Retrieve both similar decisions and relevant failed assumptions.

        Args:
            decision: The note being briefed.
            k: Number of similar decisions to return.
            note_threshold: Minimum similarity for a similar decision.
            assumption_threshold: Minimum similarity for a failed assumption.
            assumption_k: Number of failed assumptions to return.

        Returns:
            Tuple of (similar decisions, failed assumptions).
        """
        similar = self.similar_to_decision(
            decision, k=k, threshold=note_threshold, exclude_self=True
        )
        failed = self.failed_assumptions_for_decision(
            decision, k=assumption_k, threshold=assumption_threshold, exclude_self=True
        )
        return similar, failed
