"""Leave-one-out evaluation of the failed-assumption retrieval.

The question this answers is narrow and honest: **when a decision is made, does
WhatIf surface a past assumption failure that was actually relevant to it?**

For every decided note that has at least one failed assumption:

1. the note is excluded from the index, so it cannot see its own outcome;
2. the note is queried exactly as a brief would query it;
3. the failed assumptions surfaced from the *other* notes are compared against the
   hidden note's true failed assumptions, using the same cosine similarity the
   rest of the system uses;
4. a hit is counted when a surfaced assumption is similar enough to one of the
   true ones.

The hit rate is reported as-is. A low number is a finding, not a failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from whatif.config import Config
from whatif.notes import Decision, parse_all_decisions
from whatif.retrieve import FailedAssumption, Retriever, cosine_similarity
from whatif.store import DecisionStore, decision_text

logger = logging.getLogger(__name__)


# Thresholds the report also reports a rate for, so the headline number can be
# read in context rather than taken on faith.
REPORT_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass
class SurfacedAssumption:
    """One failed assumption surfaced from another note."""

    text: str
    source_note_id: str
    query_similarity: float
    best_match: float = 0.0
    best_match_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_note_id": self.source_note_id,
            "query_similarity": round(self.query_similarity, 4),
            "best_match": round(self.best_match, 4),
            "best_match_text": self.best_match_text,
        }


@dataclass
class EvalCase:
    """The result for one held-out note."""

    note_id: str
    domain: str = ""
    true_failed_assumptions: list[str] = field(default_factory=list)
    surfaced: list[SurfacedAssumption] = field(default_factory=list)
    best_match: float = 0.0
    best_match_true: str = ""
    best_match_surfaced: str = ""
    best_match_source: str = ""
    hit: bool = False
    surfaced_count: int = 0
    retrieved_count: int = 0
    query_similarity_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "domain": self.domain,
            "hit": self.hit,
            "best_match": round(self.best_match, 4),
            "best_match_true": self.best_match_true,
            "best_match_surfaced": self.best_match_surfaced,
            "best_match_source": self.best_match_source,
            "surfaced_count": self.surfaced_count,
            "retrieved_count": self.retrieved_count,
            "true_failed_assumptions": list(self.true_failed_assumptions),
            "surfaced": [item.to_dict() for item in self.surfaced],
        }


@dataclass
class EvalReport:
    """The result of a whole evaluation run."""

    cases: list[EvalCase] = field(default_factory=list)
    threshold: float = 0.6
    notes_total: int = 0
    notes_skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    model_warning: str = ""
    embed_model: str = ""
    llm_model: str = ""

    @property
    def hits(self) -> int:
        """Number of notes where a relevant failure was surfaced."""
        return sum(1 for case in self.cases if case.hit)

    @property
    def total(self) -> int:
        """Number of notes that could be evaluated."""
        return len(self.cases)

    @property
    def hit_rate(self) -> float:
        """Hits over total, as a fraction."""
        if not self.cases:
            return 0.0
        return self.hits / len(self.cases)

    def rates_by_threshold(self) -> list[dict[str, Any]]:
        """Recompute the hit rate at several thresholds.

        The surfaced set is filtered by the query similarity threshold and the hit
        is decided by the same threshold, matching what a brief would do.
        """
        rows: list[dict[str, Any]] = []
        for threshold in REPORT_THRESHOLDS:
            hits = 0
            for case in self.cases:
                surfaced = [
                    item
                    for item in case.surfaced
                    if item.query_similarity >= threshold
                ]
                if any(item.best_match >= threshold for item in surfaced):
                    hits += 1
            rows.append(
                {
                    "threshold": threshold,
                    "hits": hits,
                    "total": len(self.cases),
                    "rate": round(hits / len(self.cases), 4) if self.cases else 0.0,
                }
            )
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "threshold": self.threshold,
            "hits": self.hits,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "notes_total": self.notes_total,
            "notes_skipped": self.notes_skipped,
            "skipped_reasons": dict(self.skipped_reasons),
            "model_warning": self.model_warning,
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "rates_by_threshold": self.rates_by_threshold(),
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string with a trailing newline."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False) + "\n"


def query_text_for(decision: Decision) -> str:
    """Build the query a brief would use for a note.

    The outcome is never included, so the evaluation cannot see the answer.
    """
    if decision.assumptions:
        return "\n".join(decision.assumptions)
    return decision_text(decision)


def _embed_all(texts: Sequence[str], embedder: Any) -> list[np.ndarray]:
    """Embed a list of texts, returning one array per text."""
    vectors: list[np.ndarray] = []
    for text in texts:
        vector = embedder(text)
        vectors.append(np.asarray(vector, dtype=np.float32))
    return vectors


def evaluate_note(
    decision: Decision,
    retriever: Retriever,
    embedder: Any,
    threshold: float,
    assumption_k: int = 50,
    stored_vectors: dict[tuple[str, str], np.ndarray] | None = None,
) -> EvalCase:
    """Evaluate one held-out note.

    Args:
        decision: The note being held out.
        retriever: Retriever over the full index.
        embedder: Callable turning text into a vector.
        threshold: Similarity a surfaced assumption must reach to count as a hit.
        assumption_k: How many surfaced assumptions to consider.
        stored_vectors: Vectors already in the index, keyed by (note id, assumption).
            Used so the surfaced assumptions do not have to be embedded again.

    Returns:
        The case result.
    """
    true_failures = list(decision.outcome.get("failed_assumptions") or [])
    case = EvalCase(
        note_id=decision.id,
        domain=decision.domain,
        true_failed_assumptions=true_failures,
        query_similarity_threshold=threshold,
    )

    if not true_failures:
        return case

    query = query_text_for(decision)

    # The note is excluded, so it cannot see its own failures.
    surfaced = retriever.failed_assumptions_by_text(
        query,
        k=assumption_k,
        exclude_ids=[decision.id],
        threshold=0.0,
    )

    # Embed the hidden truth so the surfaced assumptions can be compared to it.
    # The surfaced assumptions already have vectors in the index, so they are
    # reused rather than embedded a second time.
    true_vectors = _embed_all(true_failures, embedder)

    for item in surfaced:
        candidate = None
        if stored_vectors is not None:
            candidate = stored_vectors.get((item.source_note_id, item.text))
        if candidate is None:
            candidate = np.asarray(embedder(item.text), dtype=np.float32)

        best = 0.0
        best_text = ""
        for failure, vector in zip(true_failures, true_vectors):
            similarity = cosine_similarity(candidate, vector)
            if similarity > best:
                best = similarity
                best_text = failure
        item.best_match = best
        item.best_match_text = best_text
        if best > case.best_match:
            case.best_match = best
            case.best_match_true = best_text
            case.best_match_surfaced = item.text
            case.best_match_source = item.source_note_id

    case.surfaced = [
        SurfacedAssumption(
            text=item.text,
            source_note_id=item.source_note_id,
            query_similarity=item.similarity,
            best_match=item.best_match,
            best_match_text=item.best_match_text,
        )
        for item in surfaced
    ]

    # The surfaced set is filtered the way a brief filters it, then the hit is
    # decided by the same threshold.
    visible = [item for item in case.surfaced if item.query_similarity >= threshold]
    case.retrieved_count = len(case.surfaced)
    case.surfaced_count = len(visible)
    case.hit = any(item.best_match >= threshold for item in visible)

    return case


def run_eval(
    config: Config,
    store: DecisionStore,
    embedder: Any,
    threshold: float | None = None,
    assumption_k: int = 50,
) -> EvalReport:
    """Run the leave-one-out evaluation.

    Args:
        config: Active configuration.
        store: The index to evaluate against.
        embedder: Callable turning text into a vector.
        threshold: Similarity a hit must reach. Defaults to the configured value.
        assumption_k: How many surfaced assumptions to consider per note.

    Returns:
        The full report.
    """
    threshold = (
        threshold if threshold is not None else config.retrieval.failed_assumption_threshold
    )

    report = EvalReport(threshold=threshold)
    report.embed_model = config.models.embed_model
    report.llm_model = config.models.llm_model

    decisions, parse_errors = parse_all_decisions(config)
    for error in parse_errors:
        logger.error("Skipping invalid note: %s", error)
        report.notes_skipped += 1
        report.skipped_reasons["invalid_note"] = (
            report.skipped_reasons.get("invalid_note", 0) + 1
        )

    report.notes_total = len(decisions)
    retriever = Retriever(store, embedder=embedder)

    # Reuse the vectors already in the index instead of embedding the surfaced
    # assumptions again.
    stored_vectors = {
        (item["note_id"], item["assumption"]): item["vector"]
        for item in store.get_assumption_embeddings()
    }

    for decision in decisions:
        if decision.status != "decided":
            report.notes_skipped += 1
            report.skipped_reasons["not_decided"] = (
                report.skipped_reasons.get("not_decided", 0) + 1
            )
            continue
        if not decision.has_failed_assumptions():
            report.notes_skipped += 1
            report.skipped_reasons["no_failed_assumptions"] = (
                report.skipped_reasons.get("no_failed_assumptions", 0) + 1
            )
            continue

        try:
            case = evaluate_note(
                decision,
                retriever,
                embedder,
                threshold=threshold,
                assumption_k=assumption_k,
                stored_vectors=stored_vectors,
            )
        except Exception as exc:  # noqa: BLE001 - one bad note must not stop the run
            logger.error("Failed to evaluate %s: %s", decision.id, exc)
            report.notes_skipped += 1
            report.skipped_reasons["error"] = report.skipped_reasons.get("error", 0) + 1
            continue

        report.cases.append(case)

    return report


def format_report(report: EvalReport) -> str:
    """Render the report for a terminal."""
    lines: list[str] = []
    lines.append("Leave-one-out evaluation")
    lines.append("=" * 60)
    lines.append("")
    if report.embed_model:
        lines.append(f"Embedding model: {report.embed_model}")
        lines.append(
            "The hit rate is a property of the embedding model and the notes, "
            "not of the harness alone."
        )
        lines.append("")

    for case in report.cases:
        status = "HIT " if case.hit else "MISS"
        lines.append(f"{status}  {case.note_id}  (best match {case.best_match:.3f})")
        if case.best_match_true:
            lines.append(f"        true     : {case.best_match_true}")
            lines.append(
                f"        surfaced : {case.best_match_surfaced} "
                f"[from {case.best_match_source}]"
            )
        lines.append(
            f"        retrieved {case.retrieved_count} past failures, "
            f"{case.surfaced_count} above the {case.query_similarity_threshold:.2f} threshold"
        )
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Notes evaluated : {report.total}")
    lines.append(f"Notes skipped   : {report.notes_skipped}")
    for reason, count in sorted(report.skipped_reasons.items()):
        lines.append(f"  - {reason}: {count}")
    lines.append("")
    lines.append(f"Hit rate at threshold {report.threshold:.2f}: "
                 f"{report.hits}/{report.total} = {report.hit_rate:.0%}")
    lines.append("")

    lines.append("Hit rate at other thresholds:")
    for row in report.rates_by_threshold():
        lines.append(
            f"  {row['threshold']:.2f}  {row['hits']}/{row['total']} = {row['rate']:.0%}"
        )
    lines.append("")

    if report.model_warning:
        lines.append(f"Warning: {report.model_warning}")
        lines.append("")

    return "\n".join(lines)


def write_report(report: EvalReport, path: Path | str) -> Path:
    """Write the report as JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")
    logger.info("Wrote evaluation results to %s", target)
    return target


def check_model_consistency(store: DecisionStore, config: Config) -> str:
    """Warn if the index was built with a different embedding model.

    Comparing vectors from two different models produces numbers that look like
    similarities but mean nothing, so this is worth flagging loudly.
    """
    rows = store.conn.execute("SELECT DISTINCT model FROM embeddings").fetchall()
    models = {row["model"] for row in rows if row["model"]}
    if not models:
        return ""
    expected = config.models.embed_model
    if expected not in models:
        return (
            f"The index was built with embedding model(s) {sorted(models)} but the "
            f"config says {expected}. Re-run 'whatif index' to rebuild it, or the "
            f"similarity numbers in this report are not meaningful."
        )
    return ""


def summarize_cases(report: EvalReport) -> list[dict[str, Any]]:
    """Return one flat row per case, for tabular output."""
    rows: list[dict[str, Any]] = []
    for case in report.cases:
        rows.append(
            {
                "note_id": case.note_id,
                "hit": case.hit,
                "best_match": round(case.best_match, 4),
                "surfaced_count": case.surfaced_count,
                "matched_source": case.best_match_source,
            }
        )
    return rows


def iter_case_ids(report: EvalReport) -> Iterable[str]:
    """Yield the ids of every evaluated note."""
    for case in report.cases:
        yield case.note_id
