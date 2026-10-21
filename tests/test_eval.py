"""Tests for eval.py - the leave-one-out evaluation."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from whatif.config import create_default_config
from whatif.eval import (
    REPORT_THRESHOLDS,
    EvalCase,
    EvalReport,
    SurfacedAssumption,
    check_model_consistency,
    evaluate_note,
    format_report,
    iter_case_ids,
    query_text_for,
    run_eval,
    summarize_cases,
    write_report,
)
from whatif.notes import Decision, parse_all_decisions
from whatif.retrieve import Retriever
from whatif.store import DecisionStore, index_decisions


def vector_from_text(text: str, dim: int = 32) -> list[float]:
    """Deterministic pseudo-embedding that rewards shared words."""
    words = [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]
    raw = [0.0] * dim
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        raw[digest[0] % dim] += digest[1] / 255.0
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


def make_decision(
    note_id: str,
    status: str = "decided",
    failed: list[str] | None = None,
    assumptions: list[str] | None = None,
    context: str = "Some context about the decision.",
    file_path: Path | None = None,
) -> Decision:
    """Build a Decision for testing."""
    return Decision(
        id=note_id,
        type="decision",
        status=status,
        created=datetime.date(2026, 1, 15),
        domain="study",
        options=["A", "B"],
        chosen="A",
        criteria=[{"name": "cost", "weight": 1}],
        scores={"A": {"cost": 3}, "B": {"cost": 2}},
        assumptions=assumptions if assumptions is not None else ["I have enough time."],
        risks=[],
        outcome={
            "reviewed_on": datetime.date(2026, 6, 1) if status == "decided" else None,
            "result": "worse" if status == "decided" else "",
            "failed_assumptions": failed if failed is not None else [],
        },
        tags=[],
        context=context,
        file_path=file_path or Path(f"/tmp/notes/{note_id}.md"),
    )


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
def indexed(seeded_vault: Path) -> tuple[Config, DecisionStore]:
    """An indexed vault and its store."""
    config = create_default_config(seeded_vault)
    index_decisions(config, embedder=WordEmbedder())
    store = DecisionStore(config.vault.db_path)
    yield config, store
    store.close()


class TestQueryText:
    def test_uses_the_assumptions(self) -> None:
        decision = make_decision("2026-01-15-a", assumptions=["I have enough time."])
        assert query_text_for(decision) == "I have enough time."

    def test_falls_back_to_the_context(self) -> None:
        decision = make_decision("2026-01-15-a", assumptions=[])
        assert "context about the decision" in query_text_for(decision)

    def test_never_includes_the_outcome(self) -> None:
        decision = make_decision(
            "2026-01-15-a",
            assumptions=["I have enough time."],
            context="A context sentence.",
        )
        text = query_text_for(decision)
        assert "worse" not in text
        assert "reviewed_on" not in text


class TestEvaluateNote:
    def test_hits_when_a_related_failure_is_surfaced(self) -> None:
        store = DecisionStore(Path("/tmp/whatif-eval-test-1.sqlite"))
        try:
            embedder = WordEmbedder()
            # Two notes that failed on the same kind of assumption.
            other = make_decision(
                "2026-02-20-other",
                failed=["I would have enough free time to finish the project."],
                assumptions=["I would have enough free time to finish the project."],
            )
            target = make_decision(
                "2026-01-15-target",
                failed=["I would have enough free time to finish the project."],
                assumptions=["I would have enough free time to finish the project."],
            )
            store.upsert_note(other, embedding=embedder("other note"), model="test")
            store.set_assumption_embedding(
                other.id,
                "I would have enough free time to finish the project.",
                embedder("I would have enough free time to finish the project."),
            )

            retriever = Retriever(store, embedder=embedder)
            case = evaluate_note(target, retriever, embedder, threshold=0.6)

            assert case.hit is True
            assert case.best_match_source == other.id
        finally:
            store.close()

    def test_misses_when_nothing_related_is_surfaced(self) -> None:
        store = DecisionStore(Path("/tmp/whatif-eval-test-2.sqlite"))
        try:
            embedder = WordEmbedder()
            other = make_decision(
                "2026-02-20-other",
                failed=["The weather in the destination city stays mild."],
                assumptions=["The weather in the destination city stays mild."],
            )
            target = make_decision(
                "2026-01-15-target",
                failed=["I would have enough free time to finish the project."],
                assumptions=["I would have enough free time to finish the project."],
            )
            store.upsert_note(other, embedding=embedder("other note"), model="test")
            store.set_assumption_embedding(
                other.id,
                "The weather in the destination city stays mild.",
                embedder("The weather in the destination city stays mild."),
            )

            retriever = Retriever(store, embedder=embedder)
            case = evaluate_note(target, retriever, embedder, threshold=0.9)

            assert case.hit is False
        finally:
            store.close()

    def test_never_surfaces_the_notes_own_failure(self) -> None:
        store = DecisionStore(Path("/tmp/whatif-eval-test-3.sqlite"))
        try:
            embedder = WordEmbedder()
            target = make_decision(
                "2026-01-15-target",
                failed=["I would have enough free time."],
                assumptions=["I would have enough free time."],
            )
            store.upsert_note(target, embedding=embedder("target"), model="test")
            store.set_assumption_embedding(
                target.id,
                "I would have enough free time.",
                embedder("I would have enough free time."),
            )

            retriever = Retriever(store, embedder=embedder)
            case = evaluate_note(target, retriever, embedder, threshold=0.0)

            assert case.surfaced == []
            assert case.hit is False
        finally:
            store.close()

    def test_note_without_failures_is_a_no_op(self) -> None:
        store = DecisionStore(Path("/tmp/whatif-eval-test-4.sqlite"))
        try:
            embedder = WordEmbedder()
            target = make_decision("2026-01-15-target", failed=[])
            retriever = Retriever(store, embedder=embedder)
            case = evaluate_note(target, retriever, embedder, threshold=0.6)
            assert case.hit is False
            assert case.surfaced == []
        finally:
            store.close()


class TestRunEval:
    def test_evaluates_only_decided_notes_with_failures(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())

        assert report.total == 6, "6 seed notes are decided and have failed assumptions"
        assert report.hits <= report.total
        assert report.notes_total == 10
        assert report.notes_skipped == 4

    def test_skip_reasons_are_recorded(self, indexed: tuple[Config, DecisionStore]) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        assert report.skipped_reasons.get("not_decided") == 2
        assert report.skipped_reasons.get("no_failed_assumptions") == 2

    def test_hit_rate_is_between_zero_and_one(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        assert 0.0 <= report.hit_rate <= 1.0
        assert report.hits == sum(1 for c in report.cases if c.hit)

    def test_rate_at_the_configured_threshold(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder(), threshold=0.6)
        for case in report.cases:
            visible = [s for s in case.surfaced if s.query_similarity >= 0.6]
            expected = any(s.best_match >= 0.6 for s in visible)
            assert case.hit is expected

    def test_higher_threshold_never_increases_hits(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        low = run_eval(config, store, embedder=WordEmbedder(), threshold=0.1)
        high = run_eval(config, store, embedder=WordEmbedder(), threshold=0.95)
        assert high.hits <= low.hits

    def test_rates_by_threshold_are_monotonic(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        rows = report.rates_by_threshold()
        assert [row["threshold"] for row in rows] == list(REPORT_THRESHOLDS)
        rates = [row["rate"] for row in rows]
        assert rates == sorted(rates, reverse=True)

    def test_empty_vault(self, temp_dir: Path) -> None:
        config = create_default_config(temp_dir)
        store = DecisionStore(config.vault.db_path)
        try:
            report = run_eval(config, store, embedder=WordEmbedder())
            assert report.total == 0
            assert report.hit_rate == 0.0
        finally:
            store.close()

    def test_model_warning_when_the_index_uses_another_model(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        config.models.embed_model = "some-other-model"
        warning = check_model_consistency(store, config)
        assert "some-other-model" in warning
        assert "rebuild" in warning.lower() or "Re-run" in warning

    def test_no_model_warning_when_models_match(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        assert check_model_consistency(store, config) == ""


class TestReport:
    def test_as_dict(self) -> None:
        report = EvalReport(
            cases=[
                EvalCase(note_id="a", hit=True, best_match=0.8),
                EvalCase(note_id="b", hit=False, best_match=0.2),
            ],
            threshold=0.6,
        )
        payload = report.as_dict()
        assert payload["hits"] == 1
        assert payload["total"] == 2
        assert payload["hit_rate"] == 0.5
        assert len(payload["cases"]) == 2

    def test_to_json_is_valid(self) -> None:
        report = EvalReport(cases=[EvalCase(note_id="a", hit=True)])
        payload = json.loads(report.to_json())
        assert payload["hits"] == 1

    def test_format_report_mentions_the_rate(self) -> None:
        report = EvalReport(
            cases=[EvalCase(note_id="a", hit=True, best_match=0.9)], threshold=0.6
        )
        text = format_report(report)
        assert "Hit rate" in text
        assert "1/1" in text
        assert "HIT" in text

    def test_format_report_shows_misses(self) -> None:
        report = EvalReport(
            cases=[EvalCase(note_id="a", hit=False, best_match=0.1)], threshold=0.6
        )
        text = format_report(report)
        assert "MISS" in text

    def test_the_report_names_the_embedding_model(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        """A number without the model that produced it is not a result."""
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        assert report.embed_model == config.models.embed_model
        assert report.embed_model in format_report(report)
        assert report.embed_model in report.to_json()
    def test_write_report(self, temp_dir: Path) -> None:
        report = EvalReport(cases=[EvalCase(note_id="a", hit=True)], threshold=0.6)
        path = write_report(report, temp_dir / ".whatif" / "eval-results.json")
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["hits"] == 1

    def test_summarize_cases(self) -> None:
        report = EvalReport(cases=[EvalCase(note_id="a", hit=True, best_match=0.8)])
        rows = summarize_cases(report)
        assert rows[0]["note_id"] == "a"
        assert rows[0]["hit"] is True

    def test_iter_case_ids(self) -> None:
        report = EvalReport(cases=[EvalCase(note_id="a"), EvalCase(note_id="b")])
        assert list(iter_case_ids(report)) == ["a", "b"]


class TestSeedDataEvaluation:
    """The evaluation has to run honestly on the real seed notes."""

    def test_evaluates_the_seed_notes(self, indexed: tuple[Config, DecisionStore]) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())

        assert report.total == 6
        for case in report.cases:
            assert case.true_failed_assumptions
            assert case.surfaced_count >= 0
            assert 0.0 <= case.best_match <= 1.0

    def test_report_distinguishes_retrieved_from_surfaced(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        """The report must say how many failures were found and how many passed."""
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        for case in report.cases:
            assert case.retrieved_count >= case.surfaced_count
        text = format_report(report)
        assert "retrieved" in text
        assert "above the" in text

    def test_the_rate_is_reported_as_it_is(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        """The harness must not inflate its own number."""
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        expected = sum(1 for case in report.cases if case.hit)
        assert report.hits == expected
        assert report.hit_rate == expected / report.total

    def test_a_note_never_matches_its_own_failure(
        self, indexed: tuple[Config, DecisionStore]
    ) -> None:
        config, store = indexed
        report = run_eval(config, store, embedder=WordEmbedder())
        for case in report.cases:
            assert case.best_match_source != case.note_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
