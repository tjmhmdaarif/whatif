"""Tests for scoring.py - the weighted comparison computed in code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from whatif.notes import Decision, parse_all_decisions
from whatif.config import create_default_config
from whatif.scoring import (
    CriterionResult,
    OptionResult,
    ScoringResult,
    compute_scores,
    criterion_weights,
    format_comparison_table,
    format_criterion_table,
    iter_scored_options,
    leading_option,
    summarize,
    weighted_total,
)


def make_decision(
    options: list[str],
    criteria: list[dict],
    scores: dict[str, dict[str, int]],
    chosen: str = "",
) -> Decision:
    """Build a Decision directly, bypassing file parsing."""
    return Decision(
        id="2026-01-15-test",
        type="decision",
        status="open",
        created=__import__("datetime").date(2026, 1, 15),
        domain="other",
        options=options,
        chosen=chosen,
        criteria=criteria,
        scores=scores,
        assumptions=[],
        risks=[],
        outcome={"reviewed_on": None, "result": "", "failed_assumptions": []},
        tags=[],
    )


def make_raw_decision(
    options: list[str],
    criteria: list[dict],
    scores: dict[str, dict[str, int]],
    chosen: str = "",
) -> Decision:
    """Build a Decision without running schema validation.

    The Decision dataclass rejects empty criteria and fewer than two options.
    Some scoring paths need to be reachable with that data, so this bypasses
    ``__post_init__``.
    """
    decision = Decision.__new__(Decision)
    decision.id = "2026-01-15-test"
    decision.type = "decision"
    decision.status = "open"
    decision.created = __import__("datetime").date(2026, 1, 15)
    decision.domain = "other"
    decision.options = options
    decision.chosen = chosen
    decision.criteria = criteria
    decision.scores = scores
    decision.assumptions = []
    decision.risks = []
    decision.outcome = {"reviewed_on": None, "result": "", "failed_assumptions": []}
    decision.tags = []
    decision.context = ""
    decision.options_body = ""
    decision.decision_body = ""
    decision.outcome_body = ""
    decision.file_path = None
    decision.content_hash = ""
    return decision


@pytest.fixture
def seeded_decisions(temp_dir: Path) -> list[Decision]:
    """The 10 seed decisions, parsed."""
    vault = temp_dir / "vault"
    (vault / "Decisions").mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "decisions"
    for note in source.glob("*.md"):
        (vault / "Decisions" / note.name).write_text(
            note.read_text(encoding="utf-8"), encoding="utf-8"
        )
    decisions, _ = parse_all_decisions(create_default_config(vault))
    return decisions


class TestWeightedTotal:
    def test_simple_sum(self) -> None:
        total, covered, scored = weighted_total(
            {"cost": 4, "battery": 3}, {"cost": 3, "battery": 2}
        )
        assert total == 4 * 3 + 3 * 2
        assert covered == 5
        assert set(scored) == {"cost", "battery"}

    def test_empty_scores(self) -> None:
        total, covered, scored = weighted_total({}, {"cost": 3})
        assert total == 0
        assert covered == 0
        assert scored == []

    def test_criterion_without_weight_ignored(self) -> None:
        total, covered, scored = weighted_total({"cost": 5, "unweighted": 5}, {"cost": 2})
        assert total == 10
        assert covered == 2
        assert scored == ["cost"]

    def test_missing_criterion_contributes_nothing(self) -> None:
        total, covered, _ = weighted_total({"cost": 5}, {"cost": 2, "battery": 3})
        assert total == 10
        assert covered == 2

    def test_zero_weight_criterion(self) -> None:
        total, covered, _ = weighted_total({"cost": 5}, {"cost": 0})
        assert total == 0
        assert covered == 0


class TestCriterionWeights:
    def test_extracts_weights(self) -> None:
        weights = criterion_weights([{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}])
        assert weights == {"cost": 3, "battery": 2}

    def test_skips_invalid_entries(self) -> None:
        weights = criterion_weights(
            [{"name": "cost", "weight": 3}, {"name": "", "weight": 1}, {"weight": 2}]
        )
        assert weights == {"cost": 3}

    def test_skips_negative_weight(self) -> None:
        assert criterion_weights([{"name": "cost", "weight": -1}]) == {}


class TestComputeScores:
    def test_computes_totals(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}],
            scores={"A": {"cost": 4, "battery": 3}, "B": {"cost": 2, "battery": 5}},
        )
        result = compute_scores(decision)

        # A: 4*3 + 3*2 = 18.  B: 2*3 + 5*2 = 16.
        assert result.option("A").total == 18
        assert result.option("B").total == 16
        assert result.leader == "A"
        assert result.ranking == ["A", "B"]

    def test_max_points(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}],
            scores={"A": {"cost": 4, "battery": 3}, "B": {"cost": 2, "battery": 5}},
        )
        result = compute_scores(decision)
        assert result.max_points == 5 * (3 + 2)

    def test_tie_detected(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 3}, "B": {"cost": 3}},
        )
        result = compute_scores(decision)
        assert result.leader is None
        assert result.is_tie is True
        assert set(result.tied_leaders) == {"A", "B"}
        assert any("tied" in w for w in result.warnings)

    def test_no_tie_gives_single_leader(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 3}, "B": {"cost": 4}},
        )
        result = compute_scores(decision)
        assert result.leader == "B"
        assert result.is_tie is False

    def test_missing_score_for_one_option(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}],
            scores={"A": {"cost": 4, "battery": 3}, "B": {"cost": 2}},
        )
        result = compute_scores(decision)

        b = result.option("B")
        assert b.missing_criteria == ["battery"]
        assert b.complete is False
        assert b.covered_weight == 3
        assert b.total_weight == 5
        assert b.coverage == pytest.approx(0.6)
        assert any("B" in w and "battery" in w for w in result.warnings)

    def test_option_with_no_scores_at_all(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}},
        )
        result = compute_scores(decision)

        b = result.option("B")
        assert b.total == 0
        assert b.scored_criteria == []
        assert b.complete is False
        assert any("No scores entered for 'B'" in w for w in result.warnings)

    def test_score_for_unweighted_criterion_flagged(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4, "vibes": 5}, "B": {"cost": 2}},
        )
        result = compute_scores(decision)

        assert result.option("A").extra_criteria == ["vibes"]
        assert any("not weighted" in w for w in result.warnings)
        # The unweighted criterion must not inflate the total.
        assert result.option("A").total == 12

    def test_no_criteria(self) -> None:
        decision = make_raw_decision(options=["A", "B"], criteria=[], scores={})
        result = compute_scores(decision)
        assert result.has_data is False
        assert any("No criteria" in w for w in result.warnings)

    def test_no_scores(self) -> None:
        decision = make_decision(
            options=["A", "B"], criteria=[{"name": "cost", "weight": 3}], scores={}
        )
        result = compute_scores(decision)
        assert result.has_data is False
        assert any("No scores" in w for w in result.warnings)

    def test_criterion_result_per_criterion(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        result = compute_scores(decision)
        criterion = result.criteria[0]
        assert criterion.name == "cost"
        assert criterion.weight == 3
        assert criterion.scores == {"A": 4, "B": 2}
        assert criterion.weighted == {"A": 12.0, "B": 6.0}
        assert criterion.max_points == 15

    def test_to_dict_is_serialisable(self) -> None:
        import json

        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        result = compute_scores(decision)
        payload = json.dumps(result.to_dict())
        assert "leader" in payload


class TestLeadingOption:
    def test_uses_recorded_choice_first(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 5}, "B": {"cost": 1}},
            chosen="A",
        )
        # A leads on score but the note records B as the choice.
        assert leading_option(decision) == "A"

    def test_uses_computed_leader_when_no_choice(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 5}, "B": {"cost": 1}},
        )
        assert leading_option(decision) == "A"

    def test_tie_falls_back_to_first_option(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 3}, "B": {"cost": 3}},
        )
        assert leading_option(decision) == "A"

    def test_recorded_choice_wins_over_computed_leader(self) -> None:
        """The recorded choice is the lean, even if the scores point elsewhere."""
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 5}, "B": {"cost": 1}},
            chosen="B",
        )
        assert leading_option(decision) == "B"

    def test_no_options_returns_none(self) -> None:
        decision = make_raw_decision(
            options=[], criteria=[{"name": "cost", "weight": 1}], scores={}
        )
        assert leading_option(decision) is None


class TestComparisonTable:
    def test_contains_computed_label(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert "COMPUTED" in table

    def test_contains_option_names_and_totals(self) -> None:
        decision = make_decision(
            options=["Option A", "Option B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"Option A": {"cost": 4}, "Option B": {"cost": 2}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert "Option A" in table
        assert "Option B" in table
        assert "12" in table  # 4 * 3
        assert "6" in table  # 2 * 3

    def test_header_shows_weights(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert "cost (w3)" in table

    def test_missing_score_shown_as_dash(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}],
            scores={"A": {"cost": 4, "battery": 3}, "B": {"cost": 2}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert "incomplete scoring" in table
        assert "battery" in table

    def test_tie_is_noted(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 3}, "B": {"cost": 3}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert "tied" in table
        assert "A, B are tied at 3 points" in table

    def test_tie_note_uses_the_tied_total_not_the_first_option(self) -> None:
        """A tie must report the tied total, not the first option's total."""
        decision = make_decision(
            options=["A", "B", "C"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 1}, "B": {"cost": 3}, "C": {"cost": 3}},
        )
        result = compute_scores(decision)
        assert result.tied_leaders == ["B", "C"]
        assert result.is_tie is True
        table = format_comparison_table(result)
        assert "B, C are tied at 3 points" in table
        assert "tied at 1" not in table

    def test_no_percentages_in_table(self) -> None:
        """The table must not present scores as percentages or probabilities."""
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}, {"name": "battery", "weight": 2}],
            scores={"A": {"cost": 4, "battery": 3}, "B": {"cost": 2, "battery": 5}},
        )
        table = format_comparison_table(compute_scores(decision))
        assert not re.search(r"\d+\s*%", table)
        assert not re.search(r"\d+\s*(percent|probability)", table, re.IGNORECASE)

    def test_criterion_table(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        table = format_criterion_table(compute_scores(decision))
        assert "12" in table
        assert "6" in table


class TestSummarize:
    def test_leader_summary(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 3}],
            scores={"A": {"cost": 4}, "B": {"cost": 2}},
        )
        summary = summarize(compute_scores(decision))
        assert "A leads" in summary
        assert "COMPUTED" in summary

    def test_tie_summary(self) -> None:
        decision = make_decision(
            options=["A", "B"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 3}, "B": {"cost": 3}},
        )
        summary = summarize(compute_scores(decision))
        assert "tied" in summary

    def test_no_data_summary(self) -> None:
        decision = make_decision(
            options=["A", "B"], criteria=[{"name": "cost", "weight": 1}], scores={}
        )
        assert "No scores" in summarize(compute_scores(decision))


class TestIterScoredOptions:
    def test_ordered_by_ranking(self) -> None:
        decision = make_decision(
            options=["A", "B", "C"],
            criteria=[{"name": "cost", "weight": 1}],
            scores={"A": {"cost": 1}, "B": {"cost": 3}, "C": {"cost": 2}},
        )
        result = compute_scores(decision)
        assert [o.option for o in iter_scored_options(result)] == ["B", "C", "A"]


class TestSeedDecisions:
    def test_all_seed_decisions_score_without_error(self, seeded_decisions: list[Decision]) -> None:
        assert len(seeded_decisions) == 10
        for decision in seeded_decisions:
            result = compute_scores(decision)
            assert result.has_data, f"{decision.id} produced no usable scores"
            assert result.leader is not None or result.is_tie

    def test_every_seed_option_is_scored_completely(
        self, seeded_decisions: list[Decision]
    ) -> None:
        """The seed notes should be complete so the brief has a clean table."""
        for decision in seeded_decisions:
            result = compute_scores(decision)
            for option in result.options:
                assert option.complete, (
                    f"{decision.id} option {option.option} is missing: {option.missing_criteria}"
                )

    def test_leader_matches_recorded_choice_in_seed_data(
        self, seeded_decisions: list[Decision]
    ) -> None:
        """In the seed data the recorded choice should agree with the computed leader."""
        for decision in seeded_decisions:
            if not decision.chosen:
                continue
            result = compute_scores(decision)
            assert result.leader == decision.chosen, (
                f"{decision.id}: recorded {decision.chosen!r} but computed {result.leader!r}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
