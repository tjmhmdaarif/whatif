"""Weighted scoring of decision options.

Every number in the brief's comparison table comes from this module. The LLM never
computes a score, a total, or a ranking; it only writes prose around numbers that
were calculated here.

The scoring rule is deliberately simple and stated in the output: for each option,
multiply the score of every criterion by that criterion's weight, then add them up.
The maximum possible total is ``5 * sum(weights)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from whatif.notes import Criterion, Decision

logger = logging.getLogger(__name__)

# The scale used in the note schema.
MIN_SCORE = 1
MAX_SCORE = 5


@dataclass
class CriterionResult:
    """One criterion and how each option scored on it."""

    name: str
    weight: int
    scores: dict[str, int] = field(default_factory=dict)
    weighted: dict[str, float] = field(default_factory=dict)

    @property
    def max_points(self) -> float:
        """Maximum points this criterion can contribute to any option."""
        return self.weight * MAX_SCORE

    @property
    def min_points(self) -> float:
        """Minimum points this criterion can contribute to any option."""
        return self.weight * MIN_SCORE


@dataclass
class OptionResult:
    """The weighted result for a single option."""

    option: str
    total: float = 0.0
    covered_weight: int = 0
    total_weight: int = 0
    scored_criteria: list[str] = field(default_factory=list)
    missing_criteria: list[str] = field(default_factory=list)
    extra_criteria: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True if the option has a score for every criterion."""
        return not self.missing_criteria and self.covered_weight == self.total_weight

    @property
    def coverage(self) -> float:
        """Share of the total weight that was actually scored, from 0 to 1."""
        if self.total_weight == 0:
            return 0.0
        return self.covered_weight / self.total_weight


@dataclass
class ScoringResult:
    """The full computed comparison for a decision."""

    criteria: list[CriterionResult] = field(default_factory=list)
    options: list[OptionResult] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    leader: str | None = None
    tied_leaders: list[str] = field(default_factory=list)
    max_points: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        """True if at least one option has a usable score."""
        return any(o.covered_weight > 0 for o in self.options)

    @property
    def is_tie(self) -> bool:
        """True if more than one option shares the top total."""
        return len(self.tied_leaders) > 1

    def option(self, name: str) -> OptionResult | None:
        """Return the result for one option, or None if it is unknown."""
        for result in self.options:
            if result.option == name:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return {
            "leader": self.leader,
            "tied_leaders": list(self.tied_leaders),
            "is_tie": self.is_tie,
            "max_points": self.max_points,
            "warnings": list(self.warnings),
            "ranking": list(self.ranking),
            "criteria": [
                {
                    "name": c.name,
                    "weight": c.weight,
                    "scores": dict(c.scores),
                    "weighted": dict(c.weighted),
                }
                for c in self.criteria
            ],
            "options": [
                {
                    "option": o.option,
                    "total": o.total,
                    "covered_weight": o.covered_weight,
                    "total_weight": o.total_weight,
                    "scored_criteria": list(o.scored_criteria),
                    "missing_criteria": list(o.missing_criteria),
                    "extra_criteria": list(o.extra_criteria),
                    "contributions": dict(o.contributions),
                    "complete": o.complete,
                }
                for o in self.options
            ],
        }


def criterion_weights(criteria: Sequence[Criterion]) -> dict[str, int]:
    """Return a mapping of criterion name to weight."""
    weights: dict[str, int] = {}
    for criterion in criteria:
        name = criterion.get("name")
        weight = criterion.get("weight")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(weight, int) or weight < 0:
            continue
        weights[name] = weight
    return weights


def weighted_total(
    option_scores: dict[str, int], weights: dict[str, int]
) -> tuple[float, int, list[str]]:
    """Compute the weighted total for one option.

    Args:
        option_scores: Mapping of criterion name to score for this option.
        weights: Mapping of criterion name to weight.

    Returns:
        Tuple of (total points, covered weight, list of scored criterion names).
        Criteria with no weight are ignored, and criteria the option did not score
        contribute nothing.
    """
    total = 0.0
    covered = 0
    scored: list[str] = []

    for name, score in option_scores.items():
        weight = weights.get(name)
        if weight is None:
            continue
        if not isinstance(score, int):
            continue
        total += score * weight
        covered += weight
        scored.append(name)

    return total, covered, scored


def compute_scores(decision: Decision) -> ScoringResult:
    """Compute the weighted comparison for a decision.

    The computation is entirely deterministic and happens in code. The LLM is not
    involved, and does not see the raw numbers until the brief is assembled.

    Args:
        decision: The decision to score.

    Returns:
        A :class:`ScoringResult` with per-criterion and per-option breakdowns.
    """
    result = ScoringResult()
    weights = criterion_weights(decision.criteria)
    total_weight = sum(weights.values())
    result.max_points = float(total_weight * MAX_SCORE)

    if not decision.criteria:
        result.warnings.append("No criteria defined, so options cannot be compared.")
        return result

    if not decision.scores:
        result.warnings.append("No scores entered, so options cannot be compared.")
        # Fall through: the options are still built so the brief and the canvas can
        # show which options exist and that they are unscored.

    # Per-criterion view.
    for criterion in decision.criteria:
        name = criterion.get("name", "")
        weight = criterion.get("weight", 0)
        criterion_result = CriterionResult(name=name, weight=weight)
        for option in decision.options:
            option_scores = decision.scores.get(option, {})
            score = option_scores.get(name)
            if isinstance(score, int):
                criterion_result.scores[option] = score
                criterion_result.weighted[option] = float(score * weight)
        result.criteria.append(criterion_result)

    # Per-option view.
    for option in decision.options:
        option_scores = decision.scores.get(option, {})
        option_result = OptionResult(option=option, total_weight=total_weight)

        for name, score in option_scores.items():
            if name not in weights:
                option_result.extra_criteria.append(name)

        for criterion in decision.criteria:
            name = criterion.get("name", "")
            weight = criterion.get("weight", 0)
            score = option_scores.get(name)
            if isinstance(score, int):
                option_result.contributions[name] = float(score * weight)
                option_result.scored_criteria.append(name)
            else:
                option_result.missing_criteria.append(name)

        total, covered, _ = weighted_total(option_scores, weights)
        option_result.total = total
        option_result.covered_weight = covered
        result.options.append(option_result)

    # Ranking. Incomplete options are ranked by total but flagged, so a missing
    # score never silently makes an option look competitive.
    result.ranking = [o.option for o in sorted(result.options, key=lambda o: -o.total)]

    for option_result in result.options:
        if not option_result.scored_criteria:
            result.warnings.append(
                f"No scores entered for '{option_result.option}'."
            )
        elif option_result.missing_criteria:
            result.warnings.append(
                f"'{option_result.option}' has no score for: "
                f"{', '.join(option_result.missing_criteria)}."
            )
        if option_result.extra_criteria:
            result.warnings.append(
                f"'{option_result.option}' scored criteria that are not weighted: "
                f"{', '.join(option_result.extra_criteria)}."
            )

    # Leader detection.
    if result.options:
        best = max(o.total for o in result.options)
        result.tied_leaders = [o.option for o in result.options if o.total == best]
        if len(result.tied_leaders) == 1:
            result.leader = result.tied_leaders[0]
        else:
            result.leader = None
            result.warnings.append(
                f"Options are tied at {best:g} points: {', '.join(result.tied_leaders)}."
            )

    return result


def leading_option(decision: Decision, scoring: ScoringResult | None = None) -> str | None:
    """Return the option the user is leaning toward.

    If the note records a choice, that is the leaning. Otherwise the computed
    leader is used. A tie falls back to the first option in the note, since the
    counterargument still has to argue against something.
    """
    if decision.chosen and decision.chosen in decision.options:
        return decision.chosen

    if scoring is None:
        scoring = compute_scores(decision)

    if scoring.leader:
        return scoring.leader
    if scoring.tied_leaders:
        return scoring.tied_leaders[0]
    if decision.options:
        return decision.options[0]
    return None


def format_comparison_table(scoring: ScoringResult) -> str:
    """Render the weighted comparison as a Markdown table.

    The table is labeled ``COMPUTED`` because every number in it was calculated in
    code from the user's own scores and weights, not written by the model.
    """
    lines: list[str] = []

    lines.append("| Option | " + " | ".join(
        f"{c.name} (w{c.weight})" for c in scoring.criteria
    ) + " | **Total (COMPUTED)** |")
    lines.append("|---" * (len(scoring.criteria) + 2) + "|")

    ordered = sorted(scoring.options, key=lambda o: -o.total)
    for option_result in ordered:
        cells = []
        for criterion in scoring.criteria:
            if option_result.option in criterion.scores:
                cells.append(str(criterion.scores[option_result.option]))
            else:
                cells.append("-")
        cells.append(f"**{option_result.total:g}**")
        lines.append(f"| {option_result.option} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        f"*COMPUTED: weighted total = sum of (score x weight). "
        f"Maximum possible is {scoring.max_points:g} points "
        f"(5 x sum of weights = 5 x {int(scoring.max_points / MAX_SCORE)}).*"
    )

    incomplete = [o for o in scoring.options if not o.complete]
    if incomplete:
        details = "; ".join(
            f"{o.option} missing {', '.join(o.missing_criteria) or 'no scores'}"
            for o in incomplete
        )
        lines.append(f"*COMPUTED note: incomplete scoring - {details}.*")

    if scoring.is_tie:
        tied_total = scoring.option(scoring.tied_leaders[0]).total
        lines.append(
            f"*COMPUTED note: {', '.join(scoring.tied_leaders)} are tied at "
            f"{tied_total:g} points.*"
        )

    return "\n".join(lines)


def format_criterion_table(scoring: ScoringResult) -> str:
    """Render the per-criterion weighted contributions as a Markdown table."""
    lines: list[str] = []
    lines.append("| Criterion | Weight | " + " | ".join(
        o.option for o in scoring.options
    ) + " |")
    lines.append("|---" * (len(scoring.options) + 2) + "|")

    for criterion in scoring.criteria:
        cells = []
        for option_result in scoring.options:
            if option_result.option in criterion.weighted:
                cells.append(f"{criterion.weighted[option_result.option]:g}")
            else:
                cells.append("-")
        lines.append(
            f"| {criterion.name} | {criterion.weight} | " + " | ".join(cells) + " |"
        )

    return "\n".join(lines)


def summarize(scoring: ScoringResult) -> str:
    """Return a one-line plain summary of the computed result."""
    if not scoring.has_data:
        return "No scores to compare."
    if scoring.leader:
        return (
            f"{scoring.leader} leads with {scoring.option(scoring.leader).total:g} points "
            f"(COMPUTED)."
        )
    if scoring.tied_leaders:
        total = scoring.option(scoring.tied_leaders[0]).total
        return f"{' and '.join(scoring.tied_leaders)} are tied at {total:g} points (COMPUTED)."
    return "No scores to compare."


def iter_scored_options(scoring: ScoringResult) -> Iterable[OptionResult]:
    """Yield the option results in ranking order."""
    order = {name: i for i, name in enumerate(scoring.ranking)}
    for option_result in sorted(
        scoring.options, key=lambda o: order.get(o.option, len(order))
    ):
        yield option_result
