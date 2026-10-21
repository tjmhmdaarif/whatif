"""Tests for brief.py - tag validation, retry logic, and brief assembly."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

from whatif.brief import (
    DEFAULT_LEGEND,
    REASON_CERTAINTY,
    REASON_MISSING_TAG,
    REASON_MISPLACED_TAG,
    REASON_PROBABILITY,
    REASON_UNKNOWN_NOTE,
    SYSTEM_PROMPT,
    BriefGenerator,
    BriefResult,
    ValidationIssue,
    ValidationReport,
    build_prompt,
    build_retry_prompt,
    ends_with_tag,
    find_certainty,
    find_probability,
    find_tags,
    mark_unverified,
    neutralize_probabilities,
    parse_relevance_lines,
    parse_sections,
    render_brief,
    split_claims,
    strip_tags,
    validate_claims,
)
from whatif.config import Config, create_default_config
from whatif.notes import Decision
from whatif.retrieve import FailedAssumption, SimilarDecision
from whatif.scoring import compute_scores

ALLOWED = {"2026-01-15-course-choice", "2026-03-10-club-role"}


def make_decision(
    options: list[str] | None = None,
    criteria: list[dict] | None = None,
    scores: dict[str, dict[str, int]] | None = None,
    chosen: str = "",
    assumptions: list[str] | None = None,
    context: str = "Choosing between two options for a term project.",
) -> Decision:
    """Build a Decision for testing."""
    options = options or ["Option A", "Option B"]
    criteria = criteria or [{"name": "price", "weight": 3}, {"name": "speed", "weight": 2}]
    scores = scores or {
        "Option A": {"price": 4, "speed": 3},
        "Option B": {"price": 2, "speed": 5},
    }
    return Decision(
        id="2026-11-01-new-decision",
        type="decision",
        status="open",
        created=datetime.date(2026, 11, 1),
        domain="tools",
        options=options,
        chosen=chosen,
        criteria=criteria,
        scores=scores,
        assumptions=assumptions if assumptions is not None else ["I will finish this in a month."],
        risks=[],
        outcome={"reviewed_on": None, "result": "", "failed_assumptions": []},
        tags=[],
        context=context,
    )


def make_similar(note_id: str, similarity: float = 0.8) -> SimilarDecision:
    """Build a SimilarDecision for testing."""
    return SimilarDecision(
        note_id=note_id,
        similarity=similarity,
        domain="study",
        status="decided",
        created="2026-01-15",
        summary=f"{note_id} summary",
        options=["A", "B"],
        chosen="A",
        failed_assumptions=["Something went wrong."],
    )


def make_failed(text: str, note_id: str, similarity: float = 0.7) -> FailedAssumption:
    """Build a FailedAssumption for testing."""
    return FailedAssumption(
        text=text, source_note_id=note_id, similarity=similarity, source_domain="study"
    )


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_find_evidence_tag(self) -> None:
        tags = find_tags("A claim [EVIDENCE: 2026-01-15-course-choice]")
        assert len(tags) == 1
        assert tags[0].is_evidence
        assert tags[0].value == "2026-01-15-course-choice"

    def test_find_estimate_tag(self) -> None:
        tags = find_tags("A judgement [ESTIMATE]")
        assert len(tags) == 1
        assert tags[0].is_estimate

    def test_find_multiple_tags(self) -> None:
        tags = find_tags("One [EVIDENCE: a] and two [ESTIMATE]")
        assert len(tags) == 2

    def test_ends_with_tag(self) -> None:
        assert ends_with_tag("Claim [EVIDENCE: a]") is True
        assert ends_with_tag("Claim [ESTIMATE].") is True
        assert ends_with_tag("Claim [EVIDENCE: a].") is True
        assert ends_with_tag("Claim") is False
        assert ends_with_tag("Claim [EVIDENCE: a] but more text") is False

    def test_strip_tags(self) -> None:
        assert strip_tags("Claim [EVIDENCE: a]") == "Claim"
        assert strip_tags("Claim [ESTIMATE]") == "Claim"


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


class TestForbiddenLanguage:
    def test_finds_percentage(self) -> None:
        assert find_probability("There is an 80% chance it works.") is not None
        assert find_probability("It costs 20% less.") is not None

    def test_finds_probability_word(self) -> None:
        assert find_probability("The probability of success is high.") is not None

    def test_finds_odds(self) -> None:
        assert find_probability("The odds are 3 in 4.") is None  # "3 in 4" is not matched
        assert find_probability("The odds favour it.") is None

    def test_no_false_positive_on_plain_claim(self) -> None:
        assert find_probability("The cheaper option won on price.") is None
        assert find_probability("It probably works.") is None

    def test_finds_certainty(self) -> None:
        assert find_certainty("This is guaranteed to work.") == "guaranteed"
        assert find_certainty("It will definitely work.") is not None

    def test_no_false_positive_certainty(self) -> None:
        assert find_certainty("I am certain about the price.") is None

    def test_neutralize_percentages(self) -> None:
        result = neutralize_probabilities("There is an 80% chance it works.")
        assert "%" not in result
        assert "80" not in result

    def test_neutralize_keeps_meaning(self) -> None:
        result = neutralize_probabilities("An 80% chance of success")
        assert "80%" not in result
        assert "chance" in result

    def test_neutralize_proportion(self) -> None:
        result = neutralize_probabilities("80% of employers said no")
        assert "80%" not in result
        assert "employers" in result

    def test_neutralize_probability_word(self) -> None:
        result = neutralize_probabilities("The probability of success is high")
        assert "probability" not in result.lower()


# ---------------------------------------------------------------------------
# Claim splitting
# ---------------------------------------------------------------------------


class TestSplitClaims:
    def test_headings_carry_no_claims(self) -> None:
        claims = split_claims("## Decision summary")
        assert claims == []

    def test_table_rows_carry_no_claims(self) -> None:
        claims = split_claims("| Option | Total |\n|---|---|\n| A | 12 |")
        assert claims == []

    def test_bullet_is_split_into_sentences(self) -> None:
        claims = split_claims("- First claim. Second claim.")
        assert len(claims) == 2
        assert claims[0].text == "First claim."
        assert claims[1].text == "Second claim."

    def test_paragraph_is_split_into_sentences(self) -> None:
        claims = split_claims("One. Two. Three.")
        assert len(claims) == 3

    def test_horizontal_rule_carry_no_claims(self) -> None:
        assert split_claims("---") == []

    def test_code_fence_content_ignored(self) -> None:
        claims = split_claims("```\nThis is code. Not a claim.\n```")
        assert claims == []

    def test_empty_text(self) -> None:
        assert split_claims("") == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_evidence_passes(self) -> None:
        text = "The cost issue came up before [EVIDENCE: 2026-01-15-course-choice]"
        report = validate_claims(text, ALLOWED)
        assert report.passed
        assert report.claim_count == 1

    def test_valid_estimate_passes(self) -> None:
        report = validate_claims("This is my own read of it [ESTIMATE]", ALLOWED)
        assert report.passed

    def test_missing_tag_fails(self) -> None:
        report = validate_claims("A claim with no tag at all", ALLOWED)
        assert not report.passed
        assert report.issues[0].reason == REASON_MISSING_TAG

    def test_unknown_note_fails(self) -> None:
        report = validate_claims("A claim [EVIDENCE: 2026-99-99-not-retrieved]", ALLOWED)
        assert not report.passed
        assert report.issues[0].reason == REASON_UNKNOWN_NOTE

    def test_misplaced_tag_fails(self) -> None:
        report = validate_claims(
            "A claim [EVIDENCE: 2026-01-15-course-choice] with trailing text", ALLOWED
        )
        assert not report.passed
        assert any(i.reason == REASON_MISPLACED_TAG for i in report.issues)

    def test_probability_fails(self) -> None:
        report = validate_claims("It has an 80% chance [ESTIMATE]", ALLOWED)
        assert not report.passed
        assert any(i.reason == REASON_PROBABILITY for i in report.issues)

    def test_certainty_fails(self) -> None:
        report = validate_claims("This is guaranteed [ESTIMATE]", ALLOWED)
        assert not report.passed
        assert any(i.reason == REASON_CERTAINTY for i in report.issues)

    def test_multiple_issues_reported(self) -> None:
        report = validate_claims(
            "No tag here. And 80% chance [ESTIMATE]", ALLOWED
        )
        assert len(report.issues) >= 2

    def test_by_reason(self) -> None:
        report = validate_claims("No tag. Also 50% [ESTIMATE]", ALLOWED)
        counts = report.by_reason()
        assert REASON_MISSING_TAG in counts
        assert REASON_PROBABILITY in counts

    def test_headings_and_tables_are_not_claims(self) -> None:
        report = validate_claims("## Heading\n\n| a | b |\n|---|---|\n", ALLOWED)
        assert report.passed
        assert report.claim_count == 0

    def test_unverified_tag_counts_as_present(self) -> None:
        report = validate_claims("A claim [UNVERIFIED]", ALLOWED)
        # UNVERIFIED is not an evidence tag and not an estimate, so it is a failure.
        assert not report.passed

    def test_as_dict(self) -> None:
        report = validate_claims("no tag", ALLOWED)
        payload = report.as_dict()
        assert payload["passed"] is False
        assert payload["issue_count"] == 1


# ---------------------------------------------------------------------------
# Marking unverified
# ---------------------------------------------------------------------------


class TestMarkUnverified:
    def test_missing_tag_gets_marked(self) -> None:
        text = "A claim with no tag"
        report = validate_claims(text, ALLOWED)
        result = mark_unverified(text, report)
        assert "[UNVERIFIED]" in result

    def test_unknown_note_is_replaced(self) -> None:
        text = "A claim [EVIDENCE: 2026-99-99-nope]"
        report = validate_claims(text, ALLOWED)
        result = mark_unverified(text, report)
        assert "2026-99-99-nope" not in result
        assert "[UNVERIFIED]" in result

    def test_probability_is_neutralised(self) -> None:
        text = "It has an 80% chance of working [ESTIMATE]"
        report = validate_claims(text, ALLOWED)
        result = mark_unverified(text, report)
        assert "80%" not in result
        assert "[UNVERIFIED]" in result

    def test_valid_text_untouched(self) -> None:
        text = "A good claim [EVIDENCE: 2026-01-15-course-choice]"
        report = validate_claims(text, ALLOWED)
        assert mark_unverified(text, report) == text

    def test_only_bad_lines_are_changed(self) -> None:
        text = "Good claim [EVIDENCE: 2026-01-15-course-choice]\nBad claim with no tag"
        report = validate_claims(text, ALLOWED)
        result = mark_unverified(text, report)
        assert "Good claim [EVIDENCE: 2026-01-15-course-choice]" in result
        assert result.split("\n")[0] == "Good claim [EVIDENCE: 2026-01-15-course-choice]"

    def test_bullets_keep_their_marker(self) -> None:
        text = "- A bullet with no tag"
        report = validate_claims(text, ALLOWED)
        result = mark_unverified(text, report)
        assert result.startswith("- ")


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_marker_sections(self) -> None:
        text = """<decision_summary>
A summary [ESTIMATE]
</decision_summary>
<similar_notes>
2026-01-15-course-choice: relevant
</similar_notes>
<counterargument>
A counter [EVIDENCE: 2026-01-15-course-choice]
</counterargument>
<open_questions>
A question [ESTIMATE]
</open_questions>"""
        sections = parse_sections(text)
        assert "A summary" in sections["summary"]
        assert "relevant" in sections["similar"]
        assert "A counter" in sections["counterargument"]
        assert "A question" in sections["questions"]

    def test_heading_fallback(self) -> None:
        text = """## Decision summary
A summary [ESTIMATE]

## Similar past decisions
2026-01-15-course-choice: relevant

## Counterargument
A counter [ESTIMATE]

## Open questions
A question [ESTIMATE]"""
        sections = parse_sections(text)
        assert "A summary" in sections["summary"]
        assert "relevant" in sections["similar"]
        assert "A counter" in sections["counterargument"]
        assert "A question" in sections["questions"]

    def test_missing_sections_become_empty(self) -> None:
        sections = parse_sections("nothing useful here")
        assert sections["summary"] == ""
        assert sections["counterargument"] == ""


class TestParseRelevanceLines:
    def test_parses_note_id_lines(self) -> None:
        text = "2026-01-15-course-choice: This one is about cost.\n2026-03-10-club-role: About time."
        result = parse_relevance_lines(text, ALLOWED)
        assert result["2026-01-15-course-choice"] == "This one is about cost."
        assert result["2026-03-10-club-role"] == "About time."

    def test_ignores_unknown_ids(self) -> None:
        text = "2026-99-99-unknown: nope"
        assert parse_relevance_lines(text, ALLOWED) == {}

    def test_ignores_bullets_markers(self) -> None:
        text = "- 2026-01-15-course-choice: relevant"
        assert "2026-01-15-course-choice" in parse_relevance_lines(text, ALLOWED)

    def test_ignores_lines_without_a_colon(self) -> None:
        assert parse_relevance_lines("just a sentence", ALLOWED) == {}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_contains_the_new_decision(self) -> None:
        decision = make_decision(context="I need to pick a note app.")
        prompt = build_prompt(decision, [], [], {}, "Option A")
        assert "I need to pick a note app." in prompt
        assert decision.id in prompt

    def test_prompt_contains_retrieved_notes(self) -> None:
        decision = make_decision()
        similar = [make_similar("2026-01-15-course-choice", 0.9)]
        prompt = build_prompt(decision, similar, [], {}, "Option A")
        assert "2026-01-15-course-choice" in prompt
        assert "note-id: 2026-01-15-course-choice" in prompt

    def test_prompt_lists_citable_ids(self) -> None:
        decision = make_decision()
        similar = [make_similar("2026-01-15-course-choice", 0.9)]
        prompt = build_prompt(decision, similar, [], {}, "Option A")
        assert "=== NOTE IDS YOU MAY CITE ===" in prompt

    def test_prompt_forbids_probabilities(self) -> None:
        prompt = build_prompt(make_decision(), [], [], {}, None)
        assert "probability" in prompt.lower()
        assert "80%" in prompt
        assert "guaranteed" in prompt.lower() or "certain" in prompt.lower()

    def test_prompt_requires_counter_argument(self) -> None:
        prompt = build_prompt(make_decision(), [], [], {}, "Option A")
        assert "against the option the user is leaning toward" in prompt

    def test_prompt_contains_failed_assumptions(self) -> None:
        decision = make_decision()
        failed = [make_failed("It failed before", "2026-03-10-club-role")]
        prompt = build_prompt(decision, [], failed, {}, None)
        assert "It failed before" in prompt

    def test_retry_prompt_lists_complaints(self) -> None:
        report = ValidationReport(
            issues=[ValidationIssue(REASON_MISSING_TAG, "a claim", "no tag")]
        )
        prompt = build_retry_prompt("ORIGINAL", report, ALLOWED)
        assert "ORIGINAL" in prompt
        assert "REJECTED" in prompt
        assert "no tag" in prompt.lower() or "no tag" in prompt

    def test_system_prompt_states_the_rules(self) -> None:
        assert "probability" in SYSTEM_PROMPT.lower()
        assert "EVIDENCE" in SYSTEM_PROMPT
        assert "counterargument" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderBrief:
    def _render(self, **overrides: Any) -> str:
        decision = make_decision()
        scoring = compute_scores(decision)
        defaults: dict[str, Any] = {
            "decision": decision,
            "similar": [make_similar("2026-01-15-course-choice", 0.812)],
            "failed": [make_failed("It failed before", "2026-03-10-club-role", 0.7)],
            "scoring": scoring,
            "sections": {
                "summary": "A summary [ESTIMATE]",
                "similar": "2026-01-15-course-choice: relevant because cost",
                "counterargument": "Against it [EVIDENCE: 2026-01-15-course-choice]",
                "questions": "What is the budget? [ESTIMATE]",
            },
            "relevance": {"2026-01-15-course-choice": "relevant because cost"},
            "leading": "Option A",
        }
        defaults.update(overrides)
        return render_brief(**defaults)

    def test_sections_appear_in_order(self) -> None:
        content = self._render()
        positions = [
            content.index("## Decision summary"),
            content.index("## Similar past decisions"),
            content.index("## Assumptions that failed before"),
            content.index("## Weighted comparison"),
            content.index("## Counterargument"),
            content.index("## Open questions"),
        ]
        assert positions == sorted(positions)

    def test_legend_is_at_the_top(self) -> None:
        content = self._render()
        assert content.index("[EVIDENCE: note-id]") < content.index("## Decision summary")
        assert "[ESTIMATE]" in content
        assert "[UNVERIFIED]" in content
        assert "COMPUTED" in content

    def test_weighted_table_is_computed(self) -> None:
        content = self._render()
        assert "Total (COMPUTED)" in content
        # Option A: 4*3 + 3*2 = 18
        assert "**18**" in content

    def test_similar_section_has_id_and_similarity(self) -> None:
        content = self._render()
        assert "2026-01-15-course-choice" in content
        assert "0.812" in content

    def test_failed_section_has_source_note(self) -> None:
        content = self._render()
        assert "It failed before" in content
        assert "2026-03-10-club-role" in content

    def test_counterargument_names_the_leading_option(self) -> None:
        content = self._render(leading="Option B")
        assert "## Counterargument: against Option B" in content

    def test_empty_similar_section(self) -> None:
        content = self._render(similar=[])
        assert "No past decisions were similar" in content

    def test_empty_failed_section(self) -> None:
        content = self._render(failed=[])
        assert "No past assumption failures" in content

    def test_validation_failure_is_noted(self) -> None:
        report = ValidationReport(issues=[ValidationIssue(REASON_MISSING_TAG, "x", "y")])
        content = self._render(validation=report)
        assert "[UNVERIFIED]" in content
        assert "did not pass validation" in content


# ---------------------------------------------------------------------------
# Generator with a fake model
# ---------------------------------------------------------------------------


class FakeClient:
    """Ollama client stand-in that returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt: str, model: str | None = None, system: str | None = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


class FakeRetriever:
    """Retriever stand-in returning fixed results."""

    def __init__(self, similar=None, failed=None) -> None:
        self.similar = similar if similar is not None else []
        self.failed = failed if failed is not None else []

    def retrieve(self, decision, k=5, note_threshold=0.0, assumption_threshold=0.0, **kwargs):
        return self.similar, self.failed


GOOD_RESPONSE = """<decision_summary>
The user is choosing between two options for a term project [ESTIMATE]
</decision_summary>
<similar_notes>
2026-01-15-course-choice: The cost tradeoff is the same shape
</similar_notes>
<counterargument>
A similar cost decision went badly last time [EVIDENCE: 2026-01-15-course-choice]
</counterargument>
<open_questions>
What is the real budget? [ESTIMATE]
</open_questions>"""

BAD_RESPONSE = """<decision_summary>
The user is choosing between two options for a term project
</decision_summary>
<similar_notes>
2026-01-15-course-choice: The cost tradeoff is the same shape
</similar_notes>
<counterargument>
A past note says this fails [EVIDENCE: 2026-99-99-invented]
There is an 80% chance this fails [EVIDENCE: 2026-01-15-course-choice]
</counterargument>
<open_questions>
What is the real budget? [ESTIMATE]
</open_questions>"""

# A response that invents a note id inside the similar-notes block, where the note
# id is the evidence rather than a tagged claim.
INVENTED_IN_SIMILAR_RESPONSE = """<decision_summary>
The user is choosing between two options for a term project [ESTIMATE]
</decision_summary>
<similar_notes>
2026-99-99-invented: this note does not exist
2026-01-15-course-choice: The cost tradeoff is the same shape
</similar_notes>
<counterargument>
A past note says this fails [EVIDENCE: 2026-01-15-course-choice]
</counterargument>
<open_questions>
What is the real budget? [ESTIMATE]
</open_questions>"""


@pytest.fixture
def config_with_retries(temp_dir: Path) -> Config:
    """A config with one retry allowed."""
    config = create_default_config(temp_dir)
    config.brief.max_retries = 1
    config.brief.legend = DEFAULT_LEGEND
    return config


class TestBriefGenerator:
    def test_generates_a_brief(self, config_with_retries: Config) -> None:
        client = FakeClient([GOOD_RESPONSE])
        retriever = FakeRetriever(
            similar=[make_similar("2026-01-15-course-choice", 0.8)],
            failed=[make_failed("It failed before", "2026-03-10-club-role", 0.7)],
        )
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())

        assert isinstance(result, BriefResult)
        assert "## Decision summary" in result.content
        assert "## Counterargument" in result.content
        assert result.validation is not None and result.validation.passed
        assert result.retries == 0

    def test_retries_once_then_succeeds(self, config_with_retries: Config) -> None:
        client = FakeClient([BAD_RESPONSE, GOOD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())

        assert client.calls == 2
        assert result.retries == 1
        assert result.validation is not None and result.validation.passed

    def test_retry_prompt_includes_the_complaints(self, config_with_retries: Config) -> None:
        client = FakeClient([BAD_RESPONSE, GOOD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        generator.generate(make_decision())

        assert len(client.prompts) == 2
        assert "REJECTED" in client.prompts[1]
        assert "2026-99-99-invented" in client.prompts[1]
        assert "80%" in client.prompts[1]

    def test_invented_id_in_similar_block_never_reaches_the_brief(
        self, config_with_retries: Config
    ) -> None:
        """The similar block is not tag-validated, so invented ids must be dropped."""
        client = FakeClient([INVENTED_IN_SIMILAR_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())

        assert "2026-99-99-invented" not in result.content
        # The real note still gets its deterministic summary as the relevance line.
        assert "2026-01-15-course-choice" in result.content

    def test_marks_unverified_after_failed_retry(self, config_with_retries: Config) -> None:
        client = FakeClient([BAD_RESPONSE, BAD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())

        assert client.calls == 2
        assert result.retries == 1
        assert result.validation is not None
        assert not result.validation.passed
        assert "[UNVERIFIED]" in result.content
        assert "2026-99-99-invented" not in result.content
        assert "80%" not in result.content

    def test_no_retry_when_disabled(self, temp_dir: Path) -> None:
        config = create_default_config(temp_dir)
        config.brief.max_retries = 0
        client = FakeClient([BAD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())

        assert client.calls == 1
        assert "[UNVERIFIED]" in result.content

    def test_brief_never_contains_a_percentage(self, config_with_retries: Config) -> None:
        client = FakeClient([BAD_RESPONSE, BAD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())
        assert "%" not in result.content

    def test_brief_cites_only_retrieved_notes(self, config_with_retries: Config) -> None:
        client = FakeClient([GOOD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())
        assert "2026-99-99" not in result.content

    def test_system_prompt_is_sent(self, config_with_retries: Config) -> None:
        captured: dict[str, Any] = {}

        class CapturingClient(FakeClient):
            def generate(self, prompt, model=None, system=None, **kwargs):
                captured["system"] = system
                captured["model"] = model
                return super().generate(prompt, model=model, system=system, **kwargs)

        client = CapturingClient([GOOD_RESPONSE])
        generator = BriefGenerator(config_with_retries, client, FakeRetriever())  # type: ignore[arg-type]
        generator.generate(make_decision())

        assert captured["system"] == SYSTEM_PROMPT
        assert captured["model"] == "llama3.1"

    def test_result_as_dict(self, config_with_retries: Config) -> None:
        client = FakeClient([GOOD_RESPONSE])
        retriever = FakeRetriever(similar=[make_similar("2026-01-15-course-choice", 0.8)])
        generator = BriefGenerator(config_with_retries, client, retriever)  # type: ignore[arg-type]

        result = generator.generate(make_decision())
        payload = result.as_dict()
        assert payload["id"] == "2026-11-01-new-decision"
        assert "validation" in payload
        assert "retries" in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
