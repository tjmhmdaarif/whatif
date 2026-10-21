"""Tests for notes.py - decision parsing and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from whatif.notes import (
    Decision,
    ParseResult,
    parse_decision_file,
    parse_all_decisions,
    split_body_sections,
    parse_date,
    parse_criteria,
    parse_scores,
    parse_outcome,
)
from whatif.config import Config, create_default_config


class TestParseDate:
    def test_date_object(self) -> None:
        from datetime import date
        d = date(2026, 1, 15)
        assert parse_date(d, "test", Path("test.md")) == d

    def test_datetime_object(self) -> None:
        from datetime import datetime
        dt = datetime(2026, 1, 15, 10, 30)
        assert parse_date(dt, "test", Path("test.md")) == dt.date()

    def test_iso_string(self) -> None:
        d = parse_date("2026-01-15", "test", Path("test.md"))
        assert d.year == 2026 and d.month == 1 and d.day == 15

    def test_slash_string(self) -> None:
        d = parse_date("2026/01/15", "test", Path("test.md"))
        assert d.year == 2026 and d.month == 1 and d.day == 15

    def test_european_string(self) -> None:
        d = parse_date("15-01-2026", "test", Path("test.md"))
        assert d.year == 2026 and d.month == 1 and d.day == 15

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_date("not-a-date", "test", Path("test.md"))


class TestParseCriteria:
    def test_dict_format(self) -> None:
        criteria = parse_criteria([{"name": "cost", "weight": 3}], Path("test.md"))
        assert criteria == [{"name": "cost", "weight": 3}]

    def test_string_format(self) -> None:
        criteria = parse_criteria(["cost", "quality"], Path("test.md"))
        assert criteria == [{"name": "cost", "weight": 1}, {"name": "quality", "weight": 1}]

    def test_mixed_format(self) -> None:
        criteria = parse_criteria([{"name": "cost", "weight": 3}, "quality"], Path("test.md"))
        assert criteria == [{"name": "cost", "weight": 3}, {"name": "quality", "weight": 1}]

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            parse_criteria([{"weight": 3}], Path("test.md"))

    def test_missing_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            parse_criteria([{"name": "cost"}], Path("test.md"))


class TestParseScores:
    def test_valid(self) -> None:
        scores = parse_scores(
            {"Option A": {"cost": 4, "quality": 3}, "Option B": {"cost": 2, "quality": 5}},
            ["Option A", "Option B"],
            Path("test.md"),
        )
        assert scores["Option A"]["cost"] == 4

    def test_invalid_option_raises(self) -> None:
        with pytest.raises(ValueError, match="not in options"):
            parse_scores({"Option C": {"cost": 4}}, ["Option A", "Option B"], Path("test.md"))

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            parse_scores("invalid", ["Option A"], Path("test.md"))


class TestParseOutcome:
    def test_full(self) -> None:
        outcome = parse_outcome(
            {"reviewed_on": "2026-06-15", "result": "better", "failed_assumptions": ["Assumption 1"]},
            Path("test.md"),
        )
        assert outcome["result"] == "better"
        assert outcome["failed_assumptions"] == ["Assumption 1"]

    def test_defaults(self) -> None:
        outcome = parse_outcome({}, Path("test.md"))
        assert outcome["result"] == ""
        assert outcome["failed_assumptions"] == []

    def test_invalid_result_defaults(self) -> None:
        outcome = parse_outcome({"result": "invalid"}, Path("test.md"))
        # Invalid results are kept as-is (validated later in Decision)
        assert outcome["result"] == "invalid"


class TestSplitBodySections:
    def test_all_sections(self) -> None:
        body = """## Context
This is context.

## Options
Option A details.
Option B details.

## Decision
Chose A.

## Outcome
It went well.
"""
        ctx, opts, dec, out = split_body_sections(body)
        assert "This is context" in ctx
        assert "Option A details" in opts
        assert "Chose A" in dec
        assert "It went well" in out

    def test_missing_sections(self) -> None:
        body = """## Context
Only context here.
"""
        ctx, opts, dec, out = split_body_sections(body)
        assert "Only context here" in ctx
        assert opts == ""
        assert dec == ""
        assert out == ""

    def test_unknown_section_ignored(self) -> None:
        body = """## Context
Context.

## Unknown
This is ignored.

## Options
Options.
"""
        ctx, opts, dec, out = split_body_sections(body)
        assert "Context" in ctx
        assert "Options" in opts
        assert dec == ""


class TestDecisionValidation:
    def test_valid_minimal(self, sample_decision: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.error is None
        assert result.decision.id == "2026-01-15-test-decision"
        assert result.decision.status == "decided"
        assert result.decision.options == ["Option A", "Option B"]

    def test_valid_open(self, sample_decision_open: str, temp_dir: Path) -> None:
        file_path = temp_dir / "open.md"
        file_path.write_text(sample_decision_open)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.status == "open"
        assert result.decision.chosen == ""

    def test_invalid_id_format(self, temp_dir: Path) -> None:
        content = """---
id: invalid-id
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert result.error is not None
        assert "id" in result.error

    def test_invalid_type(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: note
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "type" in result.error

    def test_invalid_status(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: maybe
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "status" in result.error

    def test_invalid_domain(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: invalid
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "domain" in result.error

    def test_options_too_few(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [OnlyOne]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "options" in result.error

    def test_chosen_not_in_options(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: decided
created: 2026-01-15
domain: study
options: [A, B]
chosen: C
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "chosen" in result.error

    def test_empty_criteria(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: []
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "criteria" in result.error

    def test_criteria_missing_keys(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "criteria[0]" in result.error

    def test_invalid_score_range(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores:
  A: {cost: 6}
  B: {cost: 3}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "scores" in result.error and "1-5" in result.error

    def test_score_for_unknown_option(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores:
  C: {cost: 3}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "scores" in result.error

    def test_invalid_assumptions_type(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: "not a list"
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "assumptions" in result.error

    def test_invalid_outcome(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome: "not a dict"
tags: []
---
## Context
"""
        file_path = temp_dir / "bad.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert "outcome" in result.error

    def test_content_hash_computed(self, sample_decision: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.content_hash
        assert len(result.decision.content_hash) == 64  # SHA256 hex

    def test_file_path_stored(self, sample_decision: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.file_path == file_path


class TestLeadingOption:
    def test_clear_winner(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria:
  - {name: cost, weight: 3}
  - {name: quality, weight: 2}
scores:
  A: {cost: 5, quality: 5}
  B: {cost: 1, quality: 1}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "test.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.leading_option() == "A"

    def test_tie_returns_none(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria:
  - {name: cost, weight: 1}
scores:
  A: {cost: 3}
  B: {cost: 3}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "test.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.leading_option() is None

    def test_no_scores_returns_none(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria:
  - {name: cost, weight: 1}
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "test.md"
        file_path.write_text(content)
        result = parse_decision_file(file_path)
        assert result.decision is not None
        assert result.decision.leading_option() is None


class TestParseAllDecisions:
    def test_multiple_files(self, temp_dir: Path) -> None:
        # Create a vault structure
        decisions_dir = temp_dir / "Decisions"
        decisions_dir.mkdir()

        decision1 = """---
id: 2026-01-15-first
type: decision
status: decided
created: 2026-01-15
domain: study
options: [A, B]
chosen: A
criteria: [{name: cost, weight: 1}]
scores: {A: {cost: 4}, B: {cost: 2}}
assumptions: []
risks: []
outcome:
  reviewed_on: 2026-06-15
  result: better
  failed_assumptions: []
tags: []
---
## Context
"""
        decision2 = """---
id: 2026-02-20-second
type: decision
status: open
created: 2026-02-20
domain: tools
options: [X, Y]
chosen:
criteria: [{name: price, weight: 2}]
scores: {X: {price: 3}, Y: {price: 5}}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        (decisions_dir / "first.md").write_text(decision1)
        (decisions_dir / "second.md").write_text(decision2)

        config = create_default_config(temp_dir)
        valid, errors = parse_all_decisions(config)

        assert len(valid) == 2
        assert len(errors) == 0
        assert valid[0].id == "2026-01-15-first"
        assert valid[1].id == "2026-02-20-second"

    def test_skips_invalid_files(self, temp_dir: Path) -> None:
        decisions_dir = temp_dir / "Decisions"
        decisions_dir.mkdir()

        (decisions_dir / "good.md").write_text("""---
id: 2026-01-15-good
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
""")

        (decisions_dir / "bad.md").write_text("""---
id: bad-id
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
""")

        config = create_default_config(temp_dir)
        valid, errors = parse_all_decisions(config)

        assert len(valid) == 1
        assert len(errors) == 1
        assert "bad-id" in errors[0] or "id" in errors[0]

    def test_missing_decisions_dir(self, temp_dir: Path) -> None:
        config = create_default_config(temp_dir)
        # Remove the Decisions dir that create_default_config made
        import shutil

        shutil.rmtree(config.vault.decisions_path)
        valid, errors = parse_all_decisions(config)
        assert len(valid) == 0
        assert len(errors) == 1
        assert "not found" in errors[0]


class TestDecisionHelpers:
    def test_is_decided(self, sample_decision: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision)
        result = parse_decision_file(file_path)
        assert result.decision.is_decided() is True

    def test_is_not_decided(self, sample_decision_open: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision_open)
        result = parse_decision_file(file_path)
        assert result.decision.is_decided() is False

    def test_has_failed_assumptions(self, sample_decision: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision)
        result = parse_decision_file(file_path)
        assert result.decision.has_failed_assumptions() is True

    def test_no_failed_assumptions(self, sample_decision_open: str, temp_dir: Path) -> None:
        file_path = temp_dir / "test.md"
        file_path.write_text(sample_decision_open)
        result = parse_decision_file(file_path)
        assert result.decision.has_failed_assumptions() is False


class TestSeedFixtures:
    """Validate the 10 seed decision notes parse and satisfy the schema."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "decisions"

    def test_ten_seed_notes_exist(self, fixtures_dir: Path) -> None:
        files = sorted(fixtures_dir.glob("*.md"))
        assert len(files) == 10, f"Expected 10 seed notes, found {len(files)}"

    def test_all_seed_notes_parse(self, fixtures_dir: Path) -> None:
        errors = []
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            if result.error:
                errors.append(result.error)
        assert not errors, "Seed notes failed to parse:\n" + "\n".join(errors)

    def test_seed_notes_have_unique_ids(self, fixtures_dir: Path) -> None:
        ids = []
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            ids.append(result.decision.id)
        assert len(ids) == len(set(ids))

    def test_seed_outcomes_filled(self, fixtures_dir: Path) -> None:
        """At least 4-5 notes must have a filled-in outcome."""
        filled = 0
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            if result.decision.outcome.get("reviewed_on"):
                filled += 1
        assert filled >= 4, f"Only {filled} seed notes have a reviewed_on outcome"

    def test_seed_failed_assumptions_count(self, fixtures_dir: Path) -> None:
        """At least 3 notes must have a non-empty failed_assumptions list."""
        with_failures = 0
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            if result.decision.has_failed_assumptions():
                with_failures += 1
        assert with_failures >= 3, f"Only {with_failures} seed notes have failed assumptions"

    def test_seed_failed_assumptions_are_subset_of_assumptions(self, fixtures_dir: Path) -> None:
        """Failed assumptions must be drawn from the note's own assumptions list."""
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            declared = set(result.decision.assumptions)
            for failed in result.decision.outcome.get("failed_assumptions", []):
                assert failed in declared, (
                    f"{result.decision.id}: failed assumption {failed!r} "
                    f"is not in the declared assumptions list"
                )

    def test_seed_decided_notes_have_outcome_result(self, fixtures_dir: Path) -> None:
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            d = result.decision
            if d.status == "decided":
                assert d.outcome.get("result") in ("better", "as-expected", "worse"), (
                    f"{d.id}: decided note must have a valid outcome.result"
                )
                assert d.outcome.get("reviewed_on"), f"{d.id}: decided note needs reviewed_on"

    def test_seed_open_notes_have_no_chosen(self, fixtures_dir: Path) -> None:
        for file_path in sorted(fixtures_dir.glob("*.md")):
            result = parse_decision_file(file_path)
            assert result.decision is not None, result.error
            d = result.decision
            if d.status == "open":
                assert d.chosen == "", f"{d.id}: open note must not have a chosen option"


class TestEdgeCases:
    """Edge cases called out in the spec: unicode filenames, malformed YAML, bad encoding."""

    def test_unicode_filename(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-unicode-test
type: decision
status: open
created: 2026-01-15
domain: study
options: [Option A, Option B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
Café naïve résumé — 日本語 emoji 🎯
"""
        file_path = temp_dir / "2026-01-15-ünïcodé-決定.md"
        file_path.write_text(content, encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.decision is not None, result.error
        assert result.decision.id == "2026-01-15-unicode-test"
        assert "日本語" in result.decision.context

    def test_unicode_content_in_fields(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-unicode-fields
type: decision
status: open
created: 2026-01-15
domain: other
options: [Café, Übung]
chosen:
criteria: [{name: coût, weight: 1}]
scores: {}
assumptions: [Le prix baisse]
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: [ünïcodé]
---
## Context
"""
        file_path = temp_dir / "unicode_fields.md"
        file_path.write_text(content, encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.decision is not None, result.error
        assert result.decision.options == ["Café", "Übung"]
        assert result.decision.assumptions == ["Le prix baisse"]
        assert result.decision.tags == ["ünïcodé"]

    def test_malformed_yaml_returns_error_not_crash(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-broken
type: decision
status: open
  created: [unclosed
---
## Context
"""
        file_path = temp_dir / "broken.md"
        file_path.write_text(content, encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert result.error is not None
        assert "broken.md" in result.error

    def test_missing_frontmatter_returns_error(self, temp_dir: Path) -> None:
        file_path = temp_dir / "nofm.md"
        file_path.write_text("Just some text with no frontmatter at all.\n")
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert result.error is not None

    def test_empty_file_returns_error(self, temp_dir: Path) -> None:
        file_path = temp_dir / "empty.md"
        file_path.write_text("", encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert result.error is not None

    def test_missing_file_returns_error(self, temp_dir: Path) -> None:
        result = parse_decision_file(temp_dir / "does_not_exist.md")
        assert result.decision is None
        assert result.error is not None
        assert "does_not_exist.md" in result.error

    def test_missing_created_field_reports_field(self, temp_dir: Path) -> None:
        content = """---
id: 2026-01-15-no-created
type: decision
status: open
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "nocreated.md"
        file_path.write_text(content, encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.decision is None
        assert result.error is not None
        assert "created" in result.error

    def test_error_message_includes_file_path(self, temp_dir: Path) -> None:
        content = """---
id: bad
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        file_path = temp_dir / "pathcheck.md"
        file_path.write_text(content, encoding="utf-8")
        result = parse_decision_file(file_path)
        assert result.error is not None
        assert "pathcheck.md" in result.error

    def test_one_bad_note_does_not_stop_others(self, temp_dir: Path) -> None:
        decisions_dir = temp_dir / "Decisions"
        decisions_dir.mkdir()

        good = """---
id: 2026-01-15-good-one
type: decision
status: open
created: 2026-01-15
domain: study
options: [A, B]
chosen:
criteria: [{name: cost, weight: 1}]
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context
"""
        (decisions_dir / "good.md").write_text(good, encoding="utf-8")
        (decisions_dir / "bad.md").write_text("---\nnot: [valid\n---\n", encoding="utf-8")
        (decisions_dir / "alsogood.md").write_text(good.replace("good-one", "good-two"), encoding="utf-8")

        from whatif.config import create_default_config

        config = create_default_config(temp_dir)
        valid, errors = parse_all_decisions(config)
        assert len(valid) == 2
        assert len(errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])