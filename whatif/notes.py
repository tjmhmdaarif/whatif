"""Decision note parsing and validation for WhatIf.

Parses Markdown files with YAML frontmatter, validates against the schema,
and produces structured Decision objects.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import frontmatter

from whatif.config import Config


# Type aliases
Criterion = dict[str, Any]  # {name: str, weight: int}
Scores = dict[str, dict[str, int]]  # {option: {criterion: 1-5}}
Outcome = dict[str, Any]  # {reviewed_on, result, failed_assumptions}


@dataclass
class Decision:
    """Parsed and validated decision note."""

    # Required fields
    id: str
    type: str  # always "decision"
    status: str  # open | decided
    created: date
    domain: str  # study | career | money | tools | other
    options: list[str]
    chosen: str
    criteria: list[Criterion]
    scores: Scores
    assumptions: list[str]
    risks: list[str]
    outcome: Outcome
    tags: list[str]

    # Body sections
    context: str = ""
    options_body: str = ""
    decision_body: str = ""
    outcome_body: str = ""

    # Metadata
    file_path: Path | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all fields. Raises ValueError on failure."""
        errors = []

        # id format: YYYY-MM-DD-slug
        if not re.match(r"^\d{4}-\d{2}-\d{2}-.+$", self.id):
            errors.append(f"id: must match YYYY-MM-DD-slug format, got '{self.id}'")

        # type
        if self.type != "decision":
            errors.append(f"type: must be 'decision', got '{self.type}'")

        # status
        if self.status not in ("open", "decided"):
            errors.append(f"status: must be 'open' or 'decided', got '{self.status}'")

        # created is already a date from parsing

        # domain
        valid_domains = {"study", "career", "money", "tools", "other"}
        if self.domain not in valid_domains:
            errors.append(f"domain: must be one of {valid_domains}, got '{self.domain}'")

        # options
        if not isinstance(self.options, list) or len(self.options) < 2:
            errors.append(f"options: must be a list with at least 2 items, got {self.options}")
        if not all(isinstance(o, str) and o for o in self.options):
            errors.append(f"options: all items must be non-empty strings")

        # chosen
        if self.chosen and self.chosen not in self.options and self.chosen.lower() != "none":
            errors.append(f"chosen: must be one of {self.options} or empty, got '{self.chosen}'")

        # criteria
        if not isinstance(self.criteria, list) or not self.criteria:
            errors.append(f"criteria: must be a non-empty list")
        else:
            for i, c in enumerate(self.criteria):
                if not isinstance(c, dict) or "name" not in c or "weight" not in c:
                    errors.append(f"criteria[{i}]: must have 'name' and 'weight' keys")
                elif not isinstance(c["name"], str) or not c["name"]:
                    errors.append(f"criteria[{i}].name: must be non-empty string")
                elif not isinstance(c["weight"], int) or c["weight"] < 1:
                    errors.append(f"criteria[{i}].weight: must be positive integer")

        # scores
        if not isinstance(self.scores, dict):
            errors.append(f"scores: must be a dict")
        else:
            for option, crit_scores in self.scores.items():
                if option not in self.options:
                    errors.append(f"scores: option '{option}' not in options list")
                elif not isinstance(crit_scores, dict):
                    errors.append(f"scores['{option}']: must be a dict")
                else:
                    for crit_name, score in crit_scores.items():
                        if not isinstance(score, int) or score < 1 or score > 5:
                            errors.append(f"scores['{option}']['{crit_name}']: must be 1-5, got {score}")

        # assumptions
        if not isinstance(self.assumptions, list):
            errors.append(f"assumptions: must be a list")
        elif not all(isinstance(a, str) for a in self.assumptions):
            errors.append(f"assumptions: all items must be strings")

        # risks
        if not isinstance(self.risks, list):
            errors.append(f"risks: must be a list")
        elif not all(isinstance(r, str) for r in self.risks):
            errors.append(f"risks: all items must be strings")

        # outcome
        if not isinstance(self.outcome, dict):
            errors.append(f"outcome: must be a dict")
        else:
            required_outcome_keys = {"reviewed_on", "result", "failed_assumptions"}
            if not required_outcome_keys.issubset(self.outcome.keys()):
                missing = required_outcome_keys - set(self.outcome.keys())
                errors.append(f"outcome: missing keys {missing}")

            if self.outcome.get("result") not in (None, "", "better", "as-expected", "worse"):
                errors.append(f"outcome.result: must be 'better', 'as-expected', 'worse', or empty")

            if not isinstance(self.outcome.get("failed_assumptions"), list):
                errors.append(f"outcome.failed_assumptions: must be a list")

        # tags
        if not isinstance(self.tags, list):
            errors.append(f"tags: must be a list")
        elif not all(isinstance(t, str) for t in self.tags):
            errors.append(f"tags: all items must be strings")

        if errors:
            raise ValueError("; ".join(errors))

    def leading_option(self) -> str | None:
        """Return the option with highest weighted score, or None if tied/no scores."""
        if not self.scores or not self.criteria:
            return None

        criterion_weights = {c["name"]: c["weight"] for c in self.criteria}

        option_totals = {}
        for option, crit_scores in self.scores.items():
            total = 0
            for crit_name, score in crit_scores.items():
                weight = criterion_weights.get(crit_name, 0)
                total += score * weight
            option_totals[option] = total

        if not option_totals:
            return None

        max_score = max(option_totals.values())
        leaders = [opt for opt, score in option_totals.items() if score == max_score]

        if len(leaders) == 1:
            return leaders[0]
        return None  # tie

    def is_decided(self) -> bool:
        return self.status == "decided"

    def has_failed_assumptions(self) -> bool:
        return bool(self.outcome.get("failed_assumptions"))

    def criterion_weight(self, name: str) -> int:
        """Get weight for a criterion by name."""
        for c in self.criteria:
            if c["name"] == name:
                return c["weight"]
        return 0


@dataclass
class ParseResult:
    """Result of parsing a decision file."""

    decision: Decision | None = None
    error: str | None = None
    file_path: Path | None = None


def parse_date(value: Any, field_name: str, file_path: Path) -> date:
    """Parse a date from various formats."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"{file_path}: {field_name}: invalid date format '{value}'")


def parse_string_list(value: Any, field_name: str, file_path: Path) -> list[str]:
    """Parse a list of strings, validating the raw YAML type first.

    Args:
        value: Raw YAML value.
        field_name: Name of the field for error messages.
        file_path: Source file for error messages.

    Returns:
        List of strings.

    Raises:
        ValueError: If the value is not a list of strings.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{file_path}: {field_name}: must be a list, got {type(value).__name__}")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"{file_path}: {field_name}[{i}]: must be a string, got {type(item).__name__}"
            )
    return list(value)


def parse_criteria(value: Any, file_path: Path) -> list[Criterion]:
    """Parse criteria from YAML (list of {name, weight} or list of strings)."""
    if not isinstance(value, list):
        raise ValueError(f"{file_path}: criteria: must be a list")

    result = []
    for i, item in enumerate(value):
        if isinstance(item, dict):
            if "name" not in item or "weight" not in item:
                raise ValueError(f"{file_path}: criteria[{i}]: must have 'name' and 'weight'")
            result.append({"name": str(item["name"]), "weight": int(item["weight"])})
        elif isinstance(item, str):
            # Backward compatibility: string -> weight 1
            result.append({"name": item, "weight": 1})
        else:
            raise ValueError(f"{file_path}: criteria[{i}]: must be dict or string")
    return result


def parse_scores(value: Any, options: list[str], file_path: Path) -> Scores:
    """Parse scores dict."""
    if not isinstance(value, dict):
        raise ValueError(f"{file_path}: scores: must be a dict")

    result = {}
    for option, crit_scores in value.items():
        if option not in options:
            raise ValueError(f"{file_path}: scores: option '{option}' not in options")
        if not isinstance(crit_scores, dict):
            raise ValueError(f"{file_path}: scores['{option}']: must be a dict")
        result[option] = {}
        for crit_name, score in crit_scores.items():
            try:
                result[option][crit_name] = int(score)
            except (ValueError, TypeError):
                raise ValueError(f"{file_path}: scores['{option}']['{crit_name}']: must be integer")
    return result


def parse_outcome(value: Any, file_path: Path) -> Outcome:
    """Parse outcome dict with defaults."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{file_path}: outcome: must be a dict")

    result = {
        "reviewed_on": value.get("reviewed_on"),
        "result": value.get("result", ""),
        "failed_assumptions": parse_string_list(
            value.get("failed_assumptions"), "outcome.failed_assumptions", file_path
        ),
    }

    if result["reviewed_on"] and isinstance(result["reviewed_on"], str):
        try:
            result["reviewed_on"] = parse_date(result["reviewed_on"], "outcome.reviewed_on", file_path)
        except ValueError:
            pass  # Keep as string if invalid

    return result


def parse_decision_file(file_path: Path) -> ParseResult:
    """Parse a single decision Markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return ParseResult(error=f"{file_path}: Failed to read file: {e}", file_path=file_path)

    # Compute content hash
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    try:
        post = frontmatter.loads(content)
    except Exception as e:
        return ParseResult(error=f"{file_path}: Failed to parse frontmatter: {e}", file_path=file_path)

    fm = post.metadata
    body = post.content

    # Handle empty chosen field (YAML parses empty as None)
    chosen = fm.get("chosen", "")
    if chosen is None:
        chosen = ""
    elif not isinstance(chosen, str):
        chosen = str(chosen)

    # Parse body sections
    context, options_body, decision_body, outcome_body = split_body_sections(body)

    try:
        options = parse_string_list(fm.get("options"), "options", file_path)
        decision = Decision(
            id=str(fm.get("id", "")),
            type=str(fm.get("type", "")),
            status=str(fm.get("status", "open")),
            created=parse_date(fm.get("created"), "created", file_path),
            domain=str(fm.get("domain", "other")),
            options=options,
            chosen=chosen,
            criteria=parse_criteria(fm.get("criteria", []), file_path),
            scores=parse_scores(fm.get("scores", {}), options, file_path),
            assumptions=parse_string_list(fm.get("assumptions"), "assumptions", file_path),
            risks=parse_string_list(fm.get("risks"), "risks", file_path),
            outcome=parse_outcome(fm.get("outcome", {}), file_path),
            tags=parse_string_list(fm.get("tags"), "tags", file_path),
            context=context,
            options_body=options_body,
            decision_body=decision_body,
            outcome_body=outcome_body,
            file_path=file_path,
            content_hash=content_hash,
        )
        return ParseResult(decision=decision, file_path=file_path)
    except ValueError as e:
        return ParseResult(error=f"{file_path}: {e}", file_path=file_path)


def split_body_sections(body: str) -> tuple[str, str, str, str]:
    """Split body into Context, Options, Decision, Outcome sections."""
    sections = {"context": "", "options": "", "decision": "", "outcome": ""}
    current = "context"
    lines = body.split("\n")

    for line in lines:
        if line.startswith("## "):
            header = line[3:].strip().lower()
            if header in sections:
                current = header
                continue
        sections[current] += line + "\n"

    # Strip trailing whitespace from each section
    for key in sections:
        sections[key] = sections[key].rstrip()

    return sections["context"], sections["options"], sections["decision"], sections["outcome"]


def parse_all_decisions(config: Config) -> tuple[list[Decision], list[str]]:
    """Parse all decision files in the vault's Decisions directory.

    Returns:
        Tuple of (valid_decisions, errors)
    """
    decisions_dir = config.vault.decisions_path
    if not decisions_dir.exists():
        return [], [f"Decisions directory not found: {decisions_dir}"]

    md_files = list(decisions_dir.glob("*.md"))
    valid = []
    errors = []

    for file_path in md_files:
        result = parse_decision_file(file_path)
        if result.decision:
            valid.append(result.decision)
        if result.error:
            errors.append(result.error)

    return valid, errors


def filter_new_or_changed(decisions: list[Decision], config: Config) -> list[Decision]:
    """Filter decisions that are new or have changed content (based on hash).

    Note: This requires the database to be set up. For M2, we just return all.
    In M3, this will check against stored hashes.
    """
    return decisions


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    test_file = Path("vault/Templates/decision.md")
    if test_file.exists():
        result = parse_decision_file(test_file)
        if result.decision:
            print(f"Parsed: {result.decision.id}")
            print(f"  Options: {result.decision.options}")
            print(f"  Criteria: {result.decision.criteria}")
            print(f"  Leading: {result.decision.leading_option()}")
        else:
            print(f"Error: {result.error}")