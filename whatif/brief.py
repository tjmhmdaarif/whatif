"""Brief generation with code-side validation of every claim.

The LLM writes prose. Code decides whether the prose is admissible:

* every claim must end with ``[EVIDENCE: note-id]`` naming a note that was actually
  retrieved, or with ``[ESTIMATE]``;
* no claim may contain a probability, a percentage, or a promise that something is
  certain;
* the counterargument must argue *against* the option the decision maker leans
  toward.

If a claim breaks a rule, the brief is generated once more with the specific
complaints fed back to the model. If the second attempt also breaks them, the
offending claims are marked ``[UNVERIFIED]`` rather than silently trusted.

The weighted comparison table is never written by the model. It is computed in
:mod:`whatif.scoring` and spliced in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from whatif.config import Config
from whatif.embeddings import OllamaClient
from whatif.notes import Decision
from whatif.retrieve import FailedAssumption, Retriever, SimilarDecision
from whatif.scoring import (
    ScoringResult,
    compute_scores,
    format_comparison_table,
    leading_option,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

EVIDENCE_TAG = "EVIDENCE"
ESTIMATE_TAG = "ESTIMATE"
UNVERIFIED_TAG = "UNVERIFIED"

_TAG_RE = re.compile(
    r"\[(?P<kind>EVIDENCE|ESTIMATE|UNVERIFIED)(?::\s*(?P<value>[^\]]*))?\]"
)

DEFAULT_LEGEND = """> **Legend**
>
> - `[EVIDENCE: note-id]` - a claim supported by a past decision that was retrieved
> - `[ESTIMATE]` - a judgement with no supporting evidence in your notes
> - `[UNVERIFIED]` - a claim that failed validation and could not be confirmed
> - `COMPUTED` - a value calculated in code from your own scores and weights
>
> WhatIf does not predict outcomes. Nothing here is a probability."""


@dataclass(frozen=True)
class Tag:
    """A single tag found in a claim."""

    kind: str
    value: str = ""
    start: int = 0
    end: int = 0

    @property
    def is_evidence(self) -> bool:
        return self.kind == EVIDENCE_TAG

    @property
    def is_estimate(self) -> bool:
        return self.kind == ESTIMATE_TAG


def find_tags(text: str) -> list[Tag]:
    """Return every tag in a piece of text, in order."""
    return [
        Tag(
            kind=match.group("kind"),
            value=(match.group("value") or "").strip(),
            start=match.start(),
            end=match.end(),
        )
        for match in _TAG_RE.finditer(text)
    ]


def ends_with_tag(text: str) -> bool:
    """Return True if the text ends with a tag, ignoring trailing punctuation."""
    cleaned = text.rstrip()
    cleaned = re.sub(r"[\s.,;:!?'\"]+$", "", cleaned)
    return bool(re.search(r"\[(?:EVIDENCE|ESTIMATE|UNVERIFIED)(?::[^\]]*)?\]$", cleaned))


def strip_tags(text: str) -> str:
    """Remove every tag from a piece of text."""
    return _TAG_RE.sub("", text).rstrip()


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------

# Numeric probabilities and percentages.
_PERCENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+of\s+the\s+time\b", re.I), "some of the time"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+of\b", re.I), "a proportion of"),
    (
        re.compile(
            r"\b(?:an?|the)\s+\d+(?:\.\d+)?\s*%\s+(?:chance|likelihood|odds)\b", re.I
        ),
        "a real chance",
    ),
    (
        re.compile(r"\b(?:an?|the)\s+\d+(?:\.\d+)?\s*%\s+probability\b", re.I),
        "a real possibility",
    ),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+(?:chance|likelihood|odds)\b", re.I), "a real chance"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%\s+probability\b", re.I), "a real possibility"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*%"), "a meaningful amount"),
]

# Words that turn a claim into a probability or a promise.
_PROBABILITY_WORDS = re.compile(
    r"\b(probabilit(?:y|ies)|percent(?:age)?\s+(?:chance|likelihood|probability))\b", re.I
)

_CERTAINTY_WORDS = re.compile(
    r"\b(guaranteed|guarantees|guarantee|certainly|is certain to|"
    r"will definitely|definitely will|assured)\b",
    re.I,
)


def find_probability(text: str) -> str | None:
    """Return the offending probability language in a claim, or None."""
    for pattern, _ in _PERCENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    match = _PROBABILITY_WORDS.search(text)
    if match:
        return match.group(0)
    return None


def find_certainty(text: str) -> str | None:
    """Return the offending certainty language in a claim, or None."""
    match = _CERTAINTY_WORDS.search(text)
    return match.group(0) if match else None


def neutralize_probabilities(text: str) -> str:
    """Rewrite probability language so the brief never states one.

    Used only on the final attempt, after the model has already failed to comply.
    """
    result = text
    for pattern, replacement in _PERCENT_PATTERNS:
        result = pattern.sub(replacement, result)
    result = _PROBABILITY_WORDS.sub(
        lambda m: "possibility" if m.group(0).lower().startswith("probabilit") else m.group(0),
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

# A sentence, optionally carrying a trailing tag. The tag has to stay with its
# sentence, otherwise "What is the budget? [ESTIMATE]" splits into two claims and
# the tag ends up orphaned.
_TAG_SUFFIX = r"(?:\s*\[(?:EVIDENCE|ESTIMATE|UNVERIFIED)(?::[^\]]*)?\])?"
_SENTENCE_RE = re.compile(rf"[^.!?\n]*[.!?]+{_TAG_SUFFIX}|[^.!?\n]+$")
_BULLET_RE = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")
_MARKER_LINE_RE = re.compile(r"^</?[a-z_][a-z_]*>$", re.IGNORECASE)


@dataclass
class Claim:
    """A single claim inside a piece of prose."""

    text: str
    line_index: int = 0
    start: int = 0
    end: int = 0
    is_bullet: bool = False

    def __str__(self) -> str:
        return self.text


def _is_verbatim_line(stripped: str) -> bool:
    """Return True for lines that carry no claims (headings, tables, rules)."""
    if not stripped:
        return True
    if stripped.startswith(("#", "|", "---", "***", "___", "```", "===")):
        return True
    # Section markers such as <decision_summary> are structure, not claims.
    if _MARKER_LINE_RE.match(stripped):
        return True
    return False


_TAG_ONLY_RE = re.compile(
    r"^\[(?:EVIDENCE|ESTIMATE|UNVERIFIED)(?::[^\]]*)?\][\s.,;:!?]*$", re.IGNORECASE
)


def merge_lone_tag_lines(text: str) -> str:
    """Join a line holding only a tag onto the line before it.

    Models often wrap a tag onto its own line. Left alone, the tag would be
    split off from the claim it belongs to and the claim would look untagged.
    """
    merged: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if merged and stripped and _TAG_ONLY_RE.match(stripped):
            merged[-1] = f"{merged[-1].rstrip()} {stripped}"
        else:
            merged.append(line)
    return "\n".join(merged)


def split_claims(text: str) -> list[Claim]:
    """Split prose into individual claims.

    A claim is one sentence. Bullets are split into sentences too, so a bullet that
    packs two claims into one line does not get a free pass.

    Headings, table rows, horizontal rules and code fences carry no claims.
    """
    claims: list[Claim] = []
    in_code_fence = False

    for line_index, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or _is_verbatim_line(stripped):
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            marker, body = bullet.group(1), bullet.group(2)
            base = len(marker)
        else:
            base = 0
            body = line

        offset = 0
        for match in _SENTENCE_RE.finditer(body):
            sentence = match.group(0)
            if len(sentence.strip()) < 3:
                offset = match.end()
                continue
            start = base + match.start()
            end = base + match.end()
            claims.append(
                Claim(
                    text=sentence.strip(),
                    line_index=line_index,
                    start=start,
                    end=end,
                    is_bullet=bullet is not None,
                )
            )
            offset = match.end()

    return claims


# The similar-notes section holds "note-id: why relevant" lines. The note id *is*
# the evidence there, so those lines are not claims and are not tag-validated.
# They are blanked out before validation, keeping the line count identical so that
# issue line numbers still line up with the original text.
_SIMILAR_BLOCK_RE = re.compile(
    r"<similar_notes>.*?</similar_notes>", re.IGNORECASE | re.DOTALL
)


def blank_similar_block(text: str) -> str:
    """Replace the similar-notes block with blank lines of the same height."""
    return _SIMILAR_BLOCK_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REASON_MISSING_TAG = "missing_tag"
REASON_UNKNOWN_NOTE = "unknown_note"
REASON_MISPLACED_TAG = "misplaced_tag"
REASON_PROBABILITY = "probability"
REASON_CERTAINTY = "certainty"


@dataclass
class ValidationIssue:
    """One rule violation found in a claim."""

    reason: str
    claim: str
    detail: str = ""
    line_index: int = 0

    def describe(self) -> str:
        """Return a human readable description used in the retry prompt."""
        if self.reason == REASON_MISSING_TAG:
            return f"This claim has no tag: {self.claim!r}"
        if self.reason == REASON_UNKNOWN_NOTE:
            return (
                f"This claim cites a note that was not retrieved: {self.claim!r}. "
                f"({self.detail})"
            )
        if self.reason == REASON_MISPLACED_TAG:
            return f"The tag must be the last thing in the claim: {self.claim!r}"
        if self.reason == REASON_PROBABILITY:
            return (
                f"This claim states a probability or percentage, which WhatIf does not "
                f"use: {self.claim!r} (found {self.detail!r})"
            )
        if self.reason == REASON_CERTAINTY:
            return (
                f"This claim promises a certain outcome, which WhatIf does not do: "
                f"{self.claim!r} (found {self.detail!r})"
            )
        return f"{self.reason}: {self.claim!r}"


@dataclass
class ValidationReport:
    """The outcome of validating one piece of generated prose."""

    issues: list[ValidationIssue] = field(default_factory=list)
    claim_count: int = 0

    @property
    def passed(self) -> bool:
        """True when no rule was broken."""
        return not self.issues

    @property
    def unverified_count(self) -> int:
        """Number of claims that will be marked unverified."""
        return len(self.issues)

    def by_reason(self) -> dict[str, int]:
        """Count the issues per reason."""
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.reason] = counts.get(issue.reason, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "claim_count": self.claim_count,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "reason": issue.reason,
                    "claim": issue.claim,
                    "detail": issue.detail,
                    "line_index": issue.line_index,
                }
                for issue in self.issues
            ],
        }


def validate_claims(text: str, retrieved_ids: Iterable[str]) -> ValidationReport:
    """Check every claim in a piece of prose against the brief's rules.

    Args:
        text: The generated prose.
        retrieved_ids: Note ids the model was allowed to cite.

    Returns:
        A report listing every violation.
    """
    allowed = set(retrieved_ids)
    claims = split_claims(text)
    issues: list[ValidationIssue] = []

    for claim in claims:
        tags = find_tags(claim.text)

        if not tags:
            issues.append(
                ValidationIssue(
                    reason=REASON_MISSING_TAG,
                    claim=claim.text,
                    detail="no [EVIDENCE: note-id] or [ESTIMATE] tag",
                    line_index=claim.line_index,
                )
            )
            continue

        evidence = [tag for tag in tags if tag.is_evidence]
        for tag in evidence:
            if tag.value not in allowed:
                issues.append(
                    ValidationIssue(
                        reason=REASON_UNKNOWN_NOTE,
                        claim=claim.text,
                        detail=f"cited note '{tag.value}' was not retrieved",
                        line_index=claim.line_index,
                    )
                )

        if not any(tag.is_estimate for tag in tags) and not evidence:
            issues.append(
                ValidationIssue(
                    reason=REASON_MISSING_TAG,
                    claim=claim.text,
                    detail="tag was not an EVIDENCE or ESTIMATE tag",
                    line_index=claim.line_index,
                )
            )

        if not ends_with_tag(claim.text):
            issues.append(
                ValidationIssue(
                    reason=REASON_MISPLACED_TAG,
                    claim=claim.text,
                    detail="tag is not at the end of the claim",
                    line_index=claim.line_index,
                )
            )

        probability = find_probability(claim.text)
        if probability:
            issues.append(
                ValidationIssue(
                    reason=REASON_PROBABILITY,
                    claim=claim.text,
                    detail=probability,
                    line_index=claim.line_index,
                )
            )

        certainty = find_certainty(claim.text)
        if certainty:
            issues.append(
                ValidationIssue(
                    reason=REASON_CERTAINTY,
                    claim=claim.text,
                    detail=certainty,
                    line_index=claim.line_index,
                )
            )

    return ValidationReport(issues=issues, claim_count=len(claims))


def mark_unverified(text: str, report: ValidationReport) -> str:
    """Mark every failing claim as ``[UNVERIFIED]``.

    An evidence tag pointing at a note that was never retrieved is replaced rather
    than kept, because the citation is not merely unconfirmed, it is wrong. A claim
    that states a probability has the number removed as well.
    """
    if not report.issues:
        return text

    reasons_by_line: dict[int, set[str]] = {}
    unknown_claims: set[str] = set()
    for issue in report.issues:
        reasons_by_line.setdefault(issue.line_index, set()).add(issue.reason)
        if issue.reason == REASON_UNKNOWN_NOTE:
            unknown_claims.add(issue.claim)

    lines = text.split("\n")
    for line_index, reasons in reasons_by_line.items():
        if line_index >= len(lines):
            continue
        line = lines[line_index]
        claims = split_claims(line)

        rebuilt: list[str] = []
        cursor = 0
        for claim in sorted(claims, key=lambda c: c.start):
            rebuilt.append(line[cursor : claim.start])
            rebuilt.append(
                _fix_claim(claim.text, reasons, claim.text in unknown_claims)
            )
            cursor = claim.end
        rebuilt.append(line[cursor:])
        lines[line_index] = "".join(rebuilt)

    return "\n".join(lines)


def _fix_claim(claim_text: str, reasons: set[str], has_unknown_id: bool) -> str:
    """Repair one claim so it no longer breaks the rules."""
    fixed = neutralize_probabilities(claim_text)

    if has_unknown_id:
        # The citation is wrong, not merely unconfirmed, so it is replaced outright.
        fixed = re.sub(r"\[EVIDENCE:\s*[^\]]*\]", f"[{UNVERIFIED_TAG}]", fixed)
        return fixed

    if reasons & {REASON_PROBABILITY, REASON_CERTAINTY}:
        # The claim is not trustworthy, so it says so even if it already had a tag.
        fixed = f"{fixed.rstrip()} [{UNVERIFIED_TAG}]"
    elif not ends_with_tag(fixed):
        fixed = f"{fixed.rstrip()} [{UNVERIFIED_TAG}]"

    return fixed


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

SECTION_MARKERS = {
    "summary": ("decision_summary",),
    "similar": ("similar_notes",),
    "counterargument": ("counterargument",),
    "questions": ("open_questions",),
}


def parse_sections(text: str) -> dict[str, str]:
    """Split the model's response into named sections.

    Looks for ``<decision_summary>`` style markers first, then falls back to
    ``## Heading`` sections.
    """
    sections: dict[str, str] = {}

    for name, (marker,) in SECTION_MARKERS.items():
        pattern = re.compile(
            rf"<{marker}>(.*?)</{marker}>", re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(text)
        if match:
            sections[name] = match.group(1).strip()

    if len(sections) == len(SECTION_MARKERS):
        return sections

    # Fall back to markdown headings.
    by_heading: dict[str, str] = {}
    current = None
    buffer: list[str] = []
    for line in text.split("\n"):
        heading = re.match(r"^#{1,4}\s+(.*)$", line.strip())
        if heading:
            if current is not None:
                by_heading[current] = "\n".join(buffer).strip()
            current = heading.group(1).strip().lower()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        by_heading[current] = "\n".join(buffer).strip()

    heading_map = {
        "summary": ("decision summary", "summary"),
        "similar": ("similar past decisions", "similar decisions", "similar notes"),
        "counterargument": ("counterargument", "counter argument", "against the leading option"),
        "questions": ("open questions", "questions"),
    }
    for name, keys in heading_map.items():
        if name in sections:
            continue
        for key in keys:
            if key in by_heading:
                sections[name] = by_heading[key]
                break

    for name in SECTION_MARKERS:
        sections.setdefault(name, "")

    return sections


def parse_relevance_lines(text: str, allowed_ids: Iterable[str]) -> dict[str, str]:
    """Parse ``note-id: why it is relevant`` lines from the model.

    Lines that do not name an allowed note are ignored.
    """
    allowed = set(allowed_ids)
    result: dict[str, str] = {}

    for raw in text.split("\n"):
        line = raw.strip().lstrip("-*+ ").strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*[:\-]\s*(.+)$", line)
        if not match:
            continue
        note_id, description = match.group(1), match.group(2).strip()
        if note_id in allowed and description:
            result.setdefault(note_id, description)

    return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an evidence reviewer for a personal decision journal.

You write about decisions the user has made before. You are given one new decision
and a set of past decisions that were retrieved as relevant.

Hard rules:
1. Every claim you write must end with exactly one tag on the same line:
   - [EVIDENCE: note-id] if the claim is supported by one of the retrieved notes
   - [ESTIMATE] if it is your own judgement with no supporting note
   Use only the note ids you were given. Never invent a note id.
2. One claim per sentence. Do not put two claims in one bullet.
3. Never state a probability, a percentage, or a chance. No "80%", no "likely",
   no "most people". Describe what happened, not how likely something is.
4. Never promise that something is certain. No "guaranteed", no "will definitely".
5. Do not agree with the user by default. The counterargument section must make the
   strongest case against the option the user is leaning toward, using the retrieved
   notes. If the evidence is mixed, say so.
6. Do not invent facts about the user's situation. You only know what is in the notes.
7. Return only the four marked sections, with no commentary outside them."""


def _format_decision_for_prompt(decision: Decision) -> str:
    """Render the new decision note for the prompt."""
    lines = [
        f"id: {decision.id}",
        f"domain: {decision.domain}",
        f"status: {decision.status}",
        f"options: {', '.join(decision.options)}",
    ]
    if decision.chosen:
        lines.append(f"recorded choice: {decision.chosen}")
    lines.append("")
    lines.append("Context:")
    lines.append(decision.context.strip() or "(none written)")

    if decision.options_body:
        lines.append("")
        lines.append("Options in detail:")
        lines.append(decision.options_body.strip())

    if decision.criteria:
        lines.append("")
        lines.append("The user's criteria and weights (for context only, do not restate):")
        for criterion in decision.criteria:
            lines.append(f"  - {criterion.get('name')} (weight {criterion.get('weight')})")

    if decision.assumptions:
        lines.append("")
        lines.append("The user's stated assumptions:")
        for assumption in decision.assumptions:
            lines.append(f"  - {assumption}")

    if decision.risks:
        lines.append("")
        lines.append("The user's stated risks:")
        for risk in decision.risks:
            lines.append(f"  - {risk}")

    return "\n".join(lines)


def _format_retrieved_note(note: SimilarDecision, body: str) -> str:
    """Render one retrieved past decision for the prompt."""
    lines = [
        f"note-id: {note.note_id}",
        f"domain: {note.domain}",
        f"created: {note.created}",
        f"status: {note.status}",
    ]
    if note.options:
        lines.append(f"options: {', '.join(note.options)}")
    if note.chosen:
        lines.append(f"recorded choice: {note.chosen}")
    if body:
        lines.append("body:")
        lines.append(body.strip())
    if note.failed_assumptions:
        lines.append("assumptions that turned out to be wrong in this note:")
        for assumption in note.failed_assumptions:
            lines.append(f"  - {assumption}")
    return "\n".join(lines)


def build_prompt(
    decision: Decision,
    similar: list[SimilarDecision],
    failed: list[FailedAssumption],
    note_bodies: dict[str, str],
    leading: str | None,
) -> str:
    """Build the prompt for the model.

    The model sees the new decision note and the retrieved notes. Nothing else.
    """
    retrieved_ids = [note.note_id for note in similar]
    failed_ids = sorted({item.source_note_id for item in failed})

    parts: list[str] = []
    parts.append("=== NEW DECISION ===")
    parts.append(_format_decision_for_prompt(decision))

    parts.append("")
    parts.append("=== RETRIEVED PAST DECISIONS (you may cite these note ids) ===")
    if similar:
        for note in similar:
            parts.append("")
            parts.append(_format_retrieved_note(note, note_bodies.get(note.note_id, "")))
    else:
        parts.append("(none were retrieved)")

    parts.append("")
    parts.append("=== ASSUMPTIONS THAT FAILED IN PAST DECISIONS (for warning only) ===")
    if failed:
        for item in failed:
            parts.append(f"  - {item.text}   [from note {item.source_note_id}]")
    else:
        parts.append("(none were found)")

    parts.append("")
    parts.append("=== NOTE IDS YOU MAY CITE ===")
    parts.append(", ".join(retrieved_ids) if retrieved_ids else "(none)")
    if failed_ids:
        parts.append("")
        parts.append(
            "These note ids are only context for failed assumptions, not citable "
            "evidence unless they also appear in the list above: "
            + ", ".join(failed_ids)
        )

    parts.append("")
    parts.append("=== THE OPTION THE USER IS LEANING TOWARD ===")
    parts.append(leading or "(not clear from the note)")

    parts.append("")
    parts.append("=== WRITE THESE FOUR SECTIONS ===")
    parts.append("")
    parts.append("Rules that will be checked on your answer:")
    parts.append(
        "- Every sentence must end with [EVIDENCE: note-id] using one of the note ids "
        "above, or with [ESTIMATE]."
    )
    parts.append(
        "- Never state a probability, a percentage or a chance. No '80%', no 'likely', "
        "no 'most students'. Never say something is guaranteed or certain."
    )
    parts.append(
        "- The counterargument section must argue against the option the user is "
        "leaning toward, not agree with them."
    )
    parts.append("")
    parts.append("<decision_summary>")
    parts.append(
        "Two to four sentences describing the decision in front of the user, the "
        "trade-off they are trying to make, and what is actually at stake. Base it "
        "on the new decision note. Tag each sentence."
    )
    parts.append("</decision_summary>")
    parts.append("")
    parts.append("<similar_notes>")
    parts.append(
        "One line per retrieved note, in the form 'note-id: why this past decision "
        "is relevant to the new one'. Do not include any other text. Do not add a "
        "tag to these lines; the note id is the evidence."
    )
    parts.append("</similar_notes>")
    parts.append("")
    parts.append("<counterargument>")
    parts.append(
        "The strongest case against the option the user is leaning toward. Use the "
        "retrieved notes, especially any past decision where an assumption like this "
        "one turned out to be wrong. If the past notes do not support a strong "
        "objection, say that plainly and state what the user should watch for. "
        "Every sentence must be tagged."
    )
    parts.append("</counterargument>")
    parts.append("")
    parts.append("<open_questions>")
    parts.append(
        "Three to five questions the user should answer before deciding. These are "
        "prompts, not claims, so tag each one [ESTIMATE]."
    )
    parts.append("</open_questions>")

    return "\n".join(parts)


def build_retry_prompt(
    original_prompt: str, report: ValidationReport, allowed_ids: Iterable[str]
) -> str:
    """Build the repair prompt fed back to the model after a failed attempt."""
    allowed = ", ".join(sorted(allowed_ids)) or "(none)"
    complaints = "\n".join(f"- {issue.describe()}" for issue in report.issues[:20])
    extra = ""
    if len(report.issues) > 20:
        extra = f"\n- ...and {len(report.issues) - 20} more."

    return f"""{original_prompt}

=== YOUR PREVIOUS ANSWER WAS REJECTED ===

{complaints}{extra}

Rules you broke, in short:
- Every sentence must end with exactly one tag: [EVIDENCE: one-of-{allowed}] or [ESTIMATE].
- Cite only these note ids: {allowed}
- No percentages, no probabilities, no chances.
- Nothing is guaranteed or certain.

Rewrite all four sections from scratch and follow the rules this time."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_similar_section(
    similar: list[SimilarDecision], relevance: dict[str, str]
) -> str:
    """Render the similar past decisions section.

    The ids and similarity scores come from the index, never from the model.
    """
    if not similar:
        return "_No past decisions were similar enough to surface._"

    lines = [
        "| Note | Similarity | Why it is relevant |",
        "|---|---|---|",
    ]
    for note in similar:
        description = relevance.get(note.note_id) or note.summary
        description = neutralize_probabilities(description).replace("|", "/").strip()
        lines.append(
            f"| `{note.note_id}` | {note.similarity:.3f} | {description} |"
        )
    return "\n".join(lines)


def _format_failed_section(failed: list[FailedAssumption]) -> str:
    """Render the assumptions that failed before."""
    if not failed:
        return "_No past assumption failures were close enough to this decision to surface._"

    lines = []
    for item in failed:
        text = item.text.replace("|", "/").strip()
        lines.append(
            f"- {text} [EVIDENCE: {item.source_note_id}] (similarity {item.similarity:.3f})"
        )
    return "\n".join(lines)


def render_brief(
    decision: Decision,
    similar: list[SimilarDecision],
    failed: list[FailedAssumption],
    scoring: ScoringResult,
    sections: dict[str, str],
    relevance: dict[str, str],
    leading: str | None,
    legend: str = DEFAULT_LEGEND,
    validation: ValidationReport | None = None,
    retries: int = 0,
) -> str:
    """Assemble the final brief Markdown.

    The prose comes from the model. Every number, every note id and the weighted
    table come from code.
    """
    parts: list[str] = []

    parts.append(f"# Decision brief: {decision.id}")
    parts.append("")
    parts.append(legend)
    parts.append("")
    parts.append(
        f"*{decision.domain} decision | created {decision.created} | "
        f"{len(similar)} past decisions retrieved | "
        f"{len(failed)} past assumption failures surfaced*"
    )
    parts.append("")

    parts.append("## Decision summary")
    parts.append("")
    parts.append(sections.get("summary", "").strip() or "_Not generated._")
    parts.append("")

    parts.append("## Similar past decisions")
    parts.append("")
    parts.append(_format_similar_section(similar, relevance))
    parts.append("")

    parts.append("## Assumptions that failed before")
    parts.append("")
    parts.append(_format_failed_section(failed))
    parts.append("")

    parts.append("## Weighted comparison")
    parts.append("")
    parts.append(format_comparison_table(scoring))
    parts.append("")

    counterargument_title = (
        f"## Counterargument: against {leading}" if leading else "## Counterargument"
    )
    parts.append(counterargument_title)
    parts.append("")
    parts.append(
        sections.get("counterargument", "").strip() or "_Not generated._"
    )
    parts.append("")

    parts.append("## Open questions")
    parts.append("")
    parts.append(sections.get("questions", "").strip() or "_Not generated._")
    parts.append("")

    if validation is not None and not validation.passed:
        counts = validation.by_reason()
        detail = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
        parts.append("---")
        parts.append("")
        parts.append(
            f"*{len(validation.issues)} claim(s) did not pass validation and are marked "
            f"[{UNVERIFIED_TAG}] ({detail}). The model was asked to fix them once and "
            f"did not.*"
        )
        parts.append("")

    if retries:
        parts.append(
            f"*This brief was generated {retries + 1} time(s); the first attempt "
            f"failed validation.*"
        )
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class BriefResult:
    """The outcome of generating one brief."""

    decision: Decision
    content: str
    similar: list[SimilarDecision] = field(default_factory=list)
    failed_assumptions: list[FailedAssumption] = field(default_factory=list)
    scoring: ScoringResult | None = None
    leading: str | None = None
    validation: ValidationReport | None = None
    retries: int = 0
    sections: dict[str, str] = field(default_factory=dict)
    relevance: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.decision.id,
            "similar": [note.to_dict() for note in self.similar],
            "failed_assumptions": [item.to_dict() for item in self.failed_assumptions],
            "leading": self.leading,
            "validation": self.validation.as_dict() if self.validation else None,
            "retries": self.retries,
        }


class BriefGenerator:
    """Generates decision briefs.

    Args:
        config: Active configuration.
        client: Ollama client used for generation.
        retriever: Retriever used to find similar notes and failed assumptions.
        note_bodies: Optional mapping of note id to its body text, for the prompt.
    """

    def __init__(
        self,
        config: Config,
        client: OllamaClient,
        retriever: Retriever,
        note_bodies: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.retriever = retriever
        self.note_bodies = note_bodies or {}

    def generate(
        self,
        decision: Decision,
        k: int | None = None,
        similarity_threshold: float | None = None,
        assumption_threshold: float | None = None,
    ) -> BriefResult:
        """Generate a brief for one decision.

        Args:
            decision: The note to brief.
            k: Number of past decisions to retrieve.
            similarity_threshold: Minimum similarity for a past decision.
            assumption_threshold: Minimum similarity for a failed assumption.

        Returns:
            The assembled brief plus everything needed to audit it.
        """
        k = k if k is not None else self.config.retrieval.default_k
        note_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.retrieval.similarity_threshold
        )
        assumption_threshold = (
            assumption_threshold
            if assumption_threshold is not None
            else self.config.retrieval.failed_assumption_threshold
        )

        similar, failed = self.retriever.retrieve(
            decision,
            k=k,
            note_threshold=note_threshold,
            assumption_threshold=assumption_threshold,
        )
        scoring = compute_scores(decision)
        leading = leading_option(decision, scoring)

        retrieved_ids = [note.note_id for note in similar]
        prompt = build_prompt(decision, similar, failed, self.note_bodies, leading)

        raw, report, retries = self._generate_with_validation(prompt, retrieved_ids)

        sections = parse_sections(raw)
        relevance = parse_relevance_lines(sections.get("similar", ""), retrieved_ids)

        content = render_brief(
            decision=decision,
            similar=similar,
            failed=failed,
            scoring=scoring,
            sections=sections,
            relevance=relevance,
            leading=leading,
            legend=self.config.brief.legend or DEFAULT_LEGEND,
            validation=report,
            retries=retries,
        )

        return BriefResult(
            decision=decision,
            content=content,
            similar=similar,
            failed_assumptions=failed,
            scoring=scoring,
            leading=leading,
            validation=report,
            retries=retries,
            sections=sections,
            relevance=relevance,
            raw_response=raw,
        )

    def _generate_with_validation(
        self, prompt: str, retrieved_ids: list[str]
    ) -> tuple[str, ValidationReport, int]:
        """Call the model and validate the result, retrying once on failure."""
        max_retries = max(0, self.config.brief.max_retries)
        raw = ""
        report = ValidationReport()

        for attempt in range(max_retries + 1):
            current_prompt = prompt
            if attempt > 0:
                current_prompt = build_retry_prompt(prompt, report, retrieved_ids)

            raw = merge_lone_tag_lines(self._call_model(current_prompt))
            report = validate_claims(blank_similar_block(raw), retrieved_ids)

            if report.passed:
                return raw, report, attempt

            logger.warning(
                "Brief attempt %d failed validation with %d issue(s): %s",
                attempt + 1,
                len(report.issues),
                report.by_reason(),
            )

        # Out of retries: mark the offending claims and keep the rest.
        repaired = mark_unverified(raw, report)
        return repaired, report, max_retries

    def _call_model(self, prompt: str) -> str:
        """Send a prompt to the model and return the raw response."""
        return self.client.generate(
            prompt,
            model=self.config.models.llm_model,
            system=SYSTEM_PROMPT,
        )


