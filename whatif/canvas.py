"""Obsidian Canvas export for decision notes.

Produces a JSON Canvas file (https://jsoncanvas.org) with a branching shape:

* a root node holding the decision,
* one node per option, showing that option's computed weighted total,
* edges from the root to every option,
* child nodes for the decision's assumptions and risks, each option linked to them.

Every id is derived from a hash of the content, so re-running the export on an
unchanged note produces a byte-identical file and the diff stays clean.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from whatif.config import Config
from whatif.notes import Decision
from whatif.scoring import ScoringResult, compute_scores

logger = logging.getLogger(__name__)


# Layout constants, in canvas units (pixels).
NODE_WIDTH = 400
NODE_HEIGHT = 200
ROOT_HEIGHT = 160
OPTION_SPACING = 500
CHILD_GAP = 40
ROOT_Y = 0
OPTION_Y = 320
ASSUMPTION_Y = 620
RISK_Y = 620


def make_id(kind: str, key: str) -> str:
    """Return a stable 16 character node id for a piece of content."""
    digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class CanvasNode:
    """A node in an Obsidian canvas."""

    id: str
    type: str  # "text" | "file" | "link"
    x: int = 0
    y: int = 0
    width: int = NODE_WIDTH
    height: int = NODE_HEIGHT
    text: str = ""
    file: str = ""
    color: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the node in the JSON Canvas format."""
        node: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }
        if self.type == "text":
            node["text"] = self.text
        elif self.type == "file":
            node["file"] = self.file
        elif self.type == "link":
            node["url"] = self.text
        if self.color:
            node["color"] = self.color
        return node


@dataclass
class CanvasEdge:
    """An edge between two canvas nodes."""

    id: str
    from_node: str
    to_node: str
    from_side: str = "bottom"
    to_side: str = "top"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the edge in the JSON Canvas format."""
        edge: dict[str, Any] = {
            "id": self.id,
            "fromNode": self.from_node,
            "toNode": self.to_node,
            "fromSide": self.from_side,
            "toSide": self.to_side,
        }
        if self.label:
            edge["label"] = self.label
        return edge


@dataclass
class Canvas:
    """A whole canvas document."""

    nodes: list[CanvasNode] = field(default_factory=list)
    edges: list[CanvasEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the canvas in the JSON Canvas format."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the canvas as a JSON string with a trailing newline."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    def node_by_id(self, node_id: str) -> CanvasNode | None:
        """Return a node by id, or None."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


def _option_text(option: str, scoring: ScoringResult) -> str:
    """Build the text of an option node."""
    result = scoring.option(option)
    if result is None:
        return f"### {option}"

    lines = [f"### {option}", ""]
    if not result.scored_criteria:
        lines.append("**Weighted total (COMPUTED): no scores entered**")
    elif scoring.max_points:
        lines.append(f"**Weighted total (COMPUTED): {result.total:g} of {scoring.max_points:g}**")
    else:
        lines.append(f"**Weighted total (COMPUTED): {result.total:g}**")
    lines.append("")

    if result.contributions:
        lines.append("| Criterion | Score | Weight | Points |")
        lines.append("|---|---|---|---|")
        for criterion in scoring.criteria:
            if option in criterion.scores:
                score = criterion.scores[option]
                points = criterion.weighted[option]
                lines.append(
                    f"| {criterion.name} | {score} | {criterion.weight} | {points:g} |"
                )
        lines.append("")

    if result.missing_criteria:
        lines.append(
            f"*No score entered for: {', '.join(result.missing_criteria)}*"
        )

    return "\n".join(lines).rstrip()


def _label_text(kind: str, body: str) -> str:
    """Build the text of an assumption or risk node."""
    heading = "Assumption" if kind == "assumption" else "Risk"
    return f"**{heading}**\n\n{body}"


def build_canvas(
    decision: Decision,
    config: Config | None = None,
    scoring: ScoringResult | None = None,
) -> Canvas:
    """Build the branching canvas for a decision.

    Args:
        decision: The note to render.
        config: Configuration, used to locate the note inside the vault.
        scoring: Precomputed scoring. Computed here if omitted.

    Returns:
        A ready to serialise canvas.
    """
    if scoring is None:
        scoring = compute_scores(decision)

    canvas = Canvas()

    # -- root node --------------------------------------------------------
    root_file = ""
    if config is not None and decision.file_path is not None:
        try:
            root_file = decision.file_path.resolve().relative_to(
                config.vault.path.resolve()
            ).as_posix()
        except ValueError:
            root_file = ""

    if root_file:
        root = CanvasNode(
            id=make_id("root", decision.id),
            type="file",
            file=root_file,
            x=0,
            y=ROOT_Y,
            width=NODE_WIDTH,
            height=ROOT_HEIGHT,
        )
    else:
        # The note lives outside the vault, so show its text instead.
        fallback = decision.context.strip() or f"Decision {decision.id}"
        root = CanvasNode(
            id=make_id("root", decision.id),
            type="text",
            text=f"# {decision.id}\n\n{fallback}",
            x=0,
            y=ROOT_Y,
            width=NODE_WIDTH,
            height=ROOT_HEIGHT,
        )
    canvas.nodes.append(root)

    # -- option nodes -----------------------------------------------------
    option_nodes: list[CanvasNode] = []
    for index, option in enumerate(decision.options):
        x = (index - (len(decision.options) - 1) / 2) * OPTION_SPACING
        node = CanvasNode(
            id=make_id("option", f"{decision.id}:{option}"),
            type="text",
            text=_option_text(option, scoring),
            x=int(x),
            y=OPTION_Y,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
        )
        option_nodes.append(node)
        canvas.nodes.append(node)

    for node in option_nodes:
        canvas.edges.append(
            CanvasEdge(
                id=make_id("edge", f"root->{node.id}"),
                from_node=root.id,
                to_node=node.id,
                from_side="bottom",
                to_side="top",
            )
        )

    # -- assumption and risk child nodes ----------------------------------
    # The schema keeps assumptions and risks at the note level, so the nodes are
    # shared between the options and each option links to them.
    def add_children(kind: str, items: list[str]) -> list[CanvasNode]:
        created: list[CanvasNode] = []
        for index, item in enumerate(items):
            node = CanvasNode(
                id=make_id(kind, f"{decision.id}:{item}"),
                type="text",
                text=_label_text(kind, item),
                x=0,
                y=0,
                width=NODE_WIDTH,
                height=NODE_HEIGHT,
                color="6" if kind == "risk" else "4",
            )
            created.append(node)
            canvas.nodes.append(node)
        return created

    assumption_nodes = add_children("assumption", decision.assumptions)
    risk_nodes = add_children("risk", decision.risks)

    # Place the children under the options and connect them.
    # The schema keeps assumptions and risks at the note level, so the child nodes
    # are shared between the options. They sit centred below the option row, and
    # every option links to them, which reads as a branching graph rather than a
    # duplicated tree.
    child_y = ASSUMPTION_Y
    for node in assumption_nodes:
        node.x = 0
        node.y = child_y
        child_y += NODE_HEIGHT + CHILD_GAP

    risk_y = child_y if assumption_nodes else RISK_Y
    for node in risk_nodes:
        node.x = 0
        node.y = risk_y
        risk_y += NODE_HEIGHT + CHILD_GAP

    for option_node in option_nodes:
        for child in assumption_nodes + risk_nodes:
            canvas.edges.append(
                CanvasEdge(
                    id=make_id("edge", f"{option_node.id}->{child.id}"),
                    from_node=option_node.id,
                    to_node=child.id,
                    from_side="bottom",
                    to_side="top",
                )
            )

    return canvas


def write_canvas(
    decision: Decision,
    config: Config,
    out_path: Path | str | None = None,
) -> Path:
    """Write the canvas for a decision and return the path written.

    Args:
        decision: The note to render.
        config: Active configuration.
        out_path: Explicit output path. Defaults to ``Canvases/<id>.canvas``.

    Returns:
        The path the canvas was written to.
    """
    canvas = build_canvas(decision, config)
    target = Path(out_path) if out_path else config.vault.canvases_path / f"{decision.id}.canvas"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canvas.to_json(), encoding="utf-8")
    logger.info("Wrote canvas to %s", target)
    return target
