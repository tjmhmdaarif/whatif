"""Tests for canvas.py - Obsidian Canvas JSON structure."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from whatif.canvas import (
    ASSUMPTION_Y,
    NODE_HEIGHT,
    NODE_WIDTH,
    OPTION_Y,
    RISK_Y,
    Canvas,
    CanvasEdge,
    CanvasNode,
    build_canvas,
    make_id,
    write_canvas,
)
from whatif.config import Config, create_default_config
from whatif.notes import Decision
from whatif.scoring import compute_scores


def make_decision(
    options: list[str] | None = None,
    assumptions: list[str] | None = None,
    risks: list[str] | None = None,
    file_path: Path | None = None,
) -> Decision:
    """Build a Decision for testing."""
    options = options or ["Option A", "Option B"]
    scores = {option: {"price": 4, "speed": 3} for option in options}
    return Decision(
        id="2026-11-01-test-decision",
        type="decision",
        status="open",
        created=datetime.date(2026, 11, 1),
        domain="tools",
        options=options,
        chosen="",
        criteria=[{"name": "price", "weight": 3}, {"name": "speed", "weight": 2}],
        scores=scores,
        assumptions=assumptions if assumptions is not None else ["Scope stays small."],
        risks=risks if risks is not None else ["Exam period collides."],
        outcome={"reviewed_on": None, "result": "", "failed_assumptions": []},
        tags=[],
        context="Choosing a tool for the term project.",
        file_path=file_path,
    )


@pytest.fixture
def vault_with_note(temp_dir: Path) -> Path:
    """A vault with one decision note inside it."""
    vault = temp_dir / "vault"
    (vault / "Decisions").mkdir(parents=True)
    note = vault / "Decisions" / "2026-11-01-test-decision.md"
    note.write_text("placeholder", encoding="utf-8")
    return vault


class TestCanvasNode:
    def test_text_node_shape(self) -> None:
        node = CanvasNode(id="abc", type="text", text="hello")
        payload = node.to_dict()
        assert payload["id"] == "abc"
        assert payload["type"] == "text"
        assert payload["text"] == "hello"
        assert payload["width"] == NODE_WIDTH
        assert payload["height"] == NODE_HEIGHT
        assert "x" in payload and "y" in payload

    def test_file_node_shape(self) -> None:
        node = CanvasNode(id="abc", type="file", file="Decisions/x.md")
        payload = node.to_dict()
        assert payload["type"] == "file"
        assert payload["file"] == "Decisions/x.md"
        assert "text" not in payload

    def test_color_is_optional(self) -> None:
        assert "color" not in CanvasNode(id="a", type="text").to_dict()
        assert CanvasNode(id="a", type="text", color="4").to_dict()["color"] == "4"


class TestCanvasEdge:
    def test_edge_shape(self) -> None:
        edge = CanvasEdge(id="e1", from_node="a", to_node="b")
        payload = edge.to_dict()
        assert payload["id"] == "e1"
        assert payload["fromNode"] == "a"
        assert payload["toNode"] == "b"
        assert payload["fromSide"] == "bottom"
        assert payload["toSide"] == "top"

    def test_label_is_optional(self) -> None:
        assert "label" not in CanvasEdge(id="e", from_node="a", to_node="b").to_dict()
        assert (
            CanvasEdge(id="e", from_node="a", to_node="b", label="x").to_dict()["label"]
            == "x"
        )


class TestMakeId:
    def test_deterministic(self) -> None:
        assert make_id("option", "abc") == make_id("option", "abc")

    def test_different_keys_differ(self) -> None:
        assert make_id("option", "abc") != make_id("option", "abd")

    def test_different_kinds_differ(self) -> None:
        assert make_id("option", "abc") != make_id("risk", "abc")

    def test_length(self) -> None:
        assert len(make_id("option", "abc")) == 16


class TestBuildCanvas:
    def test_has_a_root_node(self) -> None:
        canvas = build_canvas(make_decision())
        assert len(canvas.nodes) >= 1
        assert canvas.nodes[0].type in ("file", "text")

    def test_root_points_at_the_note_when_inside_the_vault(
        self, vault_with_note: Path
    ) -> None:
        config = create_default_config(vault_with_note)
        decision = make_decision(
            file_path=vault_with_note / "Decisions" / "2026-11-01-test-decision.md"
        )
        canvas = build_canvas(decision, config)
        root = canvas.nodes[0]
        assert root.type == "file"
        assert root.file == "Decisions/2026-11-01-test-decision.md"

    def test_root_falls_back_to_text_outside_the_vault(self) -> None:
        config = create_default_config(Path("/tmp/other-vault"))
        decision = make_decision(file_path=Path("/somewhere/else/note.md"))
        canvas = build_canvas(decision, config)
        root = canvas.nodes[0]
        assert root.type == "text"
        assert "2026-11-01-test-decision" in root.text

    def test_one_node_per_option(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B", "C"]))
        option_texts = [n.text for n in canvas.nodes if n.text.startswith("### ")]
        assert len(option_texts) == 3
        assert any("### A" in t for t in option_texts)
        assert any("### B" in t for t in option_texts)
        assert any("### C" in t for t in option_texts)

    def test_root_has_an_edge_to_every_option(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B", "C"]))
        root = canvas.nodes[0]
        outgoing = [e for e in canvas.edges if e.from_node == root.id]
        assert len(outgoing) == 3
        targets = {e.to_node for e in outgoing}
        option_ids = {n.id for n in canvas.nodes if n.text.startswith("### ")}
        assert targets == option_ids

    def test_assumption_and_risk_nodes_exist(self) -> None:
        canvas = build_canvas(
            make_decision(assumptions=["A one", "A two"], risks=["R one"])
        )
        texts = [n.text for n in canvas.nodes]
        assert any("**Assumption**" in t and "A one" in t for t in texts)
        assert any("**Assumption**" in t and "A two" in t for t in texts)
        assert any("**Risk**" in t and "R one" in t for t in texts)

    def test_each_option_links_to_the_assumption_and_risk_children(self) -> None:
        canvas = build_canvas(
            make_decision(options=["A", "B"], assumptions=["A one"], risks=["R one"])
        )
        option_ids = {n.id for n in canvas.nodes if n.text.startswith("### ")}
        child_ids = {
            n.id for n in canvas.nodes if "**Assumption**" in n.text or "**Risk**" in n.text
        }

        for option_id in option_ids:
            targets = {e.to_node for e in canvas.edges if e.from_node == option_id}
            assert child_ids.issubset(targets)

    def test_option_node_shows_the_computed_total(self) -> None:
        canvas = build_canvas(make_decision())
        text = next(n.text for n in canvas.nodes if n.text.startswith("### Option A"))
        # Option A: 4*3 + 3*2 = 18, maximum is 5 * 5 = 25
        assert "18" in text
        assert "25" in text
        assert "COMPUTED" in text

    def test_option_node_lists_criteria(self) -> None:
        canvas = build_canvas(make_decision())
        text = next(n.text for n in canvas.nodes if n.text.startswith("### Option A"))
        assert "price" in text
        assert "speed" in text

    def test_layout_places_options_below_the_root(self) -> None:
        canvas = build_canvas(make_decision())
        root = canvas.nodes[0]
        option_nodes = [n for n in canvas.nodes if n.text.startswith("### ")]
        assert root.y < all(n.y for n in option_nodes)
        assert all(n.y == OPTION_Y for n in option_nodes)

    def test_assumptions_sit_above_risks(self) -> None:
        canvas = build_canvas(
            make_decision(assumptions=["A one", "A two"], risks=["R one"])
        )
        assumptions = [n for n in canvas.nodes if "**Assumption**" in n.text]
        risks = [n for n in canvas.nodes if "**Risk**" in n.text]
        assert assumptions[0].y == ASSUMPTION_Y
        assert risks[0].y > assumptions[-1].y

    def test_children_are_centred_below_the_options(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B", "C"]))
        children = [
            n for n in canvas.nodes if "**Assumption**" in n.text or "**Risk**" in n.text
        ]
        assert children
        for child in children:
            assert child.x == 0

    def test_children_do_not_overlap(self) -> None:
        canvas = build_canvas(
            make_decision(assumptions=["A one", "A two", "A three"], risks=["R one", "R two"])
        )
        children = [
            n for n in canvas.nodes if "**Assumption**" in n.text or "**Risk**" in n.text
        ]
        ys = sorted(n.y for n in children)
        for i in range(1, len(ys)):
            assert ys[i] - ys[i - 1] >= NODE_HEIGHT

    def test_options_are_spread_horizontally(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B", "C"]))
        xs = sorted(n.x for n in canvas.nodes if n.text.startswith("### "))
        assert xs[0] < xs[1] < xs[2]

    def test_no_orphan_edges(self) -> None:
        canvas = build_canvas(make_decision())
        node_ids = {n.id for n in canvas.nodes}
        for edge in canvas.edges:
            assert edge.from_node in node_ids
            assert edge.to_node in node_ids

    def test_node_ids_are_unique(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B"]))
        ids = [n.id for n in canvas.nodes]
        assert len(ids) == len(set(ids))

    def test_edge_ids_are_unique(self) -> None:
        canvas = build_canvas(make_decision(options=["A", "B"]))
        ids = [e.id for e in canvas.edges]
        assert len(ids) == len(set(ids))

    def test_is_deterministic(self) -> None:
        first = build_canvas(make_decision()).to_json()
        second = build_canvas(make_decision()).to_json()
        assert first == second

    def test_handles_unicode(self) -> None:
        canvas = build_canvas(
            make_decision(options=["Café", "日本語"], assumptions=["Le prix baisse"])
        )
        payload = json.loads(canvas.to_json())
        texts = [n.get("text", "") for n in payload["nodes"]]
        assert any("Café" in t for t in texts)
        assert any("日本語" in t for t in texts)

    def test_no_scores_still_renders(self) -> None:
        decision = make_decision()
        decision.scores = {}
        canvas = build_canvas(decision)
        text = next(n.text for n in canvas.nodes if n.text.startswith("### Option A"))
        assert "no scores" in text.lower()

    def test_missing_score_is_noted(self) -> None:
        decision = make_decision()
        decision.scores["Option A"] = {"price": 4}
        canvas = build_canvas(decision)
        text = next(n.text for n in canvas.nodes if n.text.startswith("### Option A"))
        assert "speed" in text
        assert "No score entered" in text


class TestSerialization:
    def test_valid_json(self) -> None:
        canvas = build_canvas(make_decision())
        payload = json.loads(canvas.to_json())
        assert "nodes" in payload
        assert "edges" in payload
        assert isinstance(payload["nodes"], list)
        assert isinstance(payload["edges"], list)

    def test_ends_with_newline(self) -> None:
        assert build_canvas(make_decision()).to_json().endswith("\n")

    def test_unicode_is_not_escaped(self) -> None:
        canvas = build_canvas(make_decision(options=["Café", "日本語"]))
        assert "Café" in canvas.to_json()

    def test_to_dict_round_trips(self) -> None:
        canvas = build_canvas(make_decision())
        payload = canvas.to_dict()
        assert json.loads(json.dumps(payload)) == payload


class TestWriteCanvas:
    def test_writes_to_the_canvases_dir(self, vault_with_note: Path) -> None:
        config = create_default_config(vault_with_note)
        decision = make_decision(
            file_path=vault_with_note / "Decisions" / "2026-11-01-test-decision.md"
        )
        path = write_canvas(decision, config)
        assert path == config.vault.canvases_path / "2026-11-01-test-decision.canvas"
        assert path.exists()

    def test_written_file_is_valid_json(self, vault_with_note: Path) -> None:
        config = create_default_config(vault_with_note)
        decision = make_decision()
        path = write_canvas(decision, config)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nodes"]
        assert isinstance(payload["edges"], list)

    def test_explicit_out_path(self, vault_with_note: Path, temp_dir: Path) -> None:
        config = create_default_config(vault_with_note)
        target = temp_dir / "custom" / "my.canvas"
        path = write_canvas(make_decision(), config, target)
        assert path == target
        assert target.exists()

    def test_creates_missing_directories(self, vault_with_note: Path) -> None:
        config = create_default_config(vault_with_note)
        import shutil

        shutil.rmtree(config.vault.canvases_path, ignore_errors=True)
        path = write_canvas(make_decision(), config)
        assert path.exists()

    def test_repeated_writes_are_identical(self, vault_with_note: Path) -> None:
        config = create_default_config(vault_with_note)
        decision = make_decision()
        first = write_canvas(decision, config).read_text(encoding="utf-8")
        second = write_canvas(decision, config).read_text(encoding="utf-8")
        assert first == second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
