"""Print the layout of a canvas file. Manual verification helper.

Usage:
    python scripts/inspect_canvas.py <canvas-path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_canvas.py <canvas-path>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No such canvas: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    print(f"{path}")
    print(f"  nodes: {len(nodes)}  edges: {len(edges)}\n")

    by_id = {n["id"]: n for n in nodes}
    for node in sorted(nodes, key=lambda n: (n["y"], n["x"])):
        if node["type"] == "file":
            label = f"[file] {node['file']}"
        else:
            text = node.get("text", "")
            first = text.split("\n")[0][:52]
            label = f"[text] {first}"
        print(f"  ({node['x']:>5}, {node['y']:>5})  {label}")

    print("\n  edges:")
    for edge in edges:
        source = by_id.get(edge["fromNode"], {})
        target = by_id.get(edge["toNode"], {})
        s = source.get("file") or source.get("text", "").split("\n")[0][:30]
        t = target.get("text", "").split("\n")[0][:30]
        print(f"    {s}  ->  {t}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
