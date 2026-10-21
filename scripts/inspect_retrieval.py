"""Inspect what the retriever returns for a note. Manual verification helper.

Usage:
    python scripts/inspect_retrieval.py <note-id> [--k 5]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whatif.config import load_config  # noqa: E402
from whatif.embeddings import build_embedder  # noqa: E402
from whatif.notes import parse_decision_file  # noqa: E402
from whatif.retrieve import Retriever  # noqa: E402
from whatif.store import DecisionStore  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_retrieval.py <note-path-or-id> [--k N]")
        return 2

    target = sys.argv[1]
    k = 5
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])

    config = load_config("whatif.toml")
    store = DecisionStore(config.vault.db_path)
    try:
        note_path = Path(target)
        if not note_path.exists():
            matches = list(config.vault.decisions_path.glob(f"{target}*.md"))
            if not matches:
                print(f"No note found for {target}")
                return 1
            note_path = matches[0]

        result = parse_decision_file(note_path)
        if result.decision is None:
            print(f"Parse failed: {result.error}")
            return 1
        decision = result.decision

        retriever = Retriever(store, embedder=build_embedder(config))
        similar, failed = retriever.retrieve(
            decision,
            k=k,
            note_threshold=config.retrieval.similarity_threshold,
            assumption_threshold=config.retrieval.failed_assumption_threshold,
        )

        print(f"\nNOTE: {decision.id}  ({decision.domain}, {decision.status})")
        print(f"Assumptions: {decision.assumptions}")

        print(f"\nSIMILAR DECISIONS (k={k}):")
        if not similar:
            print("  (none above threshold)")
        for item in similar:
            print(f"  {item.similarity:+.3f}  {item.note_id}")
            print(f"          {item.summary[:110]}")

        print("\nASSUMPTIONS THAT FAILED BEFORE:")
        if not failed:
            print("  (none above threshold)")
        for item in failed:
            print(f"  {item.similarity:+.3f}  [{item.source_note_id}] {item.text}")

        print()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
