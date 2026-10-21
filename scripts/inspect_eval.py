"""Print a summary of an evaluation report. Manual verification helper.

Usage:
    python scripts/inspect_eval.py [path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("vault/.whatif/eval-results.json")

    if not path.exists():
        print(f"No report at {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"model      : {data.get('embed_model')}")
    print(f"threshold  : {data.get('threshold')}")
    print(f"hit rate   : {data['hits']}/{data['total']} = {data['hit_rate']:.0%}")
    print()

    for case in data["cases"]:
        print(f"{'HIT ' if case['hit'] else 'MISS'}  {case['note_id']}")
        print(f"    true     : {case['best_match_true']}")
        print(f"    surfaced : {case['best_match_surfaced']}  [{case['best_match_source']}]")
        top = sorted(
            case["surfaced"], key=lambda s: -s["query_similarity"]
        )[:3]
        for item in top:
            print(
                f"      q={item['query_similarity']:.3f} "
                f"best={item['best_match']:.3f}  {item['text'][:66]}"
            )
        print()

    print("rates by threshold:")
    for row in data["rates_by_threshold"]:
        print(f"  {row['threshold']:.2f}  {row['hits']}/{row['total']} = {row['rate']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
