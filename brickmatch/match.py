"""Match a detected piece inventory against the builds database.

This part is plain search, not ML: given counts per piece type, rank builds by coverage.

    python -m brickmatch.match '{"brick_2x2": 8, "plate_2x2": 1, "slope45_2x1": 1}'
"""
import json
import sys
from pathlib import Path

BUILDS_PATH = Path(__file__).resolve().parent.parent / "builds" / "builds.json"


def load_builds(path=BUILDS_PATH):
    return json.loads(Path(path).read_text())["builds"]


def score_build(build, inventory):
    """Return (coverage, missing) where coverage is the fraction of required pieces you own."""
    needed = sum(p["count"] for p in build["parts"])
    have, missing = 0, {}
    for p in build["parts"]:
        owned = inventory.get(p["piece"], 0)
        have += min(owned, p["count"])
        if owned < p["count"]:
            missing[p["piece"]] = p["count"] - owned
    return have / needed, missing


def rank_builds(inventory, builds=None, min_coverage=0.8):
    builds = builds if builds is not None else load_builds()
    results = []
    for b in builds:
        coverage, missing = score_build(b, inventory)
        if coverage >= min_coverage:
            results.append({"id": b["id"], "name": b["name"], "category": b["category"],
                            "coverage": round(coverage, 3), "missing": missing})
    return sorted(results, key=lambda r: (-r["coverage"], sum(r["missing"].values())))


if __name__ == "__main__":
    inv = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(rank_builds(inv, min_coverage=0.0), indent=2))
