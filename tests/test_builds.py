"""Sanity checks for builds/builds.json. Run with `python -m pytest tests` or `python tests/test_builds.py`."""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickmatch.match import load_builds, rank_builds  # noqa: E402

TIER1 = {p["id"] for p in yaml.safe_load((ROOT / "configs" / "pieces.yaml").read_text())["pieces"] if p["tier"] == 1}
CATEGORIES = {"tower", "wall", "bridge", "house"}


def test_builds_use_tier1_pieces_only():
    for b in load_builds():
        for p in b["parts"]:
            assert p["piece"] in TIER1, f"{b['id']} uses {p['piece']}, not a tier-1 piece"
            assert p["count"] > 0


def test_ids_unique_and_categories_known():
    builds = load_builds()
    assert len({b["id"] for b in builds}) == len(builds)
    assert {b["category"] for b in builds} == CATEGORIES


def test_exact_inventory_gives_full_coverage():
    for b in load_builds():
        inv = {p["piece"]: p["count"] for p in b["parts"]}
        top = {r["id"]: r for r in rank_builds(inv)}
        assert top[b["id"]]["coverage"] == 1.0 and top[b["id"]]["missing"] == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"ok: {len(load_builds())} builds, by category {dict(Counter(b['category'] for b in load_builds()))}")
