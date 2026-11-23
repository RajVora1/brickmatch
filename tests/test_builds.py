"""Sanity checks for builds/builds.json. Run with `python -m pytest tests` or `python tests/test_builds.py`."""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickmatch.instructions import check_layout, first_stuck_step, make_steps  # noqa: E402
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



def test_layouts_are_buildable():
    """Every layout matches its parts list, has no overlapping pieces and nothing floating."""
    for b in load_builds():
        if b.get("layout"):
            assert check_layout(b) == [], f"{b['id']}: {check_layout(b)}"


def test_layout_checks_catch_mistakes():
    good = {"id": "t", "parts": [{"piece": "brick_2x2", "count": 2}],
            "layout": [{"piece": "brick_2x2", "x": 0, "y": 0, "z": 0, "rot": 0},
                       {"piece": "brick_2x2", "x": 1, "y": 0, "z": 3, "rot": 0}]}
    assert check_layout(good) == []
    overlap = {**good, "layout": [good["layout"][0], {**good["layout"][1], "z": 1}]}
    assert any("overlaps" in m for m in check_layout(overlap))
    floating = {**good, "layout": [good["layout"][0], {**good["layout"][1], "x": 5}]}
    assert any("not pressed onto" in m for m in check_layout(floating))
    wrong_count = {**good, "parts": [{"piece": "brick_2x2", "count": 3}]}
    assert any("do not match" in m for m in check_layout(wrong_count))
    # Only the back row of a slope has studs, so a brick on the sloped front row is floating.
    on_slope = {"id": "s", "parts": [{"piece": "slope45_2x1", "count": 1}, {"piece": "brick_1x1", "count": 1}],
                "layout": [{"piece": "slope45_2x1", "x": 0, "y": 0, "z": 0, "rot": 0},
                           {"piece": "brick_1x1", "x": 0, "y": 0, "z": 3, "rot": 0}]}
    assert any("not pressed onto" in m for m in check_layout(on_slope))
    on_slope["layout"][1]["y"] = 1
    assert check_layout(on_slope) == []


def test_steps_check_off_owned_pieces_in_order():
    wall = next(b for b in load_builds() if b["id"] == "wall_low")
    steps = make_steps(wall, {"brick_1x4": 6, "brick_1x2": 2})
    assert [s["z"] for s in steps] == [0, 3, 6, 9]
    assert sum(len(s["pieces"]) for s in steps) == 10
    assert first_stuck_step(steps) == 4  # the last row's two 1x2 bricks are the missing ones
    assert first_stuck_step(make_steps(wall, {"brick_1x4": 6, "brick_1x2": 4})) is None
    assert all(p["owned"] is None for s in make_steps(wall) for p in s["pieces"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"ok: {len(load_builds())} builds, by category {dict(Counter(b['category'] for b in load_builds()))}, "
          f"{sum(bool(b.get('layout')) for b in load_builds())} with step-by-step layouts")
