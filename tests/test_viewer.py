"""Checks on the 3D viewer's piece shapes: every piece is a closed solid in the right place, and the page builds.

    python -m pytest tests/test_viewer.py
"""
from collections import Counter

import pytest

from brickmatch.instructions import SLOPES, footprint, stud_cells
from brickmatch.match import load_builds
from brickmatch.viewer import GAP, PLATE, _solid, embed, page, scene

BUILDS = [b for b in load_builds() if b.get("layout")]
PIECES = [(b["id"], i, p) for b in BUILDS for i, p in enumerate(b["layout"])]


@pytest.mark.parametrize("build_id,i,p", PIECES, ids=[f"{b}-{i}" for b, i, _ in PIECES])
def test_piece_is_a_closed_solid_on_its_spot(build_id, i, p):
    verts, faces = _solid(p)
    w, d, h = footprint(p)
    xs, ys, zs = zip(*verts)
    assert min(xs) == pytest.approx(p["x"] + GAP) and max(xs) == pytest.approx(p["x"] + w - GAP)
    assert min(ys) == pytest.approx(p["y"] + GAP) and max(ys) == pytest.approx(p["y"] + d - GAP)
    assert min(zs) == pytest.approx(p["z"] * PLATE) and max(zs) == pytest.approx((p["z"] + h) * PLATE)
    # Closed: every edge is shared by exactly two faces.
    edges = Counter(tuple(sorted((f[k], f[(k + 1) % len(f)]))) for f in faces for k in range(len(f)))
    assert set(edges.values()) == {2}


@pytest.mark.parametrize("build_id,i,p", [t for t in PIECES if t[2]["piece"] in SLOPES],
                         ids=[f"{b}-{i}" for b, i, p in PIECES if p["piece"] in SLOPES])
def test_slope_is_tall_over_its_studs_and_low_at_the_front(build_id, i, p):
    verts, _ = _solid(p)
    top = (p["z"] + 3) * PLATE
    studs = stud_cells(p)
    high = [(x, y) for x, y, z in verts if abs(z - top) < 1e-6]
    # The full-height top is exactly the stud row.
    sx, sy = zip(*studs)
    assert min(x for x, _ in high) == pytest.approx(min(sx) + (GAP if min(sx) == p["x"] else 0), abs=1e-6)
    assert max(x for x, _ in high) == pytest.approx(max(sx) + 1 - (GAP if max(sx) + 1 == p["x"] + footprint(p)[0] else 0), abs=1e-6)
    assert min(y for _, y in high) == pytest.approx(min(sy) + (GAP if min(sy) == p["y"] else 0), abs=1e-6)
    assert max(y for _, y in high) == pytest.approx(max(sy) + 1 - (GAP if max(sy) + 1 == p["y"] + footprint(p)[1] else 0), abs=1e-6)
    # And the downhill edge is one plate high.
    assert any(abs(z - (p["z"] + 1) * PLATE) < 1e-6 for _, _, z in verts)


def test_missing_pieces_match_the_instructions():
    b = next(b for b in BUILDS if b["id"] == "house_hut")
    s = scene(b, {"brick_1x4": 4, "brick_1x2": 6, "brick_1x1": 2, "slope45_2x2": 1})
    assert s["missing"] == 3 and s["total"] == 16
    assert {p["name"] for p in s["pieces"] if p["missing"]} == {"Slope 45 2 x 2"}
    assert scene(b)["missing"] == 0 and scene(b)["checked"] is False


def test_hidden_studs_are_left_out():
    b = next(b for b in BUILDS if b["id"] == "tower_lighthouse")
    s = scene(b)
    assert sum(len(p["studs"]) for p in s["pieces"]) < sum(len(stud_cells(p)) for p in b["layout"])


@pytest.mark.parametrize("b", BUILDS, ids=[b["id"] for b in BUILDS])
def test_page_builds(b):
    html = page(b, {})
    assert "__" not in html.split("<script", 1)[0] and "__DATA__" not in html and "__THREE__" not in html
    assert embed(b).startswith("<iframe srcdoc=")
