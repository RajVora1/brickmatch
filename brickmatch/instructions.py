"""Step-by-step build instructions: one step per layer, with the pieces you own checked off.

A build in builds/builds.json can have a "layout": every piece with its place on a stud grid.

    {"piece": "brick_1x4", "x": 0, "y": 0, "z": 0, "rot": 0}

- x, y: the stud of the piece's corner nearest the origin (x runs left to right, y runs front to back).
- z: height in plate units. A plate is 1 tall and a brick (or slope) is 3, so the second row of bricks is z=3.
- rot: 0, 90, 180 or 270 degrees, turning counterclockwise seen from above. 90 and 270 swap the
  piece's length and width. For slopes it also sets which way the slope faces (0 = down toward the front).
- color (optional): a color id from configs/colors.yaml. Leave it out when any color works.

The same layout is meant to feed the 3D viewer later, so nothing here is specific to drawing pictures.

Print a build's steps, checking against what you own, and save one picture per step:

    python -m brickmatch.instructions wall_low --have '{"brick_1x4": 6, "brick_1x2": 2}' --out steps/
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from brickmatch.match import load_builds

ROOT = Path(__file__).resolve().parent.parent
NAMES = {p["id"]: p["name"] for p in yaml.safe_load((ROOT / "configs/pieces.yaml").read_text())["pieces"]}
COLOR_RGB = {c["id"]: c["rgb"] for c in yaml.safe_load((ROOT / "configs/colors.yaml").read_text())["colors"]}

# Size at rot 0: (studs along x, studs along y, height in plates).
SHAPES = {
    "brick_1x1": (1, 1, 3),
    "brick_1x2": (2, 1, 3),
    "brick_1x4": (4, 1, 3),
    "brick_2x2": (2, 2, 3),
    "brick_2x4": (4, 2, 3),
    "plate_1x2": (2, 1, 1),
    "slope45_2x1": (1, 2, 3),  # 1 wide, 2 deep: studs on the back row, slope over the front row
    "slope45_2x2": (2, 2, 3),
}
SLOPES = {"slope45_2x1", "slope45_2x2"}
# Which way a slope faces (runs downhill) for each rotation, as a step in (x, y).
SLOPE_FACING = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
# Used when a placement has no color, so each piece type looks different in the pictures.
DEFAULT_COLORS = {
    "brick_1x1": "yellow", "brick_1x2": "blue", "brick_1x4": "red", "brick_2x2": "green",
    "brick_2x4": "red", "plate_1x2": "tan", "slope45_2x1": "blue", "slope45_2x2": "light_bluish_gray",
}


def footprint(p):
    """(width along x, depth along y, height) of a placed piece after rotation."""
    w, d, h = SHAPES[p["piece"]]
    return (d, w, h) if p.get("rot", 0) in (90, 270) else (w, d, h)


def cells(p):
    """Every (x, y, z) unit cell the piece fills: one stud square by one plate high."""
    w, d, h = footprint(p)
    return {(p["x"] + i, p["y"] + j, p["z"] + k) for i in range(w) for j in range(d) for k in range(h)}


def stud_cells(p):
    """The (x, y) squares on top of the piece that have studs (a slope only has them on its back row)."""
    w, d, _ = footprint(p)
    top = {(p["x"] + i, p["y"] + j) for i in range(w) for j in range(d)}
    if p["piece"] in SLOPES:
        fx, fy = SLOPE_FACING[p.get("rot", 0)]
        # The back row is the one furthest from the direction the slope faces.
        if fx:
            back = p["x"] if fx > 0 else p["x"] + w - 1
            top = {c for c in top if c[0] == back}
        else:
            back = p["y"] if fy > 0 else p["y"] + d - 1
            top = {c for c in top if c[1] == back}
    return top


def check_layout(build):
    """Return a list of problems with a build's layout (empty means it looks buildable)."""
    layout = build.get("layout") or []
    problems = []
    for i, p in enumerate(layout):
        if p["piece"] not in SHAPES:
            problems.append(f"piece {i}: unknown piece {p['piece']}")
        if p.get("rot", 0) not in SLOPE_FACING:
            problems.append(f"piece {i}: rot must be 0, 90, 180 or 270")
    if problems:
        return problems
    want = {p["piece"]: p["count"] for p in build["parts"]}
    got = dict(Counter(p["piece"] for p in layout))
    if want != got:
        problems.append(f"layout pieces {got} do not match the parts list {want}")
    filled = {}
    for i, p in enumerate(layout):
        for c in cells(p):
            if c in filled:
                problems.append(f"piece {i} ({p['piece']}) overlaps piece {filled[c]} at {c}")
                break
            filled[c] = i
    # Studs you can press onto: (x, y, height of the top surface) -> piece index.
    studs = {(x, y, p["z"] + footprint(p)[2]): i for i, p in enumerate(layout) for x, y in stud_cells(p)}
    for i, p in enumerate(layout):
        if p["z"] == 0:
            continue
        w, d, _ = footprint(p)
        under = [(p["x"] + a, p["y"] + b, p["z"]) for a in range(w) for b in range(d)]
        if not any(c in studs for c in under):
            problems.append(f"piece {i} ({p['piece']}) at z={p['z']} is not pressed onto any studs")
    return problems


def make_steps(build, have=None):
    """Group a build's layout into steps (one per layer, bottom first) and check off owned pieces.

    have: piece counts you own, e.g. from the detector. None means don't check anything off.
    Pieces are used up in step order, so if you are short, the later steps are the ones marked missing.
    """
    left = Counter(have or {})
    steps = []
    for z in sorted({p["z"] for p in build["layout"]}):
        placed = []
        for p in sorted((p for p in build["layout"] if p["z"] == z), key=lambda p: (p["y"], p["x"])):
            owned = None
            if have is not None:
                owned = left[p["piece"]] > 0
                left[p["piece"]] -= owned
            placed.append({**p, "owned": owned})
        steps.append({"number": len(steps) + 1, "z": z, "pieces": placed})
    return steps


def step_text(step):
    """One line per piece type in a step, e.g. '✓ 2 x Brick 1 x 4' or '✗ 1 x Brick 1 x 2 (missing)'."""
    lines = []
    for owned in (True, None, False):
        n = Counter(p["piece"] for p in step["pieces"] if p["owned"] is owned)
        for piece, k in sorted(n.items()):
            mark = {True: "✓", None: "•", False: "✗"}[owned]
            lines.append(f"{mark} {k} x {NAMES.get(piece, piece)}" + (" (missing)" if owned is False else ""))
    return lines


def first_stuck_step(steps):
    """The number of the first step with a missing piece, or None if you have everything."""
    return next((s["number"] for s in steps if any(p["owned"] is False for p in s["pieces"])), None)


def _rgb(p):
    c = COLOR_RGB.get(p.get("color") or DEFAULT_COLORS.get(p["piece"], "light_bluish_gray"), (160, 165, 169))
    return tuple(v / 255 for v in c)


def _shade(rgb, k):
    return tuple(min(1.0, v * k) for v in rgb)


def _piece_faces(p):
    """The flat faces (each a list of 3D corners) of one piece, with a slope cut as a wedge.

    Units: 1 = one stud; a plate is 0.4 tall and a brick 1.2, like real LEGO proportions.
    """
    w, d, h = footprint(p)
    x0, y0, z0 = p["x"], p["y"], p["z"] * 0.4
    x1, y1, z1 = x0 + w, y0 + d, z0 + h * 0.4
    if p["piece"] not in SLOPES:
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
    # Build the wedge facing -y in local coordinates, then turn it.
    lw, ld = (w, d) if p.get("rot", 0) in (0, 180) else (d, w)
    lo = z0 + 0.4  # the low front edge of a real slope is about one plate tall
    local = [
        [(0, 0, z0), (lw, 0, z0), (lw, ld, z0), (0, ld, z0)],  # bottom
        [(0, ld - 1, z1), (lw, ld - 1, z1), (lw, ld, z1), (0, ld, z1)],  # flat top with the studs
        [(0, 0, lo), (lw, 0, lo), (lw, ld - 1, z1), (0, ld - 1, z1)],  # the slope
        [(0, 0, z0), (lw, 0, z0), (lw, 0, lo), (0, 0, lo)],  # low front
        [(0, ld, z0), (lw, ld, z0), (lw, ld, z1), (0, ld, z1)],  # back
        [(0, 0, z0), (0, ld, z0), (0, ld, z1), (0, ld - 1, z1), (0, 0, lo)],  # left side
        [(lw, 0, z0), (lw, ld, z0), (lw, ld, z1), (lw, ld - 1, z1), (lw, 0, lo)],  # right side
    ]
    rot = p.get("rot", 0)

    def turn(x, y):
        return {0: (x, y), 90: (w - y, x), 180: (w - x, d - y), 270: (y, d - x)}[rot]

    return [[(x0 + turn(x, y)[0], y0 + turn(x, y)[1], z) for x, y, z in face] for face in local]


def _covered(p, face, filled):
    """True if a box piece's face (index from _piece_faces) is completely hidden by other pieces."""
    if p["piece"] in SLOPES:
        return False
    w, d, h = footprint(p)
    x, y, z = p["x"], p["y"], p["z"]
    other = {
        1: [(x + i, y + j, z + h) for i in range(w) for j in range(d)],
        2: [(x + i, y - 1, z + k) for i in range(w) for k in range(h)],
        3: [(x + i, y + d, z + k) for i in range(w) for k in range(h)],
        4: [(x - 1, y + j, z + k) for j in range(d) for k in range(h)],
        5: [(x + w, y + j, z + k) for j in range(d) for k in range(h)],
    }[face]
    return all(c in filled for c in other)


def draw_step(build, steps, number, ax=None):
    """Draw the build up to step `number`: earlier steps in grey, this step's pieces in color.

    Pieces you're missing are drawn red and see-through. Returns the matplotlib figure.
    """
    import matplotlib  # noqa: PLC0415  (imported here so the rest works without matplotlib)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
    fig = ax.figure
    shown = [(p, s["number"] == number) for s in steps if s["number"] <= number for p in s["pieces"]]
    filled = set().union(*(cells(p) for p, _ in shown))
    faces, fills, edges = [], [], []
    for p, new in shown:
        if not new:
            rgb, alpha, edge = (0.82, 0.82, 0.82), 1.0, (0.55, 0.55, 0.55)
        elif p["owned"] is False:
            rgb, alpha, edge = (0.9, 0.15, 0.15), 0.35, (0.7, 0.0, 0.0)
        else:
            rgb, alpha, edge = _rgb(p), 1.0, (0.1, 0.1, 0.1)
        # Simple lighting: tops brightest, sides darker, so the layers are easy to tell apart.
        piece_faces = _piece_faces(p)
        shades = [0.55, 1.0, 0.9, 0.7, 0.7, 0.7, 0.7] if p["piece"] in SLOPES else [0.55, 1.0, 0.75, 0.75, 0.6, 0.6]
        for i, (face, k) in enumerate(zip(piece_faces, shades)):
            if i == 0 or _covered(p, i, filled):
                continue  # bottoms never show from above, and faces pressed against another piece are hidden
            faces.append(face)
            fills.append((*_shade(rgb, k), alpha))
            edges.append(edge)
    # One collection for everything, so matplotlib sorts faces (not whole pieces) back to front.
    ax.add_collection3d(Poly3DCollection(faces, facecolors=fills, edgecolors=edges, linewidths=0.8))
    xs = [c for p in build["layout"] for c in (p["x"], p["x"] + footprint(p)[0])]
    ys = [c for p in build["layout"] for c in (p["y"], p["y"] + footprint(p)[1])]
    zs = [(p["z"] + footprint(p)[2]) * 0.4 for p in build["layout"]]
    # Same scale on every axis, sized to the whole build so the picture doesn't jump between steps.
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(0, max(zs))
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), max(zs)), zoom=0.9)
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    ax.set_title(f"Step {number} of {len(steps)}")
    return fig


def save_steps(build, steps, out_dir):
    """Save one PNG per step and return their paths."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in steps:
        fig = draw_step(build, steps, s["number"])
        path = out / f"{build['id']}_step{s['number']:02d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def get_build(build_id, builds=None):
    for b in builds if builds is not None else load_builds():
        if b["id"] == build_id:
            return b
    raise KeyError(f"no build called {build_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("build", help="build id, e.g. wall_low")
    ap.add_argument("--have", help="JSON piece counts you own; leave out to just list the steps")
    ap.add_argument("--out", help="folder to save one picture per step")
    args = ap.parse_args()
    b = get_build(args.build)
    if not b.get("layout"):
        raise SystemExit(f"{b['name']} has no layout yet, so it has no step-by-step instructions.")
    steps = make_steps(b, json.loads(args.have) if args.have else None)
    print(f"{b['name']}: {len(steps)} steps")
    for s in steps:
        print(f"\nStep {s['number']}")
        print("\n".join("  " + line for line in step_text(s)))
    if args.have:
        stuck = first_stuck_step(steps)
        print("\nYou have every piece." if stuck is None else f"\nYou'll get stuck at step {stuck}.")
    if args.out:
        print("\nSaved", *save_steps(b, steps, args.out), sep="\n  ")
