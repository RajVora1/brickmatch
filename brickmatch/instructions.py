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
    x0, y0 = grid_origin(build)
    steps = []
    for z in sorted({p["z"] for p in build["layout"]}):
        placed = []
        for p in sorted((p for p in build["layout"] if p["z"] == z), key=lambda p: (p["y"], p["x"])):
            owned = None
            if have is not None:
                owned = left[p["piece"]] > 0
                left[p["piece"]] -= owned
            placed.append({**p, "owned": owned, "n": len(placed) + 1, "where": where(p, x0, y0)})
        steps.append({"number": len(steps) + 1, "z": z, "pieces": placed, "on": _resting_on(build, z, steps)})
    return steps


def grid_origin(build):
    """The front-left stud of the whole build. Columns and rows in the instructions count from here."""
    return min(p["x"] for p in build["layout"]), min(p["y"] for p in build["layout"])


def _span(a, n):
    return f"{a}" if n == 1 else f"{a}-{a + n - 1}"


def where(p, x0=0, y0=0):
    """Say in words where a piece goes, e.g. 'left to right over columns 1-4, row 1 (front)'.

    Columns run left to right and rows run front to back, both counted from 1 at the front-left of the build.
    """
    w, d, _ = footprint(p)
    col, row = p["x"] - x0 + 1, p["y"] - y0 + 1
    cols = f"column{'s' if w > 1 else ''} {_span(col, w)}"
    rows = f"row{'s' if d > 1 else ''} {_span(row, d)}" + (" (front)" if row == 1 else "")
    if p["piece"] in SLOPES:
        facing = {0: "front", 90: "right", 180: "back", 270: "left"}[p.get("rot", 0)]
        return f"{cols}, {rows}, sloping down toward the {facing}"
    if w == d:
        return f"{cols}, {rows}"
    return f"{'left to right' if w > d else 'front to back'} over {cols}, {rows}"


def _resting_on(build, z, earlier):
    """What a step's pieces sit on: the ground, or the step(s) whose tops are at height z."""
    if z == 0:
        return "on the table"
    below = [s["number"] for s in earlier if any(q["z"] + footprint(q)[2] == z for q in s["pieces"])]
    return "on top of step " + " and ".join(map(str, below)) if below else "on top of the layer below"


def step_text(step):
    """One numbered line per piece, matching the numbers on the step picture, e.g.
    '✓ 1. Brick 1 x 4: left to right over columns 1-4, row 1 (front)'. ✗ marks a piece you're missing.
    """
    lines = []
    for p in step["pieces"]:
        mark = {True: "✓", None: "•", False: "✗"}[p["owned"]]
        lines.append(f"{mark} {p['n']}. {NAMES.get(p['piece'], p['piece'])}: {p['where']}"
                     + (" (missing)" if p["owned"] is False else ""))
    return lines


def step_title(step):
    """e.g. 'Step 2: 3 pieces, on top of step 1'."""
    n = len(step["pieces"])
    return f"Step {step['number']}: {n} piece{'s' if n > 1 else ''}, {step['on']}"


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


def _piece_color(p, new):
    """(fill rgb, alpha, edge) for a piece: grey if from an earlier step, red see-through if missing."""
    if not new:
        return (0.82, 0.82, 0.82), 1.0, (0.55, 0.55, 0.55)
    if p["owned"] is False:
        return (0.9, 0.15, 0.15), 0.35, (0.7, 0.0, 0.0)
    return _rgb(p), 1.0, (0.1, 0.1, 0.1)


def draw_step(build, steps, number):
    """Draw step `number` as two pictures side by side and return the matplotlib figure.

    Left: the build so far in 3D, with earlier steps in grey and this step's pieces in color.
    Right: a top-down map of this layer on a stud grid, so you can count studs to find each spot.
    Both pictures number the new pieces the same way as step_text. Missing pieces are red and see-through.
    """
    import matplotlib  # noqa: PLC0415  (imported here so the rest works without matplotlib)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig = plt.figure(figsize=(10, 5))
    _draw_3d(build, steps, number, fig.add_subplot(1, 2, 1, projection="3d"))
    _draw_map(build, steps, number, fig.add_subplot(1, 2, 2))
    step = steps[number - 1]
    fig.suptitle(f"{build['name']}: {step_title(step)} (of {len(steps)} steps)", fontsize=12)
    fig.tight_layout()
    return fig


def _draw_3d(build, steps, number, ax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    shown = [(p, s["number"] == number) for s in steps if s["number"] <= number for p in s["pieces"]]
    filled = set().union(*(cells(p) for p, _ in shown))
    faces, fills, edges = [], [], []
    for p, new in shown:
        rgb, alpha, edge = _piece_color(p, new)
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
    # Number each new piece on its top, matching the map and the checklist.
    for p in steps[number - 1]["pieces"]:
        w, d, h = footprint(p)
        ax.text(p["x"] + w / 2, p["y"] + d / 2, (p["z"] + h) * 0.4 + 0.15, str(p["n"]), ha="center",
                va="bottom", fontsize=10, fontweight="bold", zorder=10,
                bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": "black", "lw": 0.8})


def _draw_map(build, steps, number, ax):
    """Top-down stud grid of one layer: the layer underneath in grey, the new pieces numbered on top."""
    from matplotlib.patches import FancyArrow, Rectangle  # noqa: PLC0415

    x0, y0 = grid_origin(build)
    x1 = max(p["x"] + footprint(p)[0] for p in build["layout"])
    y1 = max(p["y"] + footprint(p)[1] for p in build["layout"])
    step = steps[number - 1]
    # Studs you're building on: the top of whatever is directly below this layer.
    below = [q for s in steps[: number - 1] for q in s["pieces"] if q["z"] + footprint(q)[2] == step["z"]]
    for q in below:
        for cx, cy in stud_cells(q):
            ax.add_patch(Rectangle((cx, cy), 1, 1, fc=(0.88, 0.88, 0.88), ec="none"))
    for p in step["pieces"]:
        w, d, _ = footprint(p)
        rgb, alpha, edge = _piece_color(p, True)
        ax.add_patch(Rectangle((p["x"] + 0.06, p["y"] + 0.06), w - 0.12, d - 0.12, fc=(*rgb, max(alpha, 0.5)),
                               ec=edge, lw=1.5, ls="--" if p["owned"] is False else "-"))
        if p["piece"] in SLOPES:  # arrow pointing downhill
            fx, fy = SLOPE_FACING[p.get("rot", 0)]
            cx, cy = p["x"] + w / 2, p["y"] + d / 2
            ax.add_patch(FancyArrow(cx - 0.3 * fx, cy - 0.3 * fy, 0.6 * fx, 0.6 * fy, width=0.06,
                                    head_width=0.25, head_length=0.2, length_includes_head=True, color="black"))
        ax.text(p["x"] + w / 2, p["y"] + d / 2 + (0.3 if p["piece"] in SLOPES else 0), str(p["n"]),
                ha="center", va="center", fontsize=11, fontweight="bold",
                bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": "black", "lw": 0.8})
    # Stud grid with column and row numbers.
    for gx in range(x0, x1 + 1):
        ax.axvline(gx, color=(0.75, 0.75, 0.75), lw=0.6, zorder=0)
    for gy in range(y0, y1 + 1):
        ax.axhline(gy, color=(0.75, 0.75, 0.75), lw=0.6, zorder=0)
    ax.set_xticks([gx + 0.5 for gx in range(x0, x1)], [str(i + 1) for i in range(x1 - x0)])
    ax.set_yticks([gy + 0.5 for gy in range(y0, y1)], [str(i + 1) for i in range(y1 - y0)])
    ax.tick_params(length=0, labelsize=8, labeltop=True, labelbottom=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0 - 0.9, y1)
    ax.text((x0 + x1) / 2, y0 - 0.45, "front", ha="center", va="center", fontsize=9, color=(0.4, 0.4, 0.4))
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("")
    ax.set_title("Seen from above: columns across, rows from the front", fontsize=9)


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
        print(f"\n{step_title(s)}")
        print("\n".join("  " + line for line in step_text(s)))
    if args.have:
        stuck = first_stuck_step(steps)
        print("\nYou have every piece." if stuck is None else f"\nYou'll get stuck at step {stuck}.")
    if args.out:
        print("\nSaved", *save_steps(b, steps, args.out), sep="\n  ")
