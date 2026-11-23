"""Instruction booklets in the style of a printed LEGO booklet, built from a build's layout.

Each page has a parts box in the top-left corner (a picture of every piece the step needs, with "2x" counts),
a big step number, and the model so far drawn with studs and real colors. Steps are kept small (a few pieces
each), the first page shows the finished model, and the last page lists every part in the build.
If you pass the pieces you own, a missing piece gets a red outline on the model and its count turns red.

    python -m brickmatch.booklet house_hut --out booklets/
    python -m brickmatch.booklet house_hut --have '{"brick_1x4": 4, "brick_1x2": 4}' --out booklets/

This writes booklets/house_hut.pdf (every page) and one PNG per page.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from brickmatch.instructions import (NAMES, SLOPES, _covered, _piece_faces, _rgb, _shade, cells, footprint,
                                     get_build, make_steps, stud_cells)

PAGE = (11, 7.5)  # inches, landscape like a booklet page
CALLOUT_FILL = (0.86, 0.92, 0.98)
CALLOUT_EDGE = (0.45, 0.6, 0.8)
STUD_R, STUD_H = 0.3, 0.18  # in stud units; a stud is about 0.6 of a stud-spacing wide and 0.2 tall
MISSING = (0.85, 0.05, 0.05)


def booklet_steps(build, have=None, max_pieces=4):
    """Small steps for a booklet: each layer, split so a step adds at most `max_pieces` pieces.

    Within a layer the back pieces come first, so later pieces don't end up hidden behind earlier ones.
    Owned/missing checks come from make_steps, so they match the plain instructions.
    """
    out = []
    for layer in make_steps(build, have):
        pieces = sorted(layer["pieces"], key=lambda p: (-p["y"], p["x"]))
        n_chunks = math.ceil(len(pieces) / max_pieces)
        size = math.ceil(len(pieces) / n_chunks)  # even out the chunks: 5 pieces -> 3 + 2, not 4 + 1
        for i in range(0, len(pieces), size):
            out.append({"number": len(out) + 1, "pieces": pieces[i:i + size]})
    return out


def _stud_faces(cx, cy, top):
    """Polygons for one stud: a short cylinder standing on the square (cx, cy) at height `top`."""
    x, y, n = cx + 0.5, cy + 0.5, 14
    ring = [(x + STUD_R * math.cos(2 * math.pi * k / n), y + STUD_R * math.sin(2 * math.pi * k / n)) for k in range(n)]
    sides = [[(*ring[k], top), (*ring[(k + 1) % n], top), (*ring[(k + 1) % n], top + STUD_H), (*ring[k], top + STUD_H)]
             for k in range(n)]
    return sides, [(px, py, top + STUD_H) for px, py in ring]


def _add_pieces(ax, pieces, filled, missing_ids=frozenset(), edge_scale=1.0):
    """Add pieces (with studs) to a 3D axis as one face collection, so matplotlib sorts faces back to front."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    faces, fills, edges, widths = [], [], [], []

    def add(face, rgb, alpha, edge, lw):
        faces.append(face)
        fills.append((*rgb, alpha))
        edges.append(edge)
        widths.append(lw)

    for p in pieces:
        rgb = _rgb(p)
        missing = id(p) in missing_ids
        alpha = 0.35 if missing else 1.0
        edge = MISSING if missing else _shade(rgb, 0.45)
        lw = (1.6 if missing else 0.6) * edge_scale
        shades = [0.55, 1.0, 0.92, 0.72, 0.72, 0.8, 0.8] if p["piece"] in SLOPES else [0.55, 1.0, 0.78, 0.78, 0.9, 0.9]
        for i, (face, k) in enumerate(zip(_piece_faces(p), shades)):
            if i == 0 or _covered(p, i, filled):
                continue
            add(face, _shade(rgb, k), alpha, edge, lw)
        top = (p["z"] + footprint(p)[2]) * 0.4
        for cx, cy in stud_cells(p):
            if (cx, cy, p["z"] + footprint(p)[2]) in filled:
                continue  # a piece is pressed onto this stud, so it's hidden
            sides, lid = _stud_faces(cx, cy, top)
            for k, face in enumerate(sides):
                if math.cos(2 * math.pi * (k + 0.5) / len(sides) + math.pi / 3) > 0.2:
                    continue  # back half of the stud never shows from our viewing angle
                add(face, _shade(rgb, 0.75), alpha, _shade(rgb, 0.75), 0)
            add(lid, _shade(rgb, 1.08), alpha, edge, 0.4 * edge_scale)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=fills, edgecolors=edges, linewidths=widths))


def _frame(ax, pieces, pad=0.4):
    """Same scale on all three axes, fitted around the pieces, seen from the front-left like a booklet."""
    xs = [c for p in pieces for c in (p["x"], p["x"] + footprint(p)[0])]
    ys = [c for p in pieces for c in (p["y"], p["y"] + footprint(p)[1])]
    zs = [c for p in pieces for c in (p["z"] * 0.4, (p["z"] + footprint(p)[2]) * 0.4 + STUD_H)]
    lo = (min(xs) - pad, min(ys) - pad, min(zs))
    hi = (max(xs) + pad, max(ys) + pad, max(zs) + pad / 2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect([h - l for l, h in zip(lo, hi)], zoom=1.0)
    ax.view_init(elev=30, azim=-58)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def _draw_model(ax, pieces, missing_ids=frozenset(), frame_to=None):
    """Draw pieces; frame_to (default: the same pieces) sets the view, so every page can share one scale."""
    filled = set().union(*(cells(p) for p in pieces))
    _add_pieces(ax, pieces, filled, missing_ids)
    _frame(ax, frame_to or pieces)


def _part_icon(ax, piece, color_of):
    """One loose piece, turned so its long side runs across, for the parts box."""
    p = {"piece": piece, "x": 0, "y": 0, "z": 0, "rot": 0, "color": color_of}
    if piece in SLOPES:
        p["rot"] = 90  # sloping toward the viewer's right, so the slope shape is easy to see
    _add_pieces(ax, [p], cells(p), edge_scale=0.8)
    _frame(ax, [p], pad=0.15)


def _parts_box(fig, rect, parts, title=None):
    """A rounded box with one picture per part type and its count underneath.

    parts: list of (piece id, color id, count, missing count).
    """
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                      transform=fig.transFigure, fc=CALLOUT_FILL, ec=CALLOUT_EDGE, lw=1.5,
                                      zorder=-1))  # behind the part pictures
    if title:
        fig.text(x + 0.012, y + h - 0.035, title, fontsize=13, fontweight="bold", color=(0.15, 0.2, 0.3))
    cols = len(parts)
    cw = w / cols
    top_gap = 0.05 if title else 0.0
    for i, (piece, color, n, n_missing) in enumerate(parts):
        icon = fig.add_axes((x + i * cw + 0.008, y + 0.05, cw - 0.016, h - 0.06 - top_gap), projection="3d")
        _part_icon(icon, piece, color)
        label = f"{n}x"
        fig.text(x + i * cw + cw / 2, y + 0.018, label, ha="center", fontsize=14, fontweight="bold",
                 color=MISSING if n_missing else "black")
        if n_missing:
            fig.text(x + i * cw + cw / 2, y - 0.025, f"missing {n_missing}", ha="center", fontsize=9, color=MISSING)


def _parts_in(pieces):
    """Group pieces by (type, color) for a parts box, with how many of each are missing."""
    n = Counter((p["piece"], p.get("color")) for p in pieces)
    miss = Counter((p["piece"], p.get("color")) for p in pieces if p.get("owned") is False)
    order = sorted(n, key=lambda k: (-footprint({"piece": k[0], "rot": 0})[0] * footprint({"piece": k[0], "rot": 0})[1], k[0]))
    return [(piece, color, n[(piece, color)], miss[(piece, color)]) for piece, color in order]


def _new_page():
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    return fig


def cover_page(build, steps):
    fig = _new_page()
    pieces = [p for s in steps for p in s["pieces"]]
    fig.text(0.05, 0.9, build["name"], fontsize=34, fontweight="bold", color=(0.12, 0.16, 0.25))
    fig.text(0.05, 0.85, f"{len(pieces)} pieces · {len(steps)} steps", fontsize=15, color=(0.35, 0.4, 0.5))
    _draw_model(fig.add_axes((0.15, 0.03, 0.8, 0.8), projection="3d"), pieces)
    return fig


def step_page(build, steps, number):
    fig = _new_page()
    step = steps[number - 1]
    so_far = [p for s in steps[:number] for p in s["pieces"]]
    missing = {id(p) for p in step["pieces"] if p.get("owned") is False}
    parts = _parts_in(step["pieces"])
    every = [p for s in steps for p in s["pieces"]]
    # Same camera and scale on every page, so the model grows in place like in a printed booklet.
    _draw_model(fig.add_axes((0.2, 0.02, 0.78, 0.9), projection="3d"), so_far, missing, frame_to=every)
    box_w = min(0.13 * len(parts), 0.6)
    _parts_box(fig, (0.04, 0.72, box_w, 0.22), parts)
    fig.text(0.05, 0.56, str(number), fontsize=64, fontweight="bold", color=(0.1, 0.1, 0.1))
    fig.text(0.97, 0.03, f"{number} / {len(steps)}", ha="right", fontsize=10, color=(0.5, 0.5, 0.5))
    return fig


def parts_page(build, steps):
    """The full parts list, like the inventory page at the back of a booklet."""
    fig = _new_page()
    pieces = [p for s in steps for p in s["pieces"]]
    parts = _parts_in(pieces)
    fig.text(0.05, 0.9, "All the parts", fontsize=24, fontweight="bold", color=(0.12, 0.16, 0.25))
    per_row = 4
    for r in range(0, len(parts), per_row):
        row = parts[r:r + per_row]
        _parts_box(fig, (0.05, 0.62 - 0.3 * (r // per_row), 0.2 * len(row), 0.24), row)
    if any(m for *_, m in parts):
        n = sum(m for *_, m in parts)
        fig.text(0.05, 0.05, f"You're missing {n} piece{'s' if n > 1 else ''} (red counts).", fontsize=13, color=MISSING)
    return fig


def make_booklet(build, have=None, out_dir="booklets"):
    """Write <out_dir>/<build id>.pdf plus one PNG per page. Returns the PDF path and the PNG paths."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    steps = booklet_steps(build, have)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{build['id']}.pdf"
    pngs = []
    pages = [cover_page(build, steps)] + [step_page(build, steps, s["number"]) for s in steps]
    pages.append(parts_page(build, steps))
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(pages):
            pdf.savefig(fig)
            png = out / f"{build['id']}_page{i:02d}.png"
            fig.savefig(png, dpi=80)
            pngs.append(png)
            plt.close(fig)
    return pdf_path, pngs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("build", help="build id, e.g. house_hut")
    ap.add_argument("--have", help="JSON piece counts you own, to mark missing pieces")
    ap.add_argument("--out", default="booklets", help="folder for the PDF and page pictures")
    args = ap.parse_args()
    pdf, pngs = make_booklet(get_build(args.build), json.loads(args.have) if args.have else None, args.out)
    print(f"Wrote {pdf} and {len(pngs)} page pictures.")
