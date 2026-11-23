"""Make the website's static data: web/generated/catalog.json plus a picture of every build and piece.

    python -m server.site_assets          # rebuild everything
    python -m server.site_assets --check  # only if something is missing (what the server does on start)

The site reads these directly, so browsing builds and matching typed-in counts work even while the API is
asleep. They come from builds/builds.json and configs/, so they're made fresh on every deploy, not committed.
"""
import argparse
import json
from pathlib import Path

import yaml

from brickmatch.instructions import DEFAULT_COLORS, SHAPES, SLOPES, make_steps
from brickmatch.match import load_builds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "generated"


def catalog():
    pieces = yaml.safe_load((ROOT / "configs/pieces.yaml").read_text())["pieces"]
    colors = yaml.safe_load((ROOT / "configs/colors.yaml").read_text())["colors"]
    builds = []
    for b in load_builds():
        builds.append({"id": b["id"], "name": b["name"], "category": b["category"],
                       "description": b.get("description", ""), "parts": b["parts"],
                       "total": sum(p["count"] for p in b["parts"]),
                       "steps": len(make_steps(b)) if b.get("layout") else 0})
    return {"pieces": [{**p, "icon": p["id"] in SHAPES, "color": DEFAULT_COLORS.get(p["id"])} for p in pieces],
            "colors": colors, "builds": builds}


def _save(fig, path):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig.savefig(path, dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_picture(build, path):
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from brickmatch.render import bounds, frame, render  # noqa: PLC0415
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0, 0, 1, 1))
    render(ax, build["layout"], edge_width=0.8)
    frame(ax, bounds(build["layout"]), pad=0.2)
    _save(fig, path)


def piece_picture(piece_id, path):
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from brickmatch.render import bounds, frame, render  # noqa: PLC0415
    p = {"piece": piece_id, "x": 0, "y": 0, "z": 0, "rot": 90 if piece_id in SLOPES else 0,
         "color": DEFAULT_COLORS.get(piece_id)}
    fig = plt.figure(figsize=(1.6, 1.2))
    ax = fig.add_axes((0, 0, 1, 1))
    render(ax, [p], edge_width=0.8)
    frame(ax, bounds([p]), pad=0.1)
    _save(fig, path)


def make_all():
    (OUT / "builds").mkdir(parents=True, exist_ok=True)
    (OUT / "pieces").mkdir(parents=True, exist_ok=True)
    cat = catalog()
    (OUT / "catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
    for b in load_builds():
        if b.get("layout"):
            build_picture(b, OUT / "builds" / f"{b['id']}.png")
    for pid in SHAPES:
        piece_picture(pid, OUT / "pieces" / f"{pid}.png")
    return OUT


def ensure():
    """Make the files if they aren't there yet (fast no-op otherwise)."""
    if not (OUT / "catalog.json").exists():
        make_all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only make the files if they're missing")
    args = ap.parse_args()
    ensure() if args.check else print(f"Wrote {make_all()}")
