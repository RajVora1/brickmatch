"""Web demo: upload a pile photo, see every piece boxed and named, the counts, and what you can build.

    python -m brickmatch.demo --detector detector_best.pt \
        --classifier runs/classifier_resnet18_1.0_pad_bal.pt --share

--share prints a public gradio.live link (valid for 72 hours, while the program keeps running).
--examples adds clickable sample photos, e.g. data/raw/my_piles/pile_010_*.jpg.
"""
import argparse
import colorsys
from collections import Counter
from pathlib import Path

import gradio as gr
import yaml
from PIL import ImageDraw, ImageFont, ImageOps

from brickmatch.match import rank_builds
from brickmatch.pipeline import detect_pieces, load_models

ROOT = Path(__file__).resolve().parent.parent
NAMES = {p["id"]: p["name"] for p in yaml.safe_load((ROOT / "configs/pieces.yaml").read_text())["pieces"]}


def box_colors(classes):
    """One clearly different outline color per piece type."""
    out = {}
    for i, c in enumerate(classes):
        r, g, b = colorsys.hsv_to_rgb(i / len(classes), 0.55, 1.0)
        out[c] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def draw(img, found, colors):
    img = img.copy()
    d = ImageDraw.Draw(img)
    line = max(2, img.width // 300)
    try:
        font = ImageFont.load_default(size=max(14, img.width // 60))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for f in found:
        x1, y1, x2, y2 = f["box"]
        c = colors[f["piece"]]
        d.rectangle((x1, y1, x2, y2), outline=c, width=line)
        text = f"{f['piece']} {f['score']:.0%}"
        tx1, ty1, tx2, ty2 = d.textbbox((x1, y1), text, font=font)
        top = max(0, y1 - (ty2 - ty1) - 4)
        d.rectangle((x1, top, x1 + tx2 - tx1 + 6, top + ty2 - ty1 + 4), fill=c)
        d.text((x1 + 3, top + 2 - (ty1 - y1)), text, fill="black", font=font)
    return img


def make_app(models, examples=()):
    colors = box_colors(models[2])

    def run(img, conf):
        if img is None:
            raise gr.Error("Upload a photo of your pieces first.")
        img = ImageOps.exif_transpose(img).convert("RGB")
        found = detect_pieces(img, models, conf)
        counts = Counter(f["piece"] for f in found)
        count_rows = []
        for piece, n in counts.most_common():
            shades = Counter(f["color"] for f in found if f["piece"] == piece)
            count_rows.append([NAMES.get(piece, piece), n, ", ".join(f"{k} {v}" for k, v in shades.most_common())])
        build_rows = []
        for b in rank_builds(dict(counts), min_coverage=0.0)[:8]:
            missing = ", ".join(f"{n} x {NAMES.get(p, p)}" for p, n in b["missing"].items()) or "nothing, you can build it"
            build_rows.append([b["name"], b["category"], f"{b['coverage']:.0%}", missing])
        unsure = sum(f["score"] < 0.6 for f in found)
        summary = f"**Found {len(found)} pieces of {len(counts)} types.**"
        if unsure:
            summary += f" The model was unsure (under 60%) about {unsure} of them."
        return draw(img, found, colors), summary, count_rows, build_rows

    with gr.Blocks(title="BrickMatch") as app:
        gr.Markdown("# BrickMatch\nPhoto of loose LEGO pieces in, what you can build with them out. "
                    "Works best on a plain background with pieces not piled on top of each other.")
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Your pieces", sources=["upload", "webcam"])
                conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Detector confidence threshold")
                go = gr.Button("Find my pieces", variant="primary")
                if examples:
                    gr.Examples(examples=[[str(e)] for e in examples], inputs=photo)
            with gr.Column():
                boxed = gr.Image(type="pil", label="What the model sees")
                summary = gr.Markdown()
        counts_tbl = gr.Dataframe(headers=["Piece", "Count", "Colors"], label="Piece counts", interactive=False)
        builds_tbl = gr.Dataframe(headers=["Build", "Type", "Pieces you have", "Missing"],
                                  label="What you can build (best first)", interactive=False)
        go.click(run, [photo, conf], [boxed, summary, counts_tbl, builds_tbl])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--examples", nargs="*", default=[], help="sample photos to show under the upload box")
    ap.add_argument("--share", action="store_true", help="make a public gradio.live link")
    args = ap.parse_args()
    make_app(load_models(args.detector, args.classifier), args.examples).launch(share=args.share)
