"""Web demo: upload a pile photo, see every piece boxed and named, the counts, what you can build,
and step-by-step instructions for a build with your pieces checked off.

    python -m brickmatch.demo --detector detector_best.pt \
        --classifier runs/classifier_resnet18_1.0_pad_bal.pt --share

--share prints a public gradio.live link (valid for 72 hours, while the program keeps running).
--examples adds clickable sample photos, e.g. data/raw/my_piles/pile_010_*.jpg.
"""
import argparse
import colorsys
import io
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from brickmatch.booklet import make_booklet
from brickmatch.instructions import draw_step, first_stuck_step, make_steps, step_text, step_title
from brickmatch.match import load_builds, rank_builds
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


def instructions(build_id, counts):
    """Step pictures (with captions) and a checklist for one build, checked against the detected counts."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    build = next((b for b in load_builds() if b["id"] == build_id), None)
    if build is None or not build.get("layout"):
        return [], "This build doesn't have step-by-step instructions yet.", None
    steps = make_steps(build, counts)
    gallery = []
    for s in steps:
        fig = draw_step(build, steps, s["number"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        gallery.append((Image.open(buf).convert("RGB"), f"Step {s['number']}"))
    lines = [f"### {build['name']}: {len(steps)} steps"]
    if counts is None:
        lines.append("Find your pieces first to have them checked off.")
    else:
        stuck = first_stuck_step(steps)
        if stuck is None:
            lines.append("**You have every piece.**")
        elif stuck == 1:
            lines.append("**You're missing pieces from step 1 on.** Pieces drawn in red are the ones you're missing.")
        else:
            lines.append(f"**You can build up to step {stuck - 1}.** Pieces drawn in red are the ones you're missing.")
    for s in steps:
        lines.append(f"**{step_title(s)}**  \n" + "  \n".join(step_text(s)))
    booklet, _ = make_booklet(build, counts, tempfile.mkdtemp(prefix="booklet_"))
    return gallery, "\n\n".join(lines), str(booklet)


def best_with_steps(counts, current):
    """After a photo, switch to the best-matching build that has instructions."""
    with_layout = {b["id"] for b in load_builds() if b.get("layout")}
    ranked = [r["id"] for r in rank_builds(counts or {}, min_coverage=0.0) if r["id"] in with_layout]
    return ranked[0] if ranked else current


def make_app(models, examples=()):
    colors = box_colors(models[2])
    with_steps = [(b["name"], b["id"]) for b in load_builds() if b.get("layout")]

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
        return draw(img, found, colors), summary, count_rows, build_rows, dict(counts)

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
        counts = gr.State(None)
        gr.Markdown("## Step-by-step instructions\nOne step per layer. Each picture shows the build so far (left) "
                    "and a map of the new layer seen from above (right), with columns counted from the left and rows "
                    "from the front. The numbers match the list. White pieces are from earlier steps, colored ones are "
                    "added in this step, and red see-through ones are pieces you're missing.")
        with gr.Row():
            pick = gr.Dropdown(with_steps, value=with_steps[0][1] if with_steps else None, label="Build")
            show = gr.Button("Show steps")
        step_pics = gr.Gallery(label="Steps (click one to see it bigger)", columns=2, height="auto",
                               object_fit="contain")
        booklet = gr.File(label="LEGO-style booklet (PDF) for this build")
        checklist = gr.Markdown()
        go.click(run, [photo, conf], [boxed, summary, counts_tbl, builds_tbl, counts]).then(
            best_with_steps, [counts, pick], pick).then(instructions, [pick, counts], [step_pics, checklist, booklet])
        show.click(instructions, [pick, counts], [step_pics, checklist, booklet])
        pick.input(instructions, [pick, counts], [step_pics, checklist, booklet])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--examples", nargs="*", default=[], help="sample photos to show under the upload box")
    ap.add_argument("--share", action="store_true", help="make a public gradio.live link")
    args = ap.parse_args()
    make_app(load_models(args.detector, args.classifier), args.examples).launch(share=args.share)
