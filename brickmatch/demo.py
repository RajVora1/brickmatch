"""Web demo: upload a pile photo, see every piece boxed and named, fix the counts if the model got some wrong,
see what you can build,
a 3D view of a build you can turn around (missing pieces in red), and step-by-step instructions
with your pieces checked off.

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
from brickmatch.counts import HEADERS, changes, read_table, table
from brickmatch.instructions import draw_step, first_stuck_step, make_steps, step_text, step_title
from brickmatch.match import load_builds, rank_builds
from brickmatch.pipeline import detect_pieces, load_models
from brickmatch.viewer import embed as viewer_embed
from brickmatch.viewer import save as viewer_save

ROOT = Path(__file__).resolve().parent.parent
NAMES = {p["id"]: p["name"] for p in yaml.safe_load((ROOT / "configs/pieces.yaml").read_text())["pieces"]}


def counts_table(**kw):
    """Editable counts table: only "You have" can be changed (on Gradio versions that support it)."""
    import inspect  # noqa: PLC0415
    params = inspect.signature(gr.Dataframe.__init__).parameters
    if "static_columns" in params:
        kw["static_columns"] = [0, 1, 3]
    return gr.Dataframe(headers=HEADERS, datatype=["str", "number", "number", "str"], interactive=True, **kw)


def build_rows(counts):
    rows = []
    for b in rank_builds(counts, min_coverage=0.0)[:8]:
        missing = ", ".join(f"{n} x {NAMES.get(p, p)}" for p, n in b["missing"].items()) or "nothing, you can build it"
        rows.append([b["name"], b["category"], f"{b['coverage']:.0%}", missing])
    return rows


def edit_note(found_counts, have, from_photo):
    if not from_photo:
        return f"**Matching against the {sum(have.values())} pieces you typed in.**"
    changed = changes(found_counts, have)
    if not changed:
        return ("**Matching against the model's counts.** If it got a piece wrong, fix the **You have** column "
                "and press **Match builds with my counts**.")
    return (f"**Matching against your counts ({sum(have.values())} pieces).** You changed {len(changed)} piece "
            "type" + ("s" if len(changed) > 1 else "") + ":  \n" + "  \n".join(changed))


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
    """3D view, step pictures (with captions), a checklist and downloads for one build,
    checked against the detected counts."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    build = next((b for b in load_builds() if b["id"] == build_id), None)
    if build is None or not build.get("layout"):
        return "", [], "This build doesn't have step-by-step instructions yet.", None, None
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
    out = tempfile.mkdtemp(prefix="booklet_")
    booklet, _ = make_booklet(build, counts, out)
    page = viewer_save(build, counts, Path(out) / f"{build['id']}_3d.html")
    return viewer_embed(build, counts), gallery, "\n\n".join(lines), str(booklet), str(page)


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
        counts = dict(Counter(f["piece"] for f in found))
        shades = {piece: ", ".join(f"{k} {v}" for k, v in Counter(f["color"] for f in found
                                                                 if f["piece"] == piece).most_common())
                  for piece in counts}
        unsure = sum(f["score"] < 0.6 for f in found)
        summary = f"**Found {len(found)} pieces of {len(counts)} types.**"
        if unsure:
            summary += f" The model was unsure (under 60%) about {unsure} of them."
        summary += " Check the counts below and fix any it got wrong."
        rows = table(counts, shades)
        return (draw(img, found, colors), summary, rows, build_rows(counts), counts, counts, rows,
                edit_note(counts, counts, True))

    def apply(rows, found_counts):
        """Match builds against whatever is in the "You have" column."""
        try:
            have = read_table(rows)
        except ValueError as e:
            raise gr.Error(str(e)) from None
        return build_rows(have), have, edit_note(found_counts or {}, have, found_counts is not None)

    def undo(model_rows, found_counts):
        if model_rows is None:  # no photo yet: clear what was typed in
            return table({}), build_rows({}), None, "Counts cleared."
        return model_rows, build_rows(found_counts), dict(found_counts), edit_note(found_counts, found_counts, True)

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
        gr.Markdown("## Check your pieces\nThe model can get pieces wrong (a 1x4 brick seen from the side can look "
                    "like a 2x4). Fix the **You have** column before matching: to fix a wrong piece, take one off the "
                    "type the model said and add one to the right type. Set a count to 0 to remove a piece. No photo? "
                    "You can also just type in what you have.")
        counts_tbl = counts_table(value=table({}), label="Piece counts (click a number in \"You have\" to change it)")
        with gr.Row():
            apply_btn = gr.Button("Match builds with my counts", variant="primary")
            undo_btn = gr.Button("Undo my changes")
        note = gr.Markdown()
        builds_tbl = gr.Dataframe(headers=["Build", "Type", "Pieces you have", "Missing"],
                                  label="What you can build (best first)", interactive=False)
        counts = gr.State(None)        # what builds are matched against (the user's counts once edited)
        found_counts = gr.State(None)  # what the model found, to show what changed
        model_rows = gr.State(None)    # the table as the model filled it, for Undo
        gr.Markdown("## Your build in 3D and step by step\nDrag the 3D view to turn it, scroll to zoom. Pieces you're "
                    "missing are red, see-through and pulsing; the Steps slider builds it up one layer at a time.")
        with gr.Row():
            pick = gr.Dropdown(with_steps, value=with_steps[0][1] if with_steps else None, label="Build")
            show = gr.Button("Show build")
        view3d = gr.HTML()
        gr.Markdown("### Step-by-step instructions\nOne step per layer. Each picture shows the build so far (left) "
                    "and a map of the new layer seen from above (right), with columns counted from the left and rows "
                    "from the front. The numbers match the list. White pieces are from earlier steps, colored ones are "
                    "added in this step, and red see-through ones are pieces you're missing.")
        step_pics = gr.Gallery(label="Steps (click one to see it bigger)", columns=2, height="auto",
                               object_fit="contain")
        with gr.Row():
            booklet = gr.File(label="LEGO-style booklet (PDF) for this build")
            page3d = gr.File(label="3D view as a web page (open in any browser)")
        checklist = gr.Markdown()
        build_outputs = [view3d, step_pics, checklist, booklet, page3d]
        go.click(run, [photo, conf], [boxed, summary, counts_tbl, builds_tbl, counts, found_counts, model_rows, note]
                 ).then(best_with_steps, [counts, pick], pick).then(instructions, [pick, counts], build_outputs)
        apply_btn.click(apply, [counts_tbl, found_counts], [builds_tbl, counts, note]).then(
            best_with_steps, [counts, pick], pick).then(instructions, [pick, counts], build_outputs)
        undo_btn.click(undo, [model_rows, found_counts], [counts_tbl, builds_tbl, counts, note]).then(
            best_with_steps, [counts, pick], pick).then(instructions, [pick, counts], build_outputs)
        show.click(instructions, [pick, counts], build_outputs)
        pick.input(instructions, [pick, counts], build_outputs)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--examples", nargs="*", default=[], help="sample photos to show under the upload box")
    ap.add_argument("--share", action="store_true", help="make a public gradio.live link")
    args = ap.parse_args()
    make_app(load_models(args.detector, args.classifier), args.examples).launch(share=args.share)
