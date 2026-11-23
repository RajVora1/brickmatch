"""The BrickMatch website's API: the part of the site that needs Python and the trained models.

    uvicorn server.app:app --port 7860       # then open http://localhost:7860

The website itself (web/) is plain HTML/JS. It can be served from anywhere (GitHub Pages), and it calls this API
for the things a browser can't do: finding pieces in a photo, drawing step pictures, making booklet PDFs and
3D views. This server also serves web/ itself, so one address is enough when running it on your own.

Endpoints
    GET  /api/health                     is the server up, are the models loaded
    POST /api/detect   (form: photo, conf)  every piece in the photo: box, type, confidence, color
    POST /api/build    {"id", "counts"}  steps (text + pictures), 3D view page, what's missing
    POST /api/booklet  {"id", "counts"}  LEGO-style booklet PDF
"""
import base64
import io
import json
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from brickmatch.instructions import first_stuck_step, make_steps, step_text, step_title
from brickmatch.match import load_builds
from server import site_assets
from server.models import Models

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MAX_UPLOAD = 15 * 1024 * 1024  # bytes
MAX_SIDE = 2048  # photos are shrunk to this many pixels on the long side (the detector looks at 1024)
MAX_COUNT = 999

@asynccontextmanager
async def lifespan(_app):
    MODELS.start()
    site_assets.ensure()  # web/generated/ (catalog + build pictures) when running locally
    yield


app = FastAPI(title="BrickMatch API", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
origins = [o.strip() for o in os.environ.get("BRICKMATCH_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

MODELS = Models()
BUILDS = {b["id"]: b for b in load_builds()}


@app.get("/api/health")
def health():
    return {"status": "ok", "models": MODELS.state, "message": MODELS.message, "classes": MODELS.classes}


@app.post("/api/detect")
def detect(photo: UploadFile = File(...), conf: float = Form(0.5)):
    if MODELS.state != "ready":
        raise HTTPException(503, MODELS.message)
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(400, "conf has to be between 0.05 and 0.95.")
    data = photo.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is too big (over 15 MB).")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "That file isn't a photo we can read. Try a JPG or PNG.") from None
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    from brickmatch.pipeline import detect_pieces  # noqa: PLC0415
    t0 = time.time()
    with MODELS.lock:
        found = detect_pieces(img, MODELS.models, conf)
    pieces = [{"box": [round(v, 1) for v in f["box"]], "piece": f["piece"], "score": round(f["score"], 3),
               "color": f["color"]} for f in found]
    return {"width": img.width, "height": img.height, "pieces": pieces,
            "counts": dict(Counter(f["piece"] for f in found)), "seconds": round(time.time() - t0, 2)}


class BuildRequest(BaseModel):
    id: str
    counts: dict[str, int] | None = None  # None: don't check pieces off


def _build_and_counts(req):
    build = BUILDS.get(req.id)
    if build is None:
        raise HTTPException(404, f"There's no build called {req.id}.")
    if not build.get("layout"):
        raise HTTPException(404, "This build doesn't have step-by-step instructions yet.")
    counts = req.counts
    if counts is not None:
        if any(n < 0 or n > MAX_COUNT for n in counts.values()) or len(counts) > 50:
            raise HTTPException(400, "Counts have to be whole numbers from 0 to 999.")
        counts = {k: v for k, v in counts.items() if v > 0}
    return build, counts


def _key(counts):
    return None if counts is None else json.dumps(counts, sort_keys=True)


@lru_cache(maxsize=256)
def _step_png(build_id, counts_key, number):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from brickmatch.instructions import draw_step  # noqa: PLC0415
    build = BUILDS[build_id]
    steps = make_steps(build, None if counts_key is None else json.loads(counts_key))
    fig = draw_step(build, steps, number)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/build")
def build_info(req: BuildRequest):
    """Everything the build page shows: the 3D view, and each step as a picture plus a numbered piece list."""
    from brickmatch.viewer import page  # noqa: PLC0415
    build, counts = _build_and_counts(req)
    steps = make_steps(build, counts)
    key = _key(counts)
    out = []
    for s in steps:
        out.append({"number": s["number"], "title": step_title(s), "lines": step_text(s),
                    "missing": sum(p["owned"] is False for p in s["pieces"]),
                    "image": "data:image/png;base64," + _step_png(build["id"], key, s["number"])})
    stuck = first_stuck_step(steps) if counts is not None else None
    return {"id": build["id"], "name": build["name"], "steps": out, "stuck": stuck,
            "missing": sum(st["missing"] for st in out), "checked": counts is not None,
            "viewer": page(build, counts, height=460)}


@app.post("/api/booklet")
def booklet(req: BuildRequest):
    import tempfile  # noqa: PLC0415
    from brickmatch.booklet import make_booklet  # noqa: PLC0415
    build, counts = _build_and_counts(req)
    with tempfile.TemporaryDirectory() as out:
        pdf, _ = make_booklet(build, counts, out)
        data = Path(pdf).read_bytes()
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{build["id"]}_booklet.pdf"'})


# The website itself. On GitHub Pages this part isn't used; the Pages copy calls the API above.
if WEB.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/", StaticFiles(directory=WEB), name="web")
