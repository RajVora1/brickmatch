"""Checks on the website's API (server/app.py) and its static files, without real model weights.

    python -m pytest tests/test_site.py
"""
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server import app as server  # noqa: E402
from server import site_assets  # noqa: E402

FAKE = [{"box": [10.0, 10.0, 60.0, 40.0], "piece": "brick_2x4", "score": 0.93, "color": "red"},
        {"box": [80.0, 20.0, 100.0, 40.0], "piece": "brick_1x1", "score": 0.55, "color": "yellow"}]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server.MODELS, "start", lambda: None)
    with TestClient(server.app) as c:
        yield c


def photo_bytes(size=(320, 240)):
    import io  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, "JPEG")
    return buf.getvalue()


def test_catalog_matches_builds_file():
    cat = site_assets.catalog()
    assert len(cat["builds"]) == len(server.BUILDS)
    for b in cat["builds"]:
        assert b["total"] == sum(p["count"] for p in b["parts"])
        assert b["steps"] > 0  # every build has a layout
    assert {p["id"] for p in cat["pieces"] if p["icon"]} >= {p["piece"] for b in cat["builds"] for p in b["parts"]}


def test_health_says_when_models_are_missing(client, monkeypatch):
    monkeypatch.setattr(server.MODELS, "state", "missing")
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models"] == "missing"
    assert client.post("/api/detect", files={"photo": ("p.jpg", photo_bytes(), "image/jpeg")}).status_code == 503


def test_detect_counts_pieces(client, monkeypatch):
    import brickmatch.pipeline as pipeline  # noqa: PLC0415
    monkeypatch.setattr(server.MODELS, "state", "ready")
    monkeypatch.setattr(server.MODELS, "models", object())
    seen = {}

    def fake_detect(img, models, conf):
        seen["size"], seen["conf"] = img.size, conf
        return FAKE

    monkeypatch.setattr(pipeline, "detect_pieces", fake_detect)
    r = client.post("/api/detect", files={"photo": ("p.jpg", photo_bytes((4000, 3000)), "image/jpeg")},
                    data={"conf": "0.4"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["counts"] == {"brick_2x4": 1, "brick_1x1": 1}
    assert (d["width"], d["height"]) == seen["size"] == (2048, 1536)  # big photos are shrunk
    assert seen["conf"] == pytest.approx(0.4)


def test_detect_rejects_non_photos(client, monkeypatch):
    monkeypatch.setattr(server.MODELS, "state", "ready")
    r = client.post("/api/detect", files={"photo": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_build_marks_missing_pieces(client):
    r = client.post("/api/build", json={"id": "house_hut", "counts": {"brick_1x4": 4, "brick_1x2": 4}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["checked"] and d["missing"] > 0 and d["stuck"] is not None
    assert all(s["image"].startswith("data:image/png;base64,") for s in d["steps"])
    assert "<canvas" in d["viewer"] or "three" in d["viewer"]
    full = client.post("/api/build", json={"id": "house_hut", "counts": None}).json()
    assert full["missing"] == 0 and full["stuck"] is None and not full["checked"]


def test_build_errors(client):
    assert client.post("/api/build", json={"id": "nope"}).status_code == 404
    assert client.post("/api/build", json={"id": "house_hut", "counts": {"brick_1x4": -1}}).status_code == 400


def test_booklet_is_a_pdf(client):
    r = client.post("/api/booklet", json={"id": "tower_simple", "counts": {"brick_2x2": 3}})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"


def test_site_is_served(client):
    assert "BrickMatch" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    json.loads(client.get("/samples/samples.json").text)
