"""Full pipeline: photo -> stage 1 boxes -> stage 2 piece types + colors -> counts -> ranked builds.

    python -m brickmatch.pipeline photo.jpg --detector runs/detector/weights/best.pt \
        --classifier runs/classifier_resnet18_1.0.pt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import models

from brickmatch.match import rank_builds

ROOT = Path(__file__).resolve().parent.parent
COLORS = yaml.safe_load((ROOT / "configs/colors.yaml").read_text())["colors"]


def nearest_color(crop):
    """Median pixel of the box's central area, matched to the closest supported color."""
    w, h = crop.size
    center = crop.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).convert("RGB")
    px = torch.tensor(list(center.getdata()), dtype=torch.float32)
    median = px.median(dim=0).values
    return min(COLORS, key=lambda c: ((median - torch.tensor(c["rgb"], dtype=torch.float32)) ** 2).sum())["id"]


def load_classifier(path):
    from train.train_classifier import ARCHS, make_transforms  # noqa: PLC0415
    ckpt = torch.load(path, map_location="cpu")
    build, _, set_head = ARCHS[ckpt["arch"]]
    model = build(weights=None)
    set_head(model, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["classes"], make_transforms(False)


def count_pieces(image_path, detector_path, classifier_path, conf=0.25):
    from ultralytics import YOLO  # noqa: PLC0415
    img = Image.open(image_path).convert("RGB")
    boxes = YOLO(detector_path).predict(img, conf=conf, max_det=500, verbose=False)[0].boxes.xyxy.tolist()
    model, classes, tf = load_classifier(classifier_path)
    by_type, by_type_color = Counter(), Counter()
    with torch.no_grad():
        for x1, y1, x2, y2 in boxes:
            crop = img.crop((x1, y1, x2, y2))
            piece = classes[model(tf(crop).unsqueeze(0)).argmax(1).item()]
            by_type[piece] += 1
            by_type_color[f"{piece}/{nearest_color(crop)}"] += 1
    return dict(by_type), dict(by_type_color)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    args = ap.parse_args()
    by_type, by_type_color = count_pieces(args.image, args.detector, args.classifier)
    print(json.dumps({"counts": by_type, "counts_by_color": by_type_color,
                      "builds": rank_builds(by_type, min_coverage=0.5)}, indent=2))
