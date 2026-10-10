"""Stage 1: fine-tune YOLO to find every LEGO piece as one generic class ("piece").

The detector does not decide piece type; stage 2 (train_classifier.py) does that on crops.
So pile photos only need boxes, all with the same label.

Expected layout (YOLO format, see docs/labeling-guide.md):
    data/detect/images/{train,val,test}/*.jpg
    data/detect/labels/{train,val,test}/*.txt   # one line per box: "0 cx cy w h" (normalized)
    configs/detect.yaml                          # dataset file pointing at the above

    python train/train_detector.py --epochs 100
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint; try yolo11s.pt if recall is low")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024, help="piles have many small pieces, so go above the 640 default")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=str(ROOT / "configs/detect.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                single_cls=True, max_det=500, project=str(ROOT / "runs"), name="detector")
    # Recall matters most: a missed piece can never be counted.
    metrics = model.val(split="test", single_cls=True, imgsz=args.imgsz, max_det=500)
    print(f"test mAP@0.5 {metrics.box.map50:.3f}  recall {metrics.box.mr:.3f}  precision {metrics.box.mp:.3f}")


if __name__ == "__main__":
    main()
