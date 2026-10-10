"""Evaluate counting accuracy and build-recommendation correctness on held-out piles.

Two questions from the project plan:
  1. How accurately does the model count pieces by type on unseen photos?
  2. Are the recommended builds actually buildable from what is really in the pile?

Inputs are per-image count dicts, e.g. {"pile_017.jpg": {"brick_2x4": 3, ...}}:
    python eval/evaluate.py predicted_counts.json true_counts.json

(Stage 1 mAP/recall comes from train/train_detector.py; stage 2 accuracy and the confusion matrix
from train/train_classifier.py. This script scores the whole pipeline.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brickmatch.match import rank_builds, score_build, load_builds  # noqa: E402


def count_metrics(pred, true):
    types = set(pred) | set(true)
    abs_err = sum(abs(pred.get(t, 0) - true.get(t, 0)) for t in types)
    total = max(sum(true.values()), 1)
    exact = all(pred.get(t, 0) == true.get(t, 0) for t in types)
    return {"abs_err": abs_err, "rel_err": abs_err / total, "exact": exact}


def main(pred_path, true_path):
    preds, trues = json.loads(Path(pred_path).read_text()), json.loads(Path(true_path).read_text())
    builds = {b["id"]: b for b in load_builds()}
    rows, rec_ok, rec_total = [], 0, 0
    for img, true in trues.items():
        pred = preds.get(img, {})
        rows.append(count_metrics(pred, true))
        # A recommendation the model calls fully buildable should be fully buildable in reality.
        for r in rank_builds(pred, min_coverage=1.0):
            rec_total += 1
            rec_ok += score_build(builds[r["id"]], true)[0] == 1.0
    n = len(rows)
    print(f"images:                 {n}")
    print(f"mean abs count error:   {sum(r['abs_err'] for r in rows) / n:.2f} pieces/image")
    print(f"mean relative error:    {sum(r['rel_err'] for r in rows) / n:.1%}")
    print(f"exact-count images:     {sum(r['exact'] for r in rows) / n:.1%}")
    if rec_total:
        print(f"recommendation precision (buildable for real): {rec_ok / rec_total:.1%} of {rec_total}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
