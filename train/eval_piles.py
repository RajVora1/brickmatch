"""End-to-end check: count the pieces in each pile photo and compare with data/raw/my_piles/counts.json.

    python train/eval_piles.py --detector runs/detector/weights/best.pt \
        --classifier runs/classifier_resnet18_1.0_pad_bal.pt

The detector trained on piles 1-6 and 8, so only val (7, 9) and test (10) are fair for the whole pipeline.
The classifier never saw a pile, so for it every pile is new.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickmatch.pipeline import count_pieces, load_models  # noqa: E402

SPLIT = {7: "val", 9: "val", 10: "test"}  # every other pile is detector training data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--piles", default=str(ROOT / "data/raw/my_piles"))
    args = ap.parse_args()

    piles = Path(args.piles)
    truth = json.loads((piles / "counts.json").read_text())
    models = load_models(args.detector, args.classifier)
    totals = {}
    for name, want in sorted(truth.items()):
        split = SPLIT.get(int(re.search(r"pile_(\d{3})", name).group(1)), "train")
        got, _ = count_pieces(piles / name, None, None, models=models)
        types = sorted(set(want) | set(got))
        # A piece counts as matched when both sides have it; extras and misses are the rest.
        matched = sum(min(want.get(t, 0), got.get(t, 0)) for t in types)
        n_want, n_got = sum(want.values()), sum(got.values())
        t = totals.setdefault(split, [0, 0, 0, 0])
        t[0] += matched; t[1] += n_want; t[2] += n_got; t[3] += got == want
        diff = ", ".join(f"{p} {want.get(p, 0)}->{got.get(p, 0)}" for p in types if want.get(p, 0) != got.get(p, 0))
        print(f"{name:16s} {split:5s} found {n_got:2d}/{n_want:2d}  {'exact' if got == want else diff}")
    print()
    for split, (matched, n_want, n_got, exact) in totals.items():
        print(f"{split:5s} pieces right {matched}/{n_want} ({matched / n_want:.0%}), "
              f"wrong or extra {n_got - matched}, photos counted exactly {exact}")


if __name__ == "__main__":
    main()
