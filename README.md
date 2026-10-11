# BrickMatch

Take a photo of your loose LEGO pieces, find out what you can build with them.

```
photo ─> [stage 1: YOLO finds every piece] ─> crops ─> [stage 2: CNN / ViT names each piece] ─> counts by type + color ─> [matcher] ─> ranked builds + missing pieces
          trained model, one class "piece"              trained model, 12-15 piece types          color read from pixels      plain search
```

Two-stage design (mentor's advice): finding pieces and naming them are separate models. Pile photos then only need boxes with one label, and the piece-type model can learn from existing single-piece image sets. Only the two stages are ML; matching counts to parts lists is search.

## Layout

| Path | What goes there |
|---|---|
| `configs/pieces.yaml` | Piece vocabulary (ids, BrickLink part numbers, tier 1 vs 2) |
| `configs/colors.yaml` | Supported colors |
| `configs/detect.yaml` | Stage 1 dataset file (Ultralytics format) |
| `data/raw/public/` | Downloaded public datasets (see `docs/datasets.md`) |
| `data/raw/my_piles/` | Your mixed pile photos + `counts.json` ground truth |
| `data/raw/single_type/<piece_id>/` | Photos of piles holding only one piece type |
| `data/detect/` | Stage 1 data: `images/` + `labels/` per split, YOLO format |
| `data/crops/<split>/<piece_id>/` | Stage 2 data: one cropped piece per image |
| `builds/builds.json` | Builds database: towers, bridges, walls, houses with parts lists |
| `train/train_detector.py` | Stage 1: fine-tune YOLO as a one-class piece finder |
| `train/train_classifier.py` | Stage 2: train ResNet-18 or ViT-B/16 on crops, with a data-size option |
| `brickmatch/pipeline.py` | Photo in, counts and ranked builds out |
| `brickmatch/match.py` | Ranks builds by coverage and lists missing pieces |
| `eval/evaluate.py` | Counting error + "are recommended builds really buildable?" |
| `docs/` | Labeling guide, dataset notes |

## Starting piece list (recommendation, adjust freely)

8 tier-1 types, all well stocked in the LEGO Classic Large Creative Brick Box (10698) used for data collection:

- **Bricks:** 1x1, 1x2, 1x4, 2x2, 2x4
- **Plates:** 1x2
- **Slopes 45°:** 2x1, 2x2

Tier 2, add once tier 1 works:
- plate 2x2, plate 2x4, window 1x2x2, door frame 1x4x6: needed for houses, but 10698 has too few of them (a Pick a Brick top-up fixes that)
- brick 2x3, plate 1x4, tile 1x2: left out at first because they're the classic confusions

**Color is not part of the class.** The classifier predicts type; color comes from the median pixel inside each box, matched to `colors.yaml`.

## Plan

1. **Setup and warm-up**: PyTorch on Colab/Kaggle; train a classifier on Kaggle "Images of LEGO Bricks".
2. **Builds and matcher**: grow `builds.json` to ~20 builds. `python -m brickmatch.match '{"brick_2x2": 8}'` already works.
3. **Collect and label**: mixed piles (boxes only) and single-type piles (free type labels). See `docs/labeling-guide.md`.
4. **Stage 1**: `python train/train_detector.py`. Watch recall.
5. **Stage 2**: `python train/train_classifier.py --arch resnet18`, then `--arch vit_b_16`; repeat with `--train-fraction 0.25 / 0.5` to see how each scales with data.
6. **End to end**: `python -m brickmatch.pipeline photo.jpg --detector ... --classifier ...`, then `eval/evaluate.py` on held-out piles.
7. **Iterate and write up.**
8. **Stretch**: photo of a real object (house, bridge) -> classify into a build category -> recommend builds of that type.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from this folder. A GPU helps from step 4 on; Google Colab or Kaggle notebooks (free GPU) work fine.

## Progress log

Newest entry first. One entry per week. Weeks run Thursday to Wednesday, so each entry is due Wednesday 11:59 pm. Copy the template below.

```
### Week N (Thu YYYY-MM-DD to Wed YYYY-MM-DD)
- **Done:**
- **Results:** (numbers: accuracy, counting error, dataset size)
- **Blockers:**
- **Next:**
```

### Week 3 (Thu 2026-10-08 to Wed 2026-10-14)
- **Done:** Scoped the project to 12 common piece types and four build categories (towers, bridges, walls, houses). Switched to a two-stage design on my mentor's advice (2026-10-06): a YOLO model finds every piece as one generic class, then a crop classifier (ResNet-18, compared with ViT-B/16) names the type. Built the starter code: piece/color configs, builds database, matcher, pipeline, training scripts for both stages, evaluation script, labeling guide and dataset notes. Bought a LEGO Classic Large Creative Brick Box (10698) for my own photos and narrowed tier 1 to the 8 types it stocks well. Set up Google Colab and trained a warm-up ResNet-18 classifier on the Kaggle "Images of LEGO Bricks" renders.
- **Results:** Warm-up ResNet-18 (5 epochs): val acc 95.4%, test acc 96.3%. The test set is rendered images, not my photos, so this is optimistic; it mainly proves the training setup works. The matcher works end to end on hand-typed counts (`python -m brickmatch.match '{"brick_2x2": 8}'`).
- **Blockers:** None. Training will run on Colab (free GPU).
- **Next:** Photograph single-type and mixed piles once the LEGO box arrives, box-label the first mixed-pile photos, grow `builds.json` to ~20 builds.
