# Collecting and labeling your own data

BrickMatch uses two models, so there are two kinds of labels, and both are cheap:

| Data | Used by | Label you add |
|---|---|---|
| Mixed pile photos | Stage 1 (find pieces) | A box around every piece, all with the same label `piece` |
| Single-type pile photos | Stage 2 (name pieces) | None per box: the whole photo is one piece type |

## Shooting
- Phone, top-down, roughly 30-60 cm above the table.
- Vary what a real user would vary: background (wood, white, carpet), lighting (daylight, lamp), pile density (spread out vs touching), phone angle.
- **Mixed piles** (`data/raw/my_piles/`): shuffle and shoot 3-5 photos per pile, named `pile_<NNN>_<k>.jpg`. Write the true counts for each pile into `data/raw/my_piles/counts.json` while the pieces are in front of you; that's the ground truth for counting.
- **Single-type piles** (`data/raw/single_type/<piece_id>/`): pour out only 2x4 bricks (any colors), shoot a few photos, repeat for each type. Run your stage 1 detector on them later and every crop is automatically labeled with that folder's type.

## Labeling boxes (stage 1)
- Tool: [Label Studio](https://labelstud.io) or [CVAT](https://www.cvat.ai), free and local. Export in **YOLO format** into `data/detect/images/<split>/` and `data/detect/labels/<split>/`.
- One tight box per visible piece, one class: `piece`. Include pieces outside the tier-1 types too, since the detector should find all of them.
- Mostly hidden pieces (under ~30% visible): box them anyway.
- Speed-up once a first detector exists: let it pre-draw boxes and only fix its mistakes.

## Making crops (stage 2)
- Crop each box to its own image and put it in `data/crops/<split>/<piece_id>/`.
- `train/` may mix public single-piece images (B200C, Gdańsk) with your own crops. Keep `test/` to your own real crops only.
- Pieces outside the vocabulary go in an `other/` folder so the classifier learns to set them aside.

## Splitting (important)
Split by **pile**, not by photo. All shots of pile 017 go into the same split, otherwise the test set contains pieces the model already saw and accuracy looks better than it is. Suggested 70 / 15 / 15, plus a few piles on a background you never trained on as a harder test.

## How much
Start with about 10 mixed piles and one single-type pile per piece type. Train, look at the errors, then collect more of what fails. A realistic full target is about 40 mixed piles x 4 photos.
