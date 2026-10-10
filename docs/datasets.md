# Public data worth using

| Source | What it has | How it helps BrickMatch |
|---|---|---|
| [Photos and rendered images of LEGO bricks](https://www.nature.com/articles/s41597-023-02682-2) (Gdańsk University of Technology, *Scientific Data* 2023) | Real photos and LDraw renders of many brick types, including multi-brick images with bounding boxes | Best match for detection. Pretrain on it, then fine-tune on your piles. Check which of its classes overlap `configs/pieces.yaml`. |
| [B200C LEGO Classification Dataset](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset) (Kaggle) | 200 part types, many rendered single-piece images | Classification only (no piles). Useful for a crop classifier or for sanity-checking class confusions. |
| [Images of LEGO Bricks](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images) (Kaggle) | ~40k renders of 50 bricks | Small and easy; good for a first PyTorch classification exercise before detection. |
| [Rebrickable downloads](https://rebrickable.com/downloads/) | Part numbers, colors, official set inventories as CSV | Source of truth for part ids and colors; mine real set inventories to grow `builds/builds.json`. |
| [LDraw parts library](https://www.ldraw.org) | 3D models of every part | Render your own synthetic piles (e.g. Blender) if you run short of real labeled images. |
| [awesome-lego-machine-learning](https://github.com/360er0/awesome-lego-machine-learning) | Curated list of LEGO ML datasets and projects | Look here for anything newer. |

Rendered images look cleaner than real photos, so expect a drop on your own piles. The point of evaluating on your own held-out pile photos is to measure exactly that gap.
Check each dataset's license before redistributing anything derived from it.
