# BrickMatch: week 3 plan (Thu 2026-10-08 to Wed 2026-10-14)

Five things to finish by Wednesday 2026-10-14, 11:59 pm. They're ordered so each one feeds the next, and none needs a trained model yet.

| # | Task | Rough time | Done when |
|---|---|---|---|
| 1 | Finalize the piece list | 30 min | `configs/pieces.yaml` matches the pieces you actually own |
| 2 | Photograph single-type piles | 1-2 h | One folder of photos per tier-1 piece in `data/raw/single_type/` |
| 3 | Photograph mixed piles + box-label a few | 3-4 h | ~10 piles shot, true counts written down, 5+ photos boxed |
| 4 | Get Colab working and train a warm-up classifier | 2-3 h | A ResNet-18 trained on a public LEGO dataset, with an accuracy number |
| 5 | Finish the week 3 progress-log entry | 15 min | Week 3 entry in the README's Progress log has this week's numbers |

Suggested order: 1 and 2 this weekend (Sat/Sun), 4 on a weeknight (it runs in the background while you do 3), 3 spread over the week, 5 on Wednesday.

---

## 1. Finalize the piece list

**Why:** every later step (photo folders, crop folders, the classifier's classes) uses these ids. Changing them later means renaming folders.

1. Dump your LEGO out and sort it roughly by type. For each of the 8 tier-1 pieces in `configs/pieces.yaml`, count how many you own.
2. Keep a piece in tier 1 only if you have **at least ~10** of it (in any colors). Fewer than that and you can't make enough training photos.
3. If you own lots of a piece that isn't on the list (say 1x6 bricks, or 1x1 round plates), you can swap it in. Find its part number by searching the piece on [Rebrickable](https://rebrickable.com) or [BrickLink](https://www.bricklink.com) (the number is printed on the part page, e.g. 2x4 brick = 3001). Add a line in the same format:
   ```yaml
   - {id: brick_1x6, part: "3009", name: "Brick 1 x 6", tier: 1}
   ```
4. Check `configs/colors.yaml` the same way: are most of your pieces in those 8 colors? Add a color if you have a lot of it (Rebrickable lists the RGB value on each color's page).
5. Don't go above ~15 tier-1 types. Fewer, distinct types beat many look-alikes.

## 2. Photograph single-type piles

**Why:** these are free labels for the classifier. Every piece in a "2x4 only" photo is a 2x4, so once the detector exists, all its crops are labeled automatically.

1. Make one folder per tier-1 id: `data/raw/single_type/brick_2x4/`, `data/raw/single_type/plate_2x2/`, and so on. The folder name must match the `id` in `pieces.yaml` exactly.
2. Pour out **only** that piece type, mixed colors, 10-30 pieces.
3. Shoot top-down with your phone, 30-60 cm above the table. Take **5-8 photos**, and between each: shuffle the pieces, flip some upside down and onto their sides, and change one thing (background, light, height). Aim for 2-3 backgrounds overall (wood table, white paper, carpet/towel).
4. Name them `brick_2x4_01.jpg`, `brick_2x4_02.jpg`, ... and copy them into the folder.
5. Repeat for every tier-1 piece. 12 pieces x 6 photos = ~70 photos, about an hour.

Tips: avoid harsh shadows (daylight near a window is best), keep the whole pile in frame, and don't zoom (move the phone instead).

## 3. Photograph mixed piles and start labeling boxes

**Why:** this is the stage 1 (detector) data and, later, your real test set. The full method is in `docs/labeling-guide.md`; this is the short version.

### 3a. Shoot ~10 mixed piles
1. Grab a random handful (20-60 pieces, mix of types, include a few pieces that *aren't* on your list).
2. **Before shooting, count it.** Write the true count per piece id into `data/raw/my_piles/counts.json`:
   ```json
   {
     "pile_001": {"brick_2x4": 4, "brick_2x2": 6, "plate_1x2": 3, "slope45_2x2": 2},
     "pile_002": {"brick_1x2": 8, "window_1x2x2": 1, "door_frame_1x4x6": 1}
   }
   ```
   Only count the tier-1 types; ignore the others.
3. Shoot 3-5 photos of that pile, shuffling between shots. Name them `pile_001_1.jpg`, `pile_001_2.jpg`, ...
4. Vary piles: some spread out, some touching/overlapping, different backgrounds and lights. Do 2 of the 10 on a background you won't use anywhere else (that's your "hard" test later).

### 3b. Label boxes on 5-10 photos with Label Studio
1. Install and start it on your computer (needs Python):
   ```
   pip install label-studio
   label-studio start
   ```
   It opens in your browser at `http://localhost:8080`. Make an account (local only).
2. **Create Project** -> name it `brickmatch-detect` -> **Labeling Setup** -> pick **Computer Vision > Object Detection with Bounding Boxes**. Delete the example labels and add exactly one label: `piece`. Save.
3. **Import** your `pile_*.jpg` photos.
4. For each photo: press `1` (selects the `piece` label), drag a tight box around a piece, repeat for every piece you can see, including off-list ones and half-hidden ones. Click **Submit**.
5. When you've done 5-10 photos: **Export** -> **YOLO**. You get a zip with `images/`, `labels/`, and `classes.txt` (which should say only `piece`).
6. Put them in the repo split **by pile** (all photos of one pile go in the same split):
   - piles 1-6 and 8 -> `data/detect/images/train/` and `data/detect/labels/train/`
   - piles 7 and 9 -> `.../val/`
   - pile 10 -> `.../test/`

   The Colab cell in `docs/box-labeling.md` does this sorting for you.

   Each `.jpg` must have a `.txt` with the same name in the matching `labels/` folder.

Expect ~2-5 minutes per photo at first. You don't need all of them boxed this week. Once a first detector exists it will pre-draw boxes and you'll only fix mistakes.

## 4. Get Colab working with a dataset download

**Why:** proves your training setup works end to end before your own data is ready, and gives you a first real number for the log.

We'll use the small Kaggle **Images of LEGO Bricks** set (~40k renders, 50 types). B200C is the bigger one for later; it's several GB, so download it to Google Drive once you need it, not this week.

### 4a. One-time setup
1. **Kaggle API key:** on [kaggle.com](https://www.kaggle.com) -> your profile -> **Settings** -> **API** -> **API Tokens** tab -> **Generate New Token**. Copy the `KGAT_...` token it shows.
2. **GitHub token** (the repo is private, so Colab needs permission to clone it): GitHub -> **Settings** -> **Developer settings** -> **Personal access tokens** -> **Fine-grained tokens** -> **Generate new token**, give it access to only `brickmatch` with **Contents: Read-only**.
3. Open [colab.research.google.com](https://colab.research.google.com) -> **New notebook**. Then **Runtime -> Change runtime type -> T4 GPU**.
4. Click the **key icon** (Secrets) in the left sidebar and add two secrets, switching on "Notebook access" for each: `KAGGLE_API_TOKEN` (your `KGAT_...` token) and `GH_TOKEN`. This keeps them out of the notebook text.

### 4b. Notebook cells
Run these one cell at a time.

```python
# Cell 1: secrets, clone the repo, install
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
!git clone https://{userdata.get('GH_TOKEN')}@github.com/RajVora1/brickmatch.git
%cd brickmatch
!pip install -q -r requirements.txt && pip install -q -U kaggle
```

```python
# Cell 2: download the dataset (a minute or two)
!kaggle datasets download -d joosthazelzet/lego-brick-images -p /content/raw --unzip
!find /content/raw -maxdepth 3 -type d | head -60
```

Look at the printed folder names. The images are grouped by part, and the part number (like `3001`, `3003`) is in the folder or file name.

```python
# Cell 3: keep only our tier-1 parts, split 70/15/15 into data/crops/<split>/<piece_id>/
import random, re, shutil, yaml
from pathlib import Path

pieces = yaml.safe_load(open("configs/pieces.yaml"))["pieces"]
part_to_id = {p["part"]: p["id"] for p in pieces if p["tier"] == 1}

by_id = {}
for img in Path("/content/raw").rglob("*"):
    if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    # part number = first standalone run of digits in the folder name, else the file name
    for text in (img.parent.name, img.name):
        m = re.match(r"\D*?(\d{4,5})\b", text)
        if m and m.group(1) in part_to_id:
            by_id.setdefault(part_to_id[m.group(1)], []).append(img)
            break

random.seed(0)
for pid, imgs in by_id.items():
    random.shuffle(imgs)
    n = len(imgs)
    splits = {"train": imgs[: int(n * .7)], "val": imgs[int(n * .7): int(n * .85)], "test": imgs[int(n * .85):]}
    for split, files in splits.items():
        out = Path("data/crops") / split / pid
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, out / f.name)
    print(f"{pid:20s} {n} images")
print("tier-1 parts with no images in this dataset:", sorted(set(part_to_id.values()) - set(by_id)))
```

Not every tier-1 part is in this dataset (the window and door frame probably aren't). That's fine for a warm-up: the classifier just trains on the ones it finds.

```python
# Cell 4: train ResNet-18 (about 10-20 min on a T4)
!python train/train_classifier.py --arch resnet18 --epochs 5
```

You'll see `val acc` each epoch and a final `test acc`. Write both down for the log.

```python
# Cell 5: save the model to Drive so it survives the session
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/brickmatch && cp -r runs /content/drive/MyDrive/brickmatch/
```

**Important caveat for your write-up:** this test set is rendered images, not your photos, so the accuracy will look high (probably 90%+). It only proves the pipeline works. The real test comes later on crops from your own piles, where you should expect it to drop. Measuring that drop is part of the project.

Optional if you finish early: open `runs/classifier_resnet18_1.0.confusion.json` and see which pieces get mixed up (likely 1x2 vs 1x4 bricks, or bricks vs plates).

## 5. Finish the week 3 progress-log entry

By Wednesday 2026-10-14, 11:59 pm, update the week 3 entry at the top of the README's Progress log so it covers the whole week. Weeks run Thursday to Wednesday, so week 4 starts Thursday 2026-10-15 with a new entry. Roughly:

```markdown
### Week 3 (Thu 2026-10-08 to Wed 2026-10-14)
- **Done:** Finalized piece list (N tier-1 types). Shot single-type piles for N types (N photos). Shot N mixed piles (N photos) with true counts; boxed N photos. Set up Colab and trained a warm-up ResNet-18 on the Kaggle LEGO renders.
- **Results:** Warm-up classifier: val acc X, test acc X on N classes (rendered images, so optimistic). Dataset so far: N single-type photos, N mixed photos, N boxes.
- **Blockers:** (anything that slowed you down, e.g. labeling speed, missing parts in the public set)
- **Next:** Train the first YOLO detector on the boxed photos; finish boxing; download B200C.
```

Fill in real numbers, even small ones. Mentors care more about honest numbers than big ones. To publish it: edit `README.md` on github.com (pencil icon) and commit, or tell Claude the numbers and it'll update the README for you.

---

## If you get stuck
- **Kaggle 401/403:** the secret name must be exactly `KAGGLE_API_TOKEN`, with Notebook access switched on. You may also need to click "Download" once on the dataset page to accept its terms.
- **git clone asks for a password / 404:** the GitHub token is wrong or doesn't include the `brickmatch` repo.
- **`every split needs the same piece folders`:** one piece ended up with too few images to appear in every split. Delete that piece's folder from all three splits and rerun.
- **Colab disconnects:** normal on the free tier. Rerun from Cell 1; your model is safe if you ran Cell 5.
