# Box-labeling your 10 piles (step by step)

**Goal:** draw a box around every piece in your 30 pile photos. These boxes teach the *Find* model (YOLO) what "a piece" looks like. Every box gets the same label, `piece`. You never pick the type here; the classifier does that later.

**Size of the job:** 113 pieces x 3 photos = about 340 boxes. At ~10 seconds a box, that's **about an hour**. You can split it over several sittings with Label Studio, because it saves as you go.

---

## Step 0: Get the renamed photos onto your computer

Download **`my_piles_photos.zip`** (attached in the chat) and unzip it. Inside `my_piles/` are `pile_001_1.jpg` ... `pile_010_3.jpg` plus `counts.json`.

Use these copies, **not** the `IMG_06xx` originals on your phone. The pile number in the filename is how the photos get split into train, val and test later.

## Step 1: Install and open Label Studio (5 min)

Label Studio needs **Python 3.12 or older**. It crashes on 3.14 with `cannot import name 'find_loader'`. If your Mac's Python is newer, install Python 3.12 from https://www.python.org/downloads/macos/ first; it sits alongside the newer one. Then in Terminal:

```
python3.12 -m venv ~/ls-env
source ~/ls-env/bin/activate
pip install label-studio
label-studio start
```

Next time, you only need the `source` line and `label-studio start`.

A browser tab opens at `http://localhost:8080`. Click **Sign up** and make an account. It only lives on your computer.

> If `pip` says "command not found", try `pip3` or `python -m pip`. If it still won't install, skip to **Plan B** at the bottom.

Leave the terminal window open while you label; closing it stops Label Studio. Next time, just run `label-studio start` again and your work will still be there.

## Step 2: Create the project (3 min)

1. Click **Create Project**. Name it `brickmatch-detect`.
2. Open the **Labeling Setup** tab and pick **Computer Vision > Object Detection with Bounding Boxes**.
3. On the right, delete the example labels (`Airplane`, `Car`). Type `piece` in the "Add label names" box and click **Add**. You should see exactly one label: `piece`.
4. Open the **Data Import** tab, drag in all 30 `pile_*.jpg` files (not `counts.json`), and click **Save**.

## Step 3: Draw the boxes (~1 hour)

Click **Label All Tasks**. For each photo:

1. Press **`1`** on the keyboard. This selects the `piece` label.
2. Click and drag a box around one piece.
3. Repeat steps 1-2 for every piece in the photo.
4. Check the box count in the **Regions** list on the right against `counts.json`. For example, pile 8 should have 24.
5. Click **Submit**. The next photo opens.

**How to draw good boxes**
- **Tight:** the box edges touch the piece's outermost edges, including studs sticking out on the side. Don't leave a gap of carpet around it.
- **Whole piece, including its side:** for a piece seen at an angle, include the visible side wall, not just the top.
- **Every piece gets a box**, even upside-down, lying on its side, or half cut off at the photo's edge.
- **Touching pieces get separate boxes.** In pile 10, the two 2x4s lying side by side are two boxes, and it's fine for the boxes to overlap.
- **Shadows aren't part of the piece.** Leave them outside the box.
- Made a mistake? Click the box and press **Backspace** to delete it, or drag its corners to resize it.

## Step 4: Export (2 min)

Go back to the project page (click `brickmatch-detect` at the top). Click **Export** and pick **YOLO with Images** (or **YOLO** if that's the only one), then click **Export**. You get a zip containing:

```
images/   the 30 photos (Label Studio adds a short code to the front of each name; that's fine)
labels/   one .txt per photo, one line per box: 0 x_center y_center width height
classes.txt   should contain just: piece
```

Open `classes.txt` and check it says only `piece`. If it lists `Airplane` or `Car`, the labels weren't deleted in Step 2.3.

## Step 5: Put it in Google Drive and split it (5 min)

1. In Google Drive, open your `brickmatch` folder (the one Cell 5 saves `runs/` into; make it if it isn't there). Upload the export zip there and rename it to **`labels_export.zip`**. Also upload **`my_piles_photos.zip`** to the same folder.
2. In Colab, run Cell 2 from `docs/colab-setup.md` as usual (clone + install). Then add and run this cell:

```python
# Mount Drive, unzip the Label Studio export, and split it BY PILE into data/detect/{train,val,test}.
from google.colab import drive; drive.mount('/content/drive')
import zipfile, re, shutil, pathlib

DRIVE = pathlib.Path('/content/drive/MyDrive/brickmatch')
SPLIT = {1:'train', 2:'train', 3:'train', 4:'train', 5:'train', 6:'train', 8:'train',
         7:'val', 9:'val',
         10:'test'}

tmp = pathlib.Path('/content/ls_export'); shutil.rmtree(tmp, ignore_errors=True)
zipfile.ZipFile(DRIVE/'labels_export.zip').extractall(tmp)
orig = pathlib.Path('/content/orig'); shutil.rmtree(orig, ignore_errors=True)
zipfile.ZipFile(DRIVE/'my_piles_photos.zip').extractall(orig)

out = pathlib.Path('/content/brickmatch/data/detect')
stats = {}
for lab in sorted(tmp.rglob('*.txt')):
    m = re.search(r'pile_(\d{3})_(\d)', lab.name)
    if not m: continue                     # skips classes.txt
    name = m.group(0)                      # e.g. pile_008_2 (drops Label Studio's prefix)
    split = SPLIT[int(m.group(1))]
    img = orig/'my_piles'/f'{name}.jpg'    # the original photo, so it always exists
    for sub, src, dst in (('images', img, f'{name}.jpg'), ('labels', lab, f'{name}.txt')):
        d = out/sub/split; d.mkdir(parents=True, exist_ok=True); shutil.copy(src, d/dst)
    n = len([l for l in lab.read_text().splitlines() if l.strip()])
    s = stats.setdefault(split, [0, 0]); s[0] += 1; s[1] += n
for split, (photos, boxes) in stats.items():
    print(f'{split}: {photos} photos, {boxes} boxes')
# Expect: train 21 photos, val 6 photos, test 3 photos.
```

If the numbers match (21 / 6 / 3), you're done. Next week the detector trains on this with `python train/train_detector.py`.

**Why this split:** all 3 photos of a pile stay together (see `docs/how-it-works.md`, "Why split by pile matters"). Train gets piles 1-6 and 8 (every background, including the big bedsheet pile). Val gets 7 and 9. Test gets pile 10, the only pile with touching pieces, which is the hardest honest check.

---

## Plan B: no install (makesense.ai)

If Label Studio won't install, use [makesense.ai](https://www.makesense.ai) in your browser. **It does not save your work**, so do all 30 photos in one sitting with the tab open.

1. Click **Get Started** and drop in the 30 `pile_*.jpg` photos. Choose **Object Detection**.
2. Add one label, `piece`, and click **Start project**.
3. Pick the **Rect** tool and draw boxes with the same rules as Step 3. For each new box, choose `piece` in the label dropdown on the right.
4. When all 30 are done: **Actions > Export Annotations > "A .zip package containing files in YOLO format"**.
5. Rename that zip to `labels_export.zip` and continue from Step 5. The Colab cell works with either tool.
