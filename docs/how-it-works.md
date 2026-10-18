# How BrickMatch works (plain English)

This is the "why" behind everything. Read it once top to bottom; come back to the file-by-file section when you open a file and wonder what it is.

---

## 1. The big picture

You want: **photo of a LEGO pile in, list of things you can build out.**

That breaks into three jobs:

| Job | Question it answers | How it's solved | Is it ML? |
|---|---|---|---|
| **Find** | "Where is every piece in this photo?" | Model 1: a *detector* (YOLO) draws a box around each piece | Yes |
| **Name** | "What type is the piece in this box?" | Model 2: a *classifier* (ResNet-18) looks at one cropped box and says "brick_2x4" | Yes |
| **Match** | "What can I build with 6 of these and 4 of those?" | Plain code compares your counts to parts lists | No, just search |

Color is the odd one out: no model needed. The code takes the middle of each box, finds the typical pixel color, and picks the closest color from `configs/colors.yaml`.

```
photo ──> [Find: YOLO] ──> boxes ──> crop each box ──> [Name: ResNet-18] ──> "brick_2x4", "plate_1x2", ...
                                                  └──> [Color: pixel math] ──> "red", "blue", ...
                                                                    │
                                                         count by type ──> [Match] ──> "You can build: Simple tower (100%), Low wall (90%, missing 2x brick_1x2)"
```

### Why two models instead of one?
One model that finds *and* names pieces needs every pile photo labeled with a box **and** the exact type of every piece. That's slow and error-prone (is that a 1x2 or a 1x4 under there?).

Splitting it means:
- **Finding** only needs boxes labeled "piece". Easy and fast to label.
- **Naming** can learn from tons of existing single-piece images online, plus your "only 2x4 bricks" photos, where every piece's type is already known for free.

That was your mentor's advice, and it's why the data plan looks the way it does.

---

## 2. ML ideas you'll keep running into

**Training.** You show the model thousands of examples ("this image is a brick_2x4"). It guesses, gets told how wrong it was (the *loss*), and nudges its internal numbers to be a bit less wrong next time. Repeat many times.

**Epoch.** One full pass through all the training images. Your warm-up ran 5 epochs, which is why it printed 5 lines.

**Train / val / test split.** You split your images into three piles:
- **train** (~70%): the model learns from these.
- **val** (validation, ~15%): checked after each epoch to see if it's improving. The script keeps the version that scores best here.
- **test** (~15%): looked at **once**, at the very end. It's the honest final grade, because the model never learned from it and you never used it to pick anything.

It's like studying with practice problems (train), taking practice quizzes to see if you're ready (val), then taking the real exam (test).

**Accuracy.** Fraction of images it labeled correctly. Your 0.963 test accuracy = 96.3% of test images right.

**Overfitting.** When a model memorizes its training images instead of learning the general idea, it scores great on train but badly on new images. That's why you always judge it on images it hasn't seen.

**Why "split by pile" matters.** If photo 1 of pile 17 is in train and photo 2 of pile 17 is in test, the model has basically seen the test answer. So all photos of one pile go in the same split.

**Transfer learning / pretrained models.** ResNet-18 and YOLO were already trained by others on millions of everyday photos, so they already "know" edges, shapes and textures. You only teach them the last step: LEGO types. That's why you can get 95% with a few minutes of training. That's also what the `Downloading ... resnet18-f37072fd.pth` line was: the pretrained starting point.

**Renders vs real photos (the domain gap).** The Kaggle images are computer-generated: perfect lighting, clean background. Your phone photos will have shadows, clutter and overlapping pieces. Models trained on one usually score lower on the other. Measuring that drop and shrinking it (by adding your own photos to training) is a big part of your project's story.

---

## 3. The plan, step by step, and why each step exists

| Step | What you do | Why it's needed |
|---|---|---|
| **1. Finalize the piece list** | Decide the exact piece types (now 8) in `configs/pieces.yaml` | Everything is organized by these names: photo folders, crop folders, the classifier's possible answers, build parts lists. Changing them later means renaming everything. |
| **2. Single-type pile photos** | Photos of piles with only one piece type | Free labels for the *Name* model: every piece in the "brick_2x4" folder is a 2x4. Teaches it what *your* pieces look like in *your* lighting. |
| **3a. Mixed pile photos + true counts** | Realistic piles, and you write down exactly what's in each | The real test of the whole system. The counts are the "answer key" for checking if BrickMatch counted correctly. |
| **3b. Box labeling** | Draw a box around every piece in some mixed-pile photos | Training data for the *Find* model. It learns "a piece looks like this, draw a box here". |
| **4. Colab warm-up** ✅ | Trained the *Name* model on online images | Proved your setup works end to end before your own data exists. You already did this. |
| **5. Progress log** | Weekly entry in the README | What your mentors read: what you did, the numbers, what's blocking you. |

**Later weeks**, roughly: train the *Find* model on your boxed photos (`train_detector.py`), use it to auto-crop your single-type photos into classifier training data, retrain the *Name* model with your real crops, run the full pipeline on held-out piles, and score it with `eval/evaluate.py`. Then compare ResNet-18 with ViT and with less data, which is the "experiment" part of the capstone.

---

## 4. What each file does

### Settings (`configs/`)
- **`pieces.yaml`**: the list of piece types. `id` is the name the code uses (`brick_2x4`), `part` is the official LEGO part number (3001) so you can look it up, `tier: 1` means "in use now", `tier: 2` means "later".
- **`colors.yaml`**: the supported colors and their RGB values. The pipeline picks whichever of these is closest to a piece's pixels.
- **`detect.yaml`**: tells YOLO where the box-labeled photos live and that there's just one class, `piece`.

### The builds database (`builds/builds.json`)
A list of builds, each with a name, a category (tower, wall, bridge, house) and a parts list like "8x brick_2x2, 2x slope45_2x1". It has 24 hand-made builds that use only the 8 tier-1 pieces, so everything it suggests can come out of the 10698 box. Online build collections (Rebrickable MOCs, LEGO idea books) use dozens of piece types, so almost none would ever match a tier-1 pile.

### Training scripts (`train/`)
- **`train_classifier.py`**: trains the *Name* model. What it does, in order:
  1. Loads images from `data/crops/train`, `val`, `test`. Each folder name is the label (`data/crops/train/brick_2x4/*.jpg` are all 2x4s).
  2. Applies random rotations, crops and color changes to training images (*augmentation*), so the model doesn't care which way a piece faces or how bright the light is.
  3. Loads pretrained ResNet-18 (or ViT with `--arch vit_b_16`) and swaps its last layer for one with one output per piece type.
  4. Trains for `--epochs` rounds, printing val accuracy each time and saving the best version to `runs/`.
  5. Scores the best version on `test` and saves a *confusion matrix*: a table of which pieces get mistaken for which (e.g. "1x2 brick called 1x4 brick 12 times").

  `--train-fraction 0.25` trains on only a quarter of the data, so you can show how accuracy grows with more data.
- **`train_detector.py`**: trains the *Find* model (YOLO) on your box-labeled photos. It reports **recall** (what fraction of real pieces it found) and **precision** (what fraction of its boxes were real pieces). Recall matters most: a piece it misses can never be counted.

### The app itself (`brickmatch/`)
- **`pipeline.py`**: the whole thing glued together. Give it a photo plus the two trained models and it: runs YOLO to get boxes → crops each box → asks the classifier what type it is → reads its color → counts everything → hands the counts to the matcher. Prints counts and ranked builds.
- **`match.py`**: the non-ML part. For each build it computes **coverage** = how many of the needed pieces you have ÷ how many it needs, and lists what's missing. Sorts best first. You can try it right now without any model:
  ```
  python -m brickmatch.match '{"brick_2x2": 8, "slope45_2x1": 2}'
  ```

### Evaluation (`eval/evaluate.py`)
Answers the two questions from your project proposal, on pile photos the models never trained on:
1. **How well does it count?** Compares predicted counts to your `counts.json` answer key: average number of pieces off per photo, and % of photos counted exactly right.
2. **Are its recommendations real?** For every build it says you can fully make, checks against the true counts whether you actually could.

### Data folders (`data/`)
- `raw/`: original photos (yours and downloaded). Never edited.
- `detect/`: photos + box label files for the *Find* model.
- `crops/`: one-piece images in a folder per type, for the *Name* model.

None of this is uploaded to GitHub (too big). It lives in Colab / Google Drive.

### Docs (`docs/`)
`week-3-plan.md` (this week's tasks), `colab-setup.md`, `labeling-guide.md` (how to shoot and box photos), `datasets.md` (public datasets), and this file.

---

## 5. What your warm-up actually did

| Cell (week plan) | What happened |
|---|---|
| 1: secrets + clone | Gave Colab your Kaggle and GitHub tokens, copied your repo onto Colab's computer, installed PyTorch etc. |
| 2: download | Downloaded ~40k rendered LEGO images from Kaggle. |
| 3: sort | Kept only images whose part number matches your tier-1 list, and split each type 70/15/15 into `data/crops/train`, `val`, `test`. |
| 4: train | Ran `train_classifier.py`: 5 epochs of ResNet-18. Val accuracy went 83% → 95%, test landed at 96.3%. |
| 5: save | Copied the trained model (`runs/`) to your Google Drive, because Colab wipes its disk when the session ends. |

What it **proves**: your setup works and the code trains a real model. What it **doesn't** prove: that it works on your photos. That's what the next weeks are for.
