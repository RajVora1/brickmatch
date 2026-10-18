# Showing BrickMatch: the live demo

A web page where you upload (or snap) a photo of loose pieces and get back:
1. the photo with every piece boxed and named, with how sure the model is,
2. piece counts by type and color,
3. the builds from `builds/builds.json`, best match first, with the pieces you're missing.

It runs on Colab and gives you a public link (`https://xxxx.gradio.live`) you can open on your laptop or phone, or send to someone. The link works for 72 hours, but only while the Colab cell is still running.

## Run it in Colab (about 5 minutes)

You need the two models already in Google Drive (you saved both last week):
- `MyDrive/brickmatch/detector_best.pt`
- `MyDrive/brickmatch/runs/classifier_resnet18_1.0_pad_bal.pt`

1. Open a new Colab notebook. A GPU makes it faster (**Runtime > Change runtime type > T4 GPU**), but CPU also works for a demo.
2. Paste this into one cell and press **Shift+Enter**. Click **Grant access** / **Connect to Google Drive** when asked.

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
!git clone -q https://{userdata.get('GH_TOKEN')}@github.com/RajVora1/brickmatch.git /content/brickmatch
%cd /content/brickmatch
!git pull -q
!pip install -q -r requirements.txt
# Your held-out pile photos (7, 9, 10) become clickable examples. The models never trained on these.
!unzip -qo /content/drive/MyDrive/brickmatch/my_piles_photos.zip -d /content/piles
!python -m brickmatch.demo --share \
    --detector /content/drive/MyDrive/brickmatch/detector_best.pt \
    --classifier /content/drive/MyDrive/brickmatch/runs/classifier_resnet18_1.0_pad_bal.pt \
    --examples $(find /content/piles -name 'pile_007_1.jpg' -o -name 'pile_009_1.jpg' -o -name 'pile_010_1.jpg' | sort)
```

3. After a minute the output shows `Running on public URL: https://....gradio.live`. Click it.
4. Click an example photo (or upload your own), then **Find my pieces**.

The cell keeps running while the demo is up; that's normal. To stop it, press the ■ stop button.

**If something goes wrong**
- `No such file or directory: ...detector_best.pt` or `classifier_...pt`: open the Drive folder on the left (📁) and check the exact file names, then fix them in the cell.
- `git clone` asks for a username: your `GH_TOKEN` secret expired. Make a new one (`docs/colab-setup.md`, Part 2).
- The examples don't show up: the demo still works, just upload a photo yourself.

## Tips for a live demo

- Lead with an example photo you know works (pile 10 is the honest one: never trained on, and the only pile with touching pieces). Then try a fresh photo live.
- For a fresh photo: plain background, good light, phone held straight above, pieces spread out, only the 8 tier-1 types.
- Lowering the confidence slider finds more pieces but adds false boxes; 0.5 is a good default.
- If it gets a piece wrong, that's a good moment, not a bad one: say which mix-up it is and why (see below).

## Talking points (honest numbers)

**The problem.** LEGO is expensive. Take a photo of the pieces you already own and find out what you can build with them, instead of buying a new set.

**What needs ML and what doesn't.** Recognizing pieces in a messy photo needs ML. Matching counts to a parts list is plain search (`brickmatch/match.py`). My first idea (typing in an inventory) was only the search part.

**Design: two models.** A YOLO detector finds every piece as one generic "piece" class; a ResNet-18 then names each cropped piece (8 types). Splitting it means pile photos only need boxes, and the type model can learn from easy single-type photos. This came from my mentor.

**Data I collected myself.** Bought one LEGO Classic box (10698). Photographed 10 counted mixed piles (3 photos each, 113 pieces), box-labeled all 30 photos in Label Studio, plus single-type photos for each of the 8 types, which gave 552 hand-cleaned piece crops. Split by pile so test photos are of piles the model never saw.

**Results.**
| Stage | Test | Result |
|---|---|---|
| Detector (finds pieces) | pile 10, 3 photos | recall 0.999, precision 1.0, mAP@0.5 0.995 |
| Detector count | all 30 photos | right number of pieces on 27 of 30 |
| Classifier (names pieces) | held-out single-type crops | 96% (115/120), up from 85% |
| Whole pipeline | pile 10 (test) | 85% of pieces get the right type (46/54) |
| Whole pipeline | piles 7 and 9 (val) | 71% (51/72) |

**What I learned from the errors.** The classifier jumped from 85% to 96% after I found that the standard center crop was cutting long bricks in half, so a 2x4 looked like a 2x2. Fix: pad crops to square and balance the classes. Remaining mix-ups: 1x4 brick vs 2x4 seen from the side, 1x2 brick vs 1x2 plate (only height differs), and small pieces read as 1x1. My guess is the gap between single-type training photos and real piles; next step is training on crops from the piles themselves.

**Caveats to say out loud.** The test sets are small (one test pile), so these numbers are noisy. Only 8 piece types so far. The builds database is 24 builds I made by hand. Color comes from pixel color, not a model, so it's rough under bad light.

**Next.** More and harder piles (touching, overlapping), train the classifier on pile crops, compare ResNet-18 with a ViT, add plates/windows/doors so houses work.
