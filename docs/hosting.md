# Putting BrickMatch online

The website lets anyone snap a photo of their pieces, fix the counts, see what they can build, and follow the steps
in 3D or as a PDF booklet. Everything here is free.

## How it fits together

```
  visitor's browser
        │
        ▼
  GitHub Pages  ───────────────►  Hugging Face Space
  rajvora1.github.io/brickmatch   <you>-brickmatch.hf.space
  the website (web/): HTML, CSS,  the API (server/): Python, PyTorch,
  JS, build pictures. Browsing    the detector + classifier. Finds pieces
  builds and matching counts run  in photos, draws steps, makes booklets
  right in the browser.           and 3D views.
                                        │ downloads the weights at start
                                        ▼
                                  Hugging Face model repo (private)
                                  detector.pt + classifier.pt
```

**Why two places?** GitHub Pages only serves files. It can't run Python or PyTorch. A Hugging Face Space is a free
small server (2 CPUs, 16 GB memory) that can. The website lives on Pages because it loads instantly and never sleeps.

**The catch with free:** the Space goes to sleep after 48 hours with no visitors. The next visitor wakes it, which
takes about a minute. The site shows "Waking up the brick sorter…" in the top corner while that happens. Browsing
builds and typing in counts still work while it wakes up.

**What's public:** the code (once the repo is public) and the website. The model weights stay in a private
Hugging Face repo that only your Space can read. Photos people upload are only used to find pieces. They aren't saved.

Both parts update themselves: every push to `main` redeploys the website and the Space
(`.github/workflows/pages.yml` and `deploy-space.yml`).

Total setup time: about 30 minutes, most of it waiting for the Space to build the first time.

---

## Part 1: Try it on your own computer (optional, 5 minutes)

```
pip install torch torchvision
pip install -r server/requirements.txt
mkdir weights
# copy your two models in, with these names:
#   weights/detector.pt    <- detector_best.pt
#   weights/classifier.pt  <- runs/classifier_resnet18_1.0_pad_bal.pt
uvicorn server.app:app --port 7860
```

Open http://localhost:7860. The same server runs the API and the website. Without weights everything except
"find pieces in a photo" still works, and the top corner says "Photo finder offline".

To check the API: http://localhost:7860/api/health should say `"models": "ready"`. http://localhost:7860/api/docs
lists every endpoint.

## Part 2: Make the GitHub repo public (1 minute)

GitHub Pages is free only for public repos.

1. I checked the history for leaked tokens (GitHub, Hugging Face, Kaggle, AWS) and model files: none found. Data
   and weights were never committed (`.gitignore`). If you ever pasted a token into a notebook that's in the repo,
   revoke that token anyway.
2. On GitHub: **Settings > General**, scroll to **Danger Zone > Change repository visibility > Make public**.

## Part 3: The API on Hugging Face (15 minutes + build time)

### 3a. Account and tokens

1. Sign up at https://huggingface.co/join (free). Your username shows in the top-right menu. Below, `<hf-user>`
   means that username.
2. Go to **Settings > Access Tokens > Create new token**:
   - Token 1: type **Write**, name `github-deploy`. GitHub uses it to upload code to the Space.
   - Token 2: type **Read**, name `space-read-models`. The Space uses it to download the weights.

   Copy each one when it's shown (it's shown once).

### 3b. Upload the model weights (private)

1. **New > Model** (https://huggingface.co/new). Name `brickmatch-models`, set to **Private**, **Create model**.
2. Upload the two files with these exact names: `detector.pt` and `classifier.pt`. Easiest from Colab, where they
   already are. Paste this into a cell (replace `<hf-user>` and paste the **Write** token when asked):

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q huggingface_hub
from huggingface_hub import HfApi, login
login()  # paste the Write token
api = HfApi()
repo = "<hf-user>/brickmatch-models"
api.upload_file(path_or_fileobj="/content/drive/MyDrive/brickmatch/detector_best.pt",
                path_in_repo="detector.pt", repo_id=repo)
api.upload_file(path_or_fileobj="/content/drive/MyDrive/brickmatch/runs/classifier_resnet18_1.0_pad_bal.pt",
                path_in_repo="classifier.pt", repo_id=repo)
print("done")
```

   (Or download the two files from Drive, rename them, and drag them into **Files and versions > Add file > Upload
   files** on the model page.)

### 3c. Create the Space

1. **New > Space** (https://huggingface.co/new-space).
   - Name: `brickmatch`
   - License: your choice (MIT is common)
   - SDK: **Docker**, template **Blank**
   - Hardware: **CPU basic, free**
   - Visibility: **Public** (a private Space can't be reached by the website)
2. In the new Space, open **Settings > Variables and secrets**:
   - **New variable**: `BRICKMATCH_MODEL_REPO` = `<hf-user>/brickmatch-models`
   - **New secret**: `HF_TOKEN` = the **Read** token
   - Optional, once the site is up: **New variable** `BRICKMATCH_ALLOWED_ORIGINS` = `https://rajvora1.github.io`
     (then only your website can call the API from a browser).

### 3d. Let GitHub upload the code to the Space

1. On GitHub: **Settings > Secrets and variables > Actions**.
   - **Secrets** tab, **New repository secret**: `HF_TOKEN` = the **Write** token.
   - **Variables** tab, **New repository variable**: `HF_SPACE` = `<hf-user>/brickmatch`.
2. **Actions** tab > **Deploy API to Hugging Face** > **Run workflow** (on `main`). It takes about a minute.
3. Back on the Space page, it now says **Building**. The first build takes 5 to 10 minutes (installing PyTorch).
   Click **Logs** to watch it. When it says **Running**, open
   `https://<hf-user>-brickmatch.hf.space/api/health`. You want `"models": "ready"`.

The Space also serves the whole website at `https://<hf-user>-brickmatch.hf.space`, so you can already try it there.

## Part 4: The website on GitHub Pages (5 minutes)

1. **Settings > Secrets and variables > Actions > Variables > New repository variable**:
   `BRICKMATCH_API_URL` = `https://<hf-user>-brickmatch.hf.space` (no slash at the end).
2. **Settings > Pages > Build and deployment > Source**: choose **GitHub Actions**.
3. **Actions** tab > **Deploy website to GitHub Pages** > **Run workflow**.
4. After about a minute the site is at **https://rajvora1.github.io/brickmatch/**. The link also shows on the
   workflow run and under **Settings > Pages**.

Check that the top-right corner says **Brick sorter ready**, then click one of the sample piles.

## Updating

Push (or merge a PR) to `main`. The Pages workflow redeploys the site when `web/`, `builds/` or the drawing code
changes. The Space workflow redeploys the API when the Python code changes. To swap in a newly trained model,
upload it to the model repo with the same name (`detector.pt` or `classifier.pt`), then on the Space click
**Settings > Restart this Space**.

## When something goes wrong

| What you see | What to do |
|---|---|
| Top corner stays on "Waking up the brick sorter…" for minutes | Open the Space page. If it says **Paused** or **Sleeping**, click to restart. If **Build error** or **Runtime error**, open **Logs**. |
| "Photo finder offline" | `/api/health` says why. Usually `BRICKMATCH_MODEL_REPO` has a typo, a file isn't named exactly `detector.pt` / `classifier.pt`, or the Space's `HF_TOKEN` secret is missing. Fix it, then **Restart this Space**. |
| "Server unreachable" on the Pages site, but the Space works | `BRICKMATCH_API_URL` is wrong (check the spelling, no slash at the end) or `BRICKMATCH_ALLOWED_ORIGINS` doesn't match the site's address. Re-run the Pages workflow after changing the variable. |
| The deploy workflow fails with 401 or 403 | The GitHub `HF_TOKEN` secret is the read token, or expired. Make a new **Write** token and update the secret. |
| The Pages workflow fails at "deploy" | **Settings > Pages > Source** isn't set to **GitHub Actions**. |
| Photos are slow (10+ seconds) | Normal-ish on the free CPU with big piles. The first photo after waking is the slowest. Paid hardware (Space **Settings > Space hardware**) is faster, but costs money. |
| The 3D view is blank | It loads three.js from cdn.jsdelivr.net. A school or office network may block it. |

## What lives where in the repo

| Path | What it is |
|---|---|
| `web/` | The website: `index.html`, `style.css`, `app.js`, `config.js` (API address), sample piles |
| `web/generated/` | Build and piece pictures + `catalog.json`, made by `python -m server.site_assets` on deploy (not committed) |
| `server/app.py` | The API: `/api/health`, `/api/detect`, `/api/build`, `/api/booklet` |
| `server/models.py` | Finds and loads the weights (env variables, `weights/`, or the HF model repo) |
| `space/` | `Dockerfile` and `README.md` for the Hugging Face Space |
| `.github/workflows/` | `pages.yml` (website) and `deploy-space.yml` (API) |
| `tests/test_site.py` | API checks with a fake model: `python -m pytest tests/test_site.py` |
