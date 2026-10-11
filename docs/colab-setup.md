# Setting up Google Colab for BrickMatch (click by click)

Colab is a free Jupyter notebook that runs in your browser on Google's computers, including a free GPU. You don't need to install anything on your own computer for this.

What you'll have at the end: a notebook that downloads a LEGO dataset, trains a ResNet-18 classifier with the repo's code, and saves the model to your Google Drive.

You need three accounts: **Google** (for Colab and Drive), **Kaggle** (free, for datasets), and **GitHub** (you already have it).

---

## Part 1: Get your Kaggle key (5 min)

Colab uses this key to download datasets from Kaggle.

1. Go to [kaggle.com](https://www.kaggle.com) and sign up or sign in. Signing in with Google is easiest.
2. Click your **profile picture** (top right), then **Settings**.
3. Open the **API Tokens** tab and click **Generate New Token** (name it `brickmatch`).
4. A popup shows a token starting with `KGAT_`. Copy it with the copy icon. You'll paste it in Part 3, and Kaggle won't show it again (you can always generate a new one).

> Older accounts may instead see **Create Legacy API Key**, which downloads `kaggle.json`. Don't use that; the new token is simpler.

Treat the key like a password: don't paste it into the notebook itself, a chat, or GitHub.

## Part 2: Get a GitHub token (5 min)

Your repo is private, so Colab needs permission to download (clone) it.

1. On [github.com](https://github.com), click your **profile picture** (top right), then **Settings**.
2. At the very bottom of the left sidebar, click **Developer settings**.
3. Click **Personal access tokens**, then **Fine-grained tokens**, then **Generate new token**.
4. Fill in:
   - **Token name:** `colab-brickmatch`
   - **Expiration:** 90 days
   - **Repository access:** pick **Only select repositories**, then choose `brickmatch`.
   - **Permissions:** click **+ Add permissions**, pick **Contents**, and make sure it says **Read-only**. The Repositories count should then show 1 (GitHub may add Metadata automatically, which is fine). Leave everything else alone.
5. Click **Generate token** at the bottom. Copy the token (it starts with `github_pat_`) right away, because GitHub only shows it once.

This token can only read the brickmatch repo, so it's low risk if it ever leaks.

## Part 3: Create the notebook and add your secrets (5 min)

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with Google.
2. Click **New notebook** (or **File > New notebook in Drive**). Rename it at the top left, for example `brickmatch_warmup.ipynb`. Colab saves it automatically to Google Drive, in a folder called **Colab Notebooks**.
3. **Turn on the GPU:** click **Runtime > Change runtime type**, choose **T4 GPU**, and click **Save**.
4. **Add the secrets:** click the **key icon** (🔑, labeled "Secrets") in the left sidebar. Then, for each row below, click **+ Add new secret**, type the name and value, and switch **Notebook access** on:

   | Name (type exactly) | Value |
   |---|---|
   | `KAGGLE_API_TOKEN` | your `KGAT_...` token |
   | `GH_TOKEN` | your `github_pat_...` token |

   Secrets are saved to your Google account, so you only do this once. They work in any notebook where Notebook access is on.

## Part 4: Check that it works

A notebook is a list of **cells**. Click **+ Code** to add a cell, paste code in, and press **Shift+Enter** (or the ▶ button) to run it. Output appears under the cell. A line starting with `!` runs a terminal command, and `%cd` changes folder.

**Cell 1: is the GPU on?**
```python
!nvidia-smi
```
You should see a table mentioning **Tesla T4**. If it says "command not found" or "No devices", go back to step 3 of Part 3.

**Cell 2: secrets, clone the repo, install the libraries**
```python
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
!git clone https://{userdata.get('GH_TOKEN')}@github.com/RajVora1/brickmatch.git
%cd /content/brickmatch
!pip install -q -r requirements.txt
!pip install -q -U kaggle   # new KGAT_ tokens need a recent kaggle
!ls
```
The last line should list `README.md`, `configs`, `train`, and so on. The first time, Colab may pop up "Grant access to secret?". Click **Grant access**.

**Cell 3 onward:** continue with Cells 2 to 5 from section 4b of `docs/week-2-plan.md`: download the dataset, sort it into `data/crops`, train, and save to Drive. In that plan, Cell 1 is replaced by Cell 2 here, so skip it.

---

## Things that will trip you up

- **Everything in `/content` is wiped** when the session ends (after about 90 minutes idle, or about 12 hours max). Your notebook and secrets are safe, but you'll need to re-run the cells from the top, which takes a few minutes. Anything you want to keep, like `runs/`, has to be copied to Drive first (Cell 5 in the plan does this).
- **"Runtime disconnected":** click **Reconnect**, then **Runtime > Run all**, or run the cells one by one from the top.
- **The free GPU is rationed.** If Colab says no GPU is available, wait a few hours or use a [Kaggle notebook](https://www.kaggle.com/code), which gives 30 free GPU hours a week. The same cells work there, minus the `google.colab` secrets lines.
- **`git clone` asks for a username or says "not found":** the `GH_TOKEN` secret is wrong, has expired, or doesn't include the brickmatch repo. Make a new token (Part 2).
- **`kaggle` says 401 Unauthorized:** the secret name is misspelled, Notebook access is off, or the `pip install -U kaggle` line was skipped.
- **`fatal: destination path 'brickmatch' already exists`** when re-running Cell 2 in the same session is harmless. To get the latest code instead, run `!cd /content/brickmatch && git pull`.
- **You changed code in the repo and Colab doesn't see it:** Colab cloned an old copy. Run the `git pull` line above.
