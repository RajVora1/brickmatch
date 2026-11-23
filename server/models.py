"""Find and load the two trained models (detector + piece classifier) for the website's API.

Where the weights come from, first match wins:
1. BRICKMATCH_DETECTOR and BRICKMATCH_CLASSIFIER: paths to the two .pt files.
2. weights/detector.pt and weights/classifier.pt in the repo folder (handy on your own computer).
3. BRICKMATCH_MODEL_REPO (e.g. "RajVora1/brickmatch-models"): a Hugging Face model repo holding
   detector.pt and classifier.pt. If the repo is private, also set HF_TOKEN (a read token).

Loading runs in a background thread so the server answers right away; /api/health says when it's ready.
"""
import os
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_FILE = "detector.pt"
CLASSIFIER_FILE = "classifier.pt"


class Models:
    def __init__(self):
        self.state = "loading"  # loading | ready | missing | error
        self.message = "Loading the models..."
        self.models = None
        self.lock = threading.Lock()  # one photo at a time: keeps memory use low on a small free server

    def start(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _paths(self):
        det, cls = os.environ.get("BRICKMATCH_DETECTOR"), os.environ.get("BRICKMATCH_CLASSIFIER")
        if det and cls:
            return det, cls
        local = ROOT / "weights"
        if (local / DETECTOR_FILE).exists() and (local / CLASSIFIER_FILE).exists():
            return str(local / DETECTOR_FILE), str(local / CLASSIFIER_FILE)
        repo = os.environ.get("BRICKMATCH_MODEL_REPO")
        if repo:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            token = os.environ.get("HF_TOKEN") or None
            return (hf_hub_download(repo, DETECTOR_FILE, token=token),
                    hf_hub_download(repo, CLASSIFIER_FILE, token=token))
        return None

    def _load(self):
        try:
            paths = self._paths()
            if paths is None:
                self.state = "missing"
                self.message = ("No model weights found. Set BRICKMATCH_MODEL_REPO (see docs/hosting.md) or put "
                                "detector.pt and classifier.pt in weights/.")
                return
            from brickmatch.pipeline import load_models  # noqa: PLC0415
            self.models = load_models(*paths)
            self.state, self.message = "ready", "Models loaded."
        except Exception as e:  # noqa: BLE001  (report any failure on /api/health instead of crashing)
            traceback.print_exc()
            self.state, self.message = "error", f"Couldn't load the models: {e}"

    @property
    def classes(self):
        return self.models[2] if self.models else []
