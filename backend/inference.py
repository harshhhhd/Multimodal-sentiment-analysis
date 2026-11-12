"""
Inference service for the IndicSent-M web app.

Loads the trained checkpoints produced by `python -m src.train` and exposes
prediction, OCR and occlusion-based explanations. Models are loaded lazily
(first request for each variant) and cached.

Model/tokenizer names, label classes, max length and image size are all read
from configs/config.yaml, so the web app always matches how you trained.
Set INDICSENT_CONFIG to use a different config file.
"""

import os
import sys
import threading
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.fusion_model import MultimodalSentimentModel  # noqa: E402
from src.preprocessing.image_preprocess import get_transforms  # noqa: E402
from src.preprocessing.text_preprocess import preprocess_text  # noqa: E402

try:
    import pytesseract

    _win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_win_tesseract)
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

MODEL_INFO = {
    "text_only": "Text only",
    "image_only": "Image only",
    "concat": "Multimodal (concatenation)",
    "attention": "Multimodal (attention)",
}
USES_TEXT = {"text_only", "concat", "attention"}
USES_IMAGE = {"image_only", "concat", "attention"}

GRID = 4          # image explanation grid is GRID x GRID cells
MAX_WORDS = 20    # words explained per request (keeps CPU latency reasonable)


class Predictor:
    def __init__(self, config_path=None):
        path = Path(config_path or os.environ.get("INDICSENT_CONFIG") or ROOT / "configs" / "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.classes = list(self.cfg["labels"]["classes"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = int(self.cfg["text_model"]["max_length"])
        self.image_size = int(self.cfg["image_model"]["image_size"])
        self.transform = get_transforms(image_size=self.image_size, train=False)
        self.ckpt_dir = ROOT / self.cfg["training"]["checkpoint_dir"]

        self._tokenizer = None
        self._models = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def checkpoint_path(self, fusion_type):
        return self.ckpt_dir / f"best_{fusion_type}.pt"

    def available_models(self):
        return [ft for ft in MODEL_INFO if self.checkpoint_path(ft).exists()]

    def _tok(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg["text_model"]["name"])
        return self._tokenizer

    def _model(self, fusion_type):
        if fusion_type not in MODEL_INFO:
            raise ValueError(f"Unknown model '{fusion_type}'.")
        ckpt = self.checkpoint_path(fusion_type)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No trained checkpoint for '{fusion_type}' at {ckpt}. "
                f"Train it first: python -m src.train --fusion_type {fusion_type}"
            )
        mtime = ckpt.stat().st_mtime
        with self._lock:
            cached = self._models.get(fusion_type)
            # Reload when the checkpoint file changed (e.g. you retrained while the server runs).
            if cached is None or cached[0] != mtime:
                model = MultimodalSentimentModel(
                    text_model_name=self.cfg["text_model"]["name"],
                    image_model_name=self.cfg["image_model"]["name"],
                    num_classes=len(self.classes),
                    fusion_type=fusion_type,
                )
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device).eval()
                cached = (mtime, model)
                self._models[fusion_type] = cached
            return cached[1]

    # --------------------------------------------------------------- encoding
    def _encode_texts(self, texts):
        enc = self._tok()(
            [preprocess_text(t) for t in texts],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _encode_image(self, image):
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _probs(self, model, ids, mask, pix):
        return torch.softmax(model(input_ids=ids, attention_mask=mask, pixel_values=pix), dim=1)

    # -------------------------------------------------------------------- OCR
    def extract_text(self, image):
        if not OCR_AVAILABLE:
            return ""
        try:
            return " ".join(pytesseract.image_to_string(image).split())
        except Exception:
            return ""

    # ---------------------------------------------------------------- predict
    def predict(self, fusion_type, image, text, explain=False):
        model = self._model(fusion_type)
        if image is None:  # text-only requests still need a placeholder image tensor
            image = Image.new("RGB", (self.image_size, self.image_size), "white")
        text = text or ""

        ids, mask = self._encode_texts([text])
        pix = self._encode_image(image)
        probs = self._probs(model, ids, mask, pix)[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])

        result = {
            "model": fusion_type,
            "model_label": MODEL_INFO[fusion_type],
            "label": self.classes[idx],
            "confidence": confidence,
            "probabilities": {c: float(probs[i]) for i, c in enumerate(self.classes)},
        }
        if explain:
            result["explanation"] = self._explain(model, fusion_type, text, pix, idx, confidence)
        return result

    # ---------------------------------------------------------------- explain
    def _explain(self, model, fusion_type, text, pix, idx, base):
        """Occlusion explanations: how much does the predicted-class probability
        drop when a word is removed / an image cell is blanked out?
        Positive score = that part supported the prediction."""
        explanation = {"words": None, "image_grid": None, "grid": GRID}

        if fusion_type in USES_TEXT and text.strip():
            all_words = text.split()
            n = min(len(all_words), MAX_WORDS)
            variants = [" ".join(all_words[:i] + all_words[i + 1:]) for i in range(n)]
            ids, mask = self._encode_texts(variants)
            p = self._probs(model, ids, mask, pix.expand(n, -1, -1, -1))[:, idx]
            explanation["words"] = [
                {"word": all_words[i], "score": base - float(p[i])} for i in range(n)
            ]

        if fusion_type in USES_IMAGE:
            cells = GRID * GRID
            ids, mask = self._encode_texts([text])
            ch, cw = self.image_size // GRID, self.image_size // GRID
            batch = pix.repeat(cells, 1, 1, 1)
            for r in range(GRID):
                for c in range(GRID):
                    batch[r * GRID + c, :, r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = 0.0
            p = self._probs(model, ids.expand(cells, -1), mask.expand(cells, -1), batch)[:, idx]
            drops = [base - float(v) for v in p]
            explanation["image_grid"] = [drops[r * GRID:(r + 1) * GRID] for r in range(GRID)]

        return explanation
