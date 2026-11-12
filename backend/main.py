"""
FastAPI app for IndicSent-M.

Run from the PROJECT ROOT (the folder that contains src/, backend/, configs/):

    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

API docs: http://127.0.0.1:8000/docs
If frontend/dist exists (after `npm run build`), the UI is served at /.

Auth: /api/auth/register, /api/auth/login, /api/auth/me are public/self-service.
/api/predict, /api/compare and /api/ocr require "Authorization: Bearer <token>".
"""

import io
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.auth import get_current_user, init_db
from backend.auth import router as auth_router
from backend.inference import MODEL_INFO, OCR_AVAILABLE, USES_IMAGE, USES_TEXT, Predictor

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="IndicSent-M API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(auth_router)
predictor = Predictor()


# ---------------------------------------------------------------- helpers
def _read_image(upload: Optional[UploadFile]):
    if upload is None:
        return None
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        return None
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")


def _resolve_text(image, text: str):
    """Use the user's text; if empty, fall back to OCR on the image."""
    text = (text or "").strip()
    if text:
        return text, "user"
    if image is not None:
        ocr_text = predictor.extract_text(image)
        if ocr_text:
            return ocr_text, "ocr"
    return "", "none"


def _missing_input(fusion_type: str, image, text: str):
    label = MODEL_INFO[fusion_type]
    if fusion_type in USES_IMAGE and image is None:
        return f"'{label}' needs an image."
    if fusion_type in USES_TEXT and not text:
        return f"'{label}' needs text. Type it in, or upload an image with readable text for OCR."
    return None


# -------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(predictor.device),
        "classes": predictor.classes,
        "ocr_available": OCR_AVAILABLE,
        "available_models": predictor.available_models(),
    }


@app.get("/api/models")
def models():
    available = set(predictor.available_models())
    return [
        {
            "id": key,
            "label": label,
            "available": key in available,
            "needs_text": key in USES_TEXT,
            "needs_image": key in USES_IMAGE,
        }
        for key, label in MODEL_INFO.items()
    ]


@app.post("/api/ocr")
def ocr(image: UploadFile = File(...), _user: dict = Depends(get_current_user)):
    img = _read_image(image)
    if img is None:
        raise HTTPException(status_code=400, detail="No image received.")
    return {"text": predictor.extract_text(img), "ocr_available": OCR_AVAILABLE}


@app.post("/api/predict")
def predict(
    image: Optional[UploadFile] = File(None),
    text: str = Form(""),
    fusion_type: str = Form("text_only"),
    explain: bool = Form(False),
    _user: dict = Depends(get_current_user),
):
    if fusion_type not in MODEL_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown model '{fusion_type}'.")

    img = _read_image(image)
    text_used, source = _resolve_text(img, text)
    problem = _missing_input(fusion_type, img, text_used)
    if problem:
        raise HTTPException(status_code=400, detail=problem)

    try:
        result = predictor.predict(fusion_type, img, text_used, explain)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result["text_used"] = text_used
    result["text_source"] = source
    return result


@app.post("/api/compare")
def compare(
    image: Optional[UploadFile] = File(None),
    text: str = Form(""),
    _user: dict = Depends(get_current_user),
):
    img = _read_image(image)
    text_used, source = _resolve_text(img, text)

    available = predictor.available_models()
    if not available:
        raise HTTPException(
            status_code=404,
            detail="No trained checkpoints found. Train at least one: python -m src.train --fusion_type text_only",
        )

    results, skipped = [], []
    for fusion_type in available:
        problem = _missing_input(fusion_type, img, text_used)
        if problem:
            skipped.append({"model": fusion_type, "model_label": MODEL_INFO[fusion_type], "reason": problem})
            continue
        results.append(predictor.predict(fusion_type, img, text_used, explain=False))

    return {"text_used": text_used, "text_source": source, "results": results, "skipped": skipped}


# Serve the built React UI (npm run build) when present, for one-command deployment.
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
