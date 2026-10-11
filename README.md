# IndicSent-M

Multimodal sentiment analysis for Indian-language memes using IndicBERT (text) and ViT (image) with attention-based fusion. Classifies Positive/Neutral/Negative with explainable predictions.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repo ships with your uploaded image set already in place:

- **Images:** `data/raw/valImages/` — 1,500 meme images (from `valImages.zip`)
- **Labels:** `data/raw/labels_template.csv` — one row per image, with `text_corrected` pre-filled via OCR (`scripts/ocr_fill_template.py`) and `overall_sentiment` left empty

**Before training, you need to:**
1. Open `data/raw/labels_template.csv` and fill in `overall_sentiment` for each row (`positive` / `neutral` / `negative`). ~973/1500 rows already have OCR-extracted text in `text_corrected` — review/correct these, since meme fonts can trip up OCR; the remaining ~527 rows have empty text (OCR found nothing) and need the caption typed in manually if you want text for those samples.
2. Once labeled, you can re-run OCR any time with:
   ```bash
   python -m scripts.ocr_fill_template --csv data/raw/labels_template.csv --image_dir data/raw/valImages
   ```
   (this overwrites `text_corrected` only — re-add labels after if you re-run it)
3. `configs/config.yaml` is already pointed at these paths (`raw_csv: data/raw/labels_template.csv`, `image_dir: data/raw/valImages`) — no path changes needed.

If you later get a different/larger dataset, regenerate the template with:
```bash
python -m scripts.generate_label_template --image_dir data/raw/<your_folder> --out data/raw/<your_csv>.csv
```

## Project structure

```
IndicSent-M/
├── data/
│   ├── raw/              # downloaded dataset (images + labels CSV)
│   └── processed/        # cleaned/cached outputs
├── src/
│   ├── preprocessing/
│   │   ├── text_preprocess.py
│   │   └── image_preprocess.py
│   ├── models/
│   │   ├── text_encoder.py      # IndicBERT wrapper
│   │   ├── image_encoder.py     # ViT wrapper
│   │   └── fusion_model.py      # concat + attention fusion + text/image-only variants
│   ├── dataset.py        # PyTorch Dataset
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── eda.ipynb
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Usage

Start with the text-only baseline to validate the full pipeline before adding image complexity:

```bash
python -m src.train --fusion_type text_only
```

Then run the other variants for the comparative analysis:

```bash
python -m src.train --fusion_type image_only
python -m src.train --fusion_type concat
python -m src.train --fusion_type attention
```

Evaluate a trained checkpoint on the test split:

```bash
python -m src.evaluate --fusion_type text_only --checkpoint checkpoints/best_text_only.pt
```

## Notes

- Run each module's `if __name__ == "__main__":` block directly (e.g. `python -m src.models.text_encoder`) to smoke-test it in isolation before running full training.
- `configs/config.yaml` is the single place to change hyperparameters, paths, and model checkpoints.
- Explainability (Grad-CAM for image, attention/LIME for text) is planned as a follow-up module (`src/explain.py`) once baselines are working.
