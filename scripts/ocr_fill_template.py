"""
Fills the 'text_corrected' column of the labels template using OCR on each
meme image. This gives you a starting point — review and correct the OCR
output, since it can misread stylized meme fonts, and you still need to add
'overall_sentiment' labels yourself.

Usage:
    python -m scripts.ocr_fill_template --csv data/raw/labels_template.csv --image_dir data/raw/valImages
"""

import argparse
import csv
import os

from PIL import Image, UnidentifiedImageError

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path: str) -> str:
    """Standalone OCR helper — kept independent of src.preprocessing.image_preprocess
    so this script doesn't require torch/torchvision just to run OCR."""
    if not OCR_AVAILABLE:
        return ""
    try:
        with Image.open(image_path) as img:
            return pytesseract.image_to_string(img).strip()
    except (UnidentifiedImageError, OSError):
        return ""


def fill_template(csv_path: str, image_dir: str):
    if not OCR_AVAILABLE:
        print("pytesseract not installed — run `pip install pytesseract` and "
              "ensure the tesseract binary is on your system PATH.")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    for i, row in enumerate(rows):
        image_path = os.path.join(image_dir, row["image_name"])
        text = extract_text_from_image(image_path)
        row["text_corrected"] = text.replace("\n", " ").strip()
        if i % 100 == 0:
            print(f"OCR'd {i}/{len(rows)}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"OCR text written back to {csv_path}. "
          "'overall_sentiment' is still empty — label that column before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/labels_template.csv")
    parser.add_argument("--image_dir", type=str, default="data/raw/valImages")
    args = parser.parse_args()

    fill_template(args.csv, args.image_dir)
