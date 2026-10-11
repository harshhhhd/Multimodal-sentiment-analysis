"""
Scans an image folder and generates a labels CSV template with one row per
image, ready for you to fill in the meme text and sentiment label.

Usage:
    python -m scripts.generate_label_template --image_dir data/raw/valImages --out data/raw/labels_template.csv
"""

import argparse
import os
import csv


def generate_template(image_dir: str, out_path: str):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "text_corrected", "overall_sentiment"])
        for fname in filenames:
            writer.writerow([fname, "", ""])

    print(f"Wrote {len(filenames)} rows to {out_path}")
    print("Fill in 'text_corrected' (meme caption/OCR text) and "
          "'overall_sentiment' (positive/neutral/negative) for each row.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/raw/valImages")
    parser.add_argument("--out", type=str, default="data/raw/labels_template.csv")
    args = parser.parse_args()

    generate_template(args.image_dir, args.out)
