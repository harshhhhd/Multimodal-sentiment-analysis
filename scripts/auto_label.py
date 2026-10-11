"""
Auto-labels the 'overall_sentiment' column using a pretrained sentiment
model (weak/silver labeling), since manually labeling thousands of memes
isn't feasible on a tight timeline.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest, which natively
outputs negative/neutral/positive (matches our label set exactly) and is
trained on social media text, a reasonable match for meme captions.

Rows with no OCR text are left unlabeled (can't infer sentiment from
nothing) and will be dropped by train.py automatically.

IMPORTANT: these are weak labels, not ground truth. State this clearly in
your report. Spot-check a random sample (see --sample_check) and manually
correct any you can before the deadline — even partial manual correction
improves quality over pure auto-labels.

Usage:
    python -m scripts.auto_label --csv data/raw/english_meme_labels.csv
    python -m scripts.auto_label --csv data/raw/english_meme_labels.csv --sample_check 30
"""

import argparse
import csv
import random

from transformers import pipeline


LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    # some checkpoints return LABEL_0/1/2 instead of names
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def auto_label(csv_path: str, text_column: str = "text_corrected",
                label_column: str = "overall_sentiment", batch_size: int = 32):
    print("Loading sentiment model (first run downloads ~500MB)...")
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=128,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Only auto-label rows that have text and no existing label
    to_label_idx = [
        i for i, r in enumerate(rows)
        if r[text_column].strip() and not r[label_column].strip()
    ]
    texts = [rows[i][text_column][:512] for i in to_label_idx]

    print(f"Auto-labeling {len(texts)} rows with text "
          f"({len(rows) - len(texts)} rows skipped: no text or already labeled)")

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        results = classifier(batch)
        for offset, result in enumerate(results):
            row_idx = to_label_idx[start + offset]
            rows[row_idx][label_column] = LABEL_MAP.get(result["label"], "neutral")
        if start % (batch_size * 10) == 0:
            print(f"Labeled {start}/{len(texts)}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labeled_count = sum(1 for r in rows if r[label_column].strip())
    print(f"Done. {labeled_count}/{len(rows)} rows now have a sentiment label.")
    print("These are MODEL-GENERATED (weak) labels — note this in your report "
          "and spot-check a sample before relying on them for final results.")


def sample_check(csv_path: str, n: int, text_column: str = "text_corrected",
                  label_column: str = "overall_sentiment"):
    """Prints a random sample of labeled rows so you can manually verify quality."""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r[label_column].strip()]
    sample = random.sample(rows, min(n, len(rows)))
    for r in sample:
        print(f"[{r[label_column]:>8}] {r[text_column][:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/english_meme_labels.csv")
    parser.add_argument("--text_column", type=str, default="text_corrected")
    parser.add_argument("--label_column", type=str, default="overall_sentiment")
    parser.add_argument("--sample_check", type=int, default=0,
                         help="If set, just prints N random labeled rows for manual review instead of labeling.")
    args = parser.parse_args()

    if args.sample_check:
        sample_check(args.csv, args.sample_check, args.text_column, args.label_column)
    else:
        auto_label(args.csv, args.text_column, args.label_column)