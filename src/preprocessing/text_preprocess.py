"""
Text preprocessing for IndicSent-M.

Steps implemented (per project Section 5.1):
1. Missing value handling
2. Text cleaning
3. Normalization (code-mixed text, romanized Indic text, emojis, repeated chars, punctuation)
4. Tokenization (via IndicBERT tokenizer, called from the dataset class, not here)
"""

import re
import emoji
import pandas as pd


def handle_missing_text(df: pd.DataFrame, text_column: str, image_column: str) -> pd.DataFrame:
    """Drop rows with no text AND no image. Rows with text but no image, or
    image but no text, are kept and flagged so the model can fall back to
    the available modality."""
    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    has_text = df[text_column].str.strip().str.len() > 0
    has_image = df[image_column].notna() & (df[image_column].astype(str).str.strip() != "")
    df = df[has_text | has_image].reset_index(drop=True)
    df["has_text"] = has_text[has_text | has_image].reset_index(drop=True)
    df["has_image"] = has_image[has_text | has_image].reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Remove noisy formatting while preserving sentiment-bearing content
    (emojis, punctuation and case are preserved here; normalized later)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"\s+", " ", text)                         # collapse whitespace
    return text.strip()


def reduce_repeated_characters(text: str) -> str:
    """'soooo gooood' -> 'soo good' (caps repeats at 2 to keep emphasis signal
    without exploding vocabulary size)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_emojis(text: str) -> str:
    """Convert emojis to text tokens so sentiment-bearing emojis survive
    tokenization instead of being dropped as unknown characters."""
    return emoji.demojize(text, delimiters=(" :", ": "))


def normalize_punctuation(text: str) -> str:
    """Collapse excessive punctuation ('!!!!' -> '!') but keep at least one,
    since punctuation intensity can carry sentiment."""
    text = re.sub(r"([!?.]){2,}", r"\1\1", text)
    return text


def preprocess_text(text: str) -> str:
    """Full pipeline for a single text sample."""
    text = clean_text(text)
    text = normalize_emojis(text)
    text = reduce_repeated_characters(text)
    text = normalize_punctuation(text)
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_column: str, image_column: str) -> pd.DataFrame:
    """Run the full text preprocessing pipeline over a dataframe."""
    df = handle_missing_text(df, text_column, image_column)
    df[text_column] = df[text_column].apply(preprocess_text)
    return df


if __name__ == "__main__":
    # Quick manual test
    sample = pd.DataFrame({
        "text_corrected": ["Yeh toooo good hai!!!! 😂😂 check http://x.com @someone", None],
        "image_name": ["meme1.jpg", "meme2.jpg"],
    })
    out = preprocess_dataframe(sample, "text_corrected", "image_name")
    print(out)
