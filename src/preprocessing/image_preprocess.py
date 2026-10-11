"""
Image preprocessing for IndicSent-M.

Steps implemented (per project Section 6):
1. Image validation (corrupted / missing / unsupported format)
2. Resizing to ViT input size
3. Normalization (ImageNet stats, standard for pretrained ViT/ResNet)
4. Light data augmentation (train split only)
5. OCR extraction of embedded meme text (optional, supplements caption text)
"""

import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def validate_image(image_path: str) -> bool:
    """Returns True if the image exists, has a supported extension, and can
    be opened without error."""
    path = Path(image_path)
    if not path.exists():
        return False
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def get_transforms(image_size: int = 224, train: bool = False) -> T.Compose:
    """Returns the torchvision transform pipeline. Light augmentation only
    for training; aggressive augmentation is avoided since it can distort
    meme meaning and embedded text."""
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(image_path: str, image_size: int = 224, train: bool = False):
    """Load, validate and transform a single image. Returns None if invalid
    so the caller can fall back to the text-only branch."""
    if not validate_image(image_path):
        return None
    transform = get_transforms(image_size=image_size, train=train)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return transform(img)


def extract_text_from_image(image_path: str) -> str:
    """OCR extraction of text burned into the meme image. Returns an empty
    string if OCR is unavailable or the image is invalid."""
    if not OCR_AVAILABLE or not validate_image(image_path):
        return ""
    try:
        with Image.open(image_path) as img:
            return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def filter_valid_images(image_dir: str, filenames) -> list:
    """Given a list of filenames, return only those that pass validation."""
    valid = []
    for fname in filenames:
        full_path = os.path.join(image_dir, str(fname))
        if validate_image(full_path):
            valid.append(fname)
    return valid


if __name__ == "__main__":
    # Quick manual test (requires a real image to run end to end)
    print(f"OCR available: {OCR_AVAILABLE}")
    print(f"Supported formats: {SUPPORTED_FORMATS}")
