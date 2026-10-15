"""Image preprocessing and OCR helpers for the document-recognition demo."""
from __future__ import annotations

import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageOps
from pytesseract import TesseractNotFoundError

SUPPORTED_LANGUAGES = {"eng", "ara", "eng+ara"}
MAX_IMAGE_PIXELS = 20_000_000


def validate_image(image: Image.Image) -> None:
    """Reject empty or excessively large images before processing."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({width * height:,} pixels); "
            f"the limit is {MAX_IMAGE_PIXELS:,} pixels."
        )


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Orient, grayscale, and threshold an image for OCR."""
    validate_image(image)
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(oriented)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def extract_text(image: Image.Image, language: str = "eng", psm: int = 6) -> str:
    """Extract OCR text from an image using an installed Tesseract language pack."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported OCR language {language!r}. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )
    if psm not in range(3, 14):
        raise ValueError("psm must be an integer from 3 to 13.")
    processed = preprocess_image(image)
    config = f"--psm {psm}"
    try:
        return pytesseract.image_to_string(
            processed, lang=language, config=config
        ).strip()
    except TesseractNotFoundError:
        raise
