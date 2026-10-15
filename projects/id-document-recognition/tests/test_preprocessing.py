import numpy as np
import pytest
from PIL import Image

from src.ocr_utils import (
    MAX_IMAGE_PIXELS,
    extract_text,
    preprocess_image,
    validate_image,
)


def test_preprocess_returns_grayscale_uint8_array():
    image = Image.new("RGB", (100, 60), color="white")
    result = preprocess_image(image)
    assert result.shape == (60, 100)
    assert result.dtype == np.uint8


def test_preprocess_handles_rgba_input():
    image = Image.new("RGBA", (20, 12), color=(255, 255, 255, 128))
    result = preprocess_image(image)
    assert result.shape == (12, 20)


def test_validate_rejects_non_pil_input():
    with pytest.raises(TypeError):
        validate_image("not an image")


def test_validate_rejects_oversized_image():
    # Pillow can create a tiny backing image without allocating the full raster.
    image = Image.new("RGB", (MAX_IMAGE_PIXELS + 1, 1), color="white")
    with pytest.raises(ValueError, match="too large"):
        validate_image(image)


def test_extract_rejects_unsupported_language_before_calling_tesseract():
    image = Image.new("RGB", (10, 10), color="white")
    with pytest.raises(ValueError, match="Unsupported OCR language"):
        extract_text(image, language="fra")


def test_extract_rejects_invalid_page_segmentation_mode():
    image = Image.new("RGB", (10, 10), color="white")
    with pytest.raises(ValueError, match="psm"):
        extract_text(image, psm=99)
