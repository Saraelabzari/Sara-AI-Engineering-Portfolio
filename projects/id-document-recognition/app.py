"""Local OCR demo for extracting text from an uploaded document image.

Use synthetic or redacted sample documents. This is not identity verification.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import TesseractNotFoundError


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert to grayscale and apply a light threshold for OCR."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Otsu thresholding can help with high-contrast scanned documents.
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def extract_text(image: Image.Image, language: str = "eng") -> str:
    processed = preprocess_image(image)
    return pytesseract.image_to_string(processed, lang=language).strip()


def main() -> None:
    st.set_page_config(page_title="ID Document OCR Demo", page_icon="🪪")
    st.title("ID Document Recognition — OCR Demo")
    st.warning(
        "Privacy reminder: use only synthetic or redacted documents. "
        "This app extracts text; it does not verify identity or detect fraud."
    )
    uploaded = st.file_uploader(
        "Upload a document image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    language = st.selectbox("OCR language", ["eng", "ara", "eng+ara"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)
            if st.button("Extract text"):
                with st.spinner("Running OCR locally..."):
                    text = extract_text(image, language=language)
                if text:
                    st.text_area("Recognized text — review carefully", text, height=260)
                    st.download_button(
                        "Download extracted text",
                        data=text,
                        file_name="recognized_text.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("No text was detected. Try a clearer, well-lit image.")
        except TesseractNotFoundError:
            st.error(
                "Tesseract OCR was not found. Install the Tesseract executable "
                "on your computer and make sure it is on PATH."
            )
        except Exception as exc:
            st.error(f"Unable to process this image: {exc}")


if __name__ == "__main__":
    main()
