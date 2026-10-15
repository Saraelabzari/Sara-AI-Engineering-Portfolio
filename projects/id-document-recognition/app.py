"""Streamlit interface for a privacy-conscious OCR demonstration."""
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from pytesseract import TesseractNotFoundError, TesseractError

from src.ocr_utils import extract_text

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def main() -> None:
    st.set_page_config(page_title="ID Document OCR Demo", page_icon="🪪")
    st.title("ID Document Recognition — OCR Demo")
    st.warning(
        "Privacy reminder: use synthetic or redacted documents only. "
        "This app extracts text; it does not verify identity, authenticity, or fraud."
    )
    st.caption(
        "Processing is performed locally by this application. Do not upload sensitive "
        "documents to a shared or untrusted computer."
    )

    uploaded = st.file_uploader(
        "Upload a document image (PNG, JPG, or TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )
    language = st.selectbox("OCR language", ["eng", "ara", "eng+ara"])
    if uploaded is None:
        st.info("Upload a test image to begin.")
        return
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("The uploaded file exceeds the 10 MB demo limit.")
        return

    try:
        image = Image.open(uploaded)
        image.verify()
        uploaded.seek(0)
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be decoded as a supported image.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Extract text", type="primary"):
        try:
            with st.spinner("Running OCR locally…"):
                text = extract_text(image, language=language)
        except TesseractNotFoundError:
            st.error(
                "Tesseract OCR was not found. Install the Tesseract executable "
                "and ensure it is available on PATH."
            )
            return
        except TesseractError as exc:
            st.error(
                "Tesseract could not process this image or language pack. "
                "Confirm that the selected language data is installed."
            )
            st.caption(str(exc))
            return
        except (TypeError, ValueError) as exc:
            st.error(str(exc))
            return

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


if __name__ == "__main__":
    main()
