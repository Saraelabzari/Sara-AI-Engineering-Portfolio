# ID Document Recognition System

A local OCR prototype that preprocesses an uploaded document image and extracts readable text with Tesseract OCR through a Streamlit interface.

> **Status:** Portfolio prototype. It has not been validated across document types, layouts, or languages. It is not an identity-verification, authenticity-checking, or fraud-detection system.

## Features

- Upload PNG, JPG/JPEG, or TIFF images
- Correct EXIF orientation and preprocess images with grayscale conversion and Otsu thresholding
- Extract text with Tesseract OCR
- Select English, Arabic, or English + Arabic OCR
- Review recognized text and download it as a text file
- Basic upload-size and image-dimension validation
- Reusable OCR functions and unit tests

## Requirements

- Python 3.10+
- Tesseract OCR installed separately on the operating system
- The Tesseract language data for the selected language (English and/or Arabic)

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To install Tesseract, follow the installation instructions for your operating system and make sure its executable is on PATH. The Python package `pytesseract` is only a wrapper; it does not install the Tesseract executable or its language packs.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The tests cover image preprocessing, input validation, supported languages, and page-segmentation settings. They do not require a live OCR call for the validation tests. Run them in your environment before presenting the project as verified.

## How it works

1. Validate the image object and reject images beyond a pixel-count limit.
2. Apply EXIF orientation correction, convert to RGB/grayscale, and use Otsu thresholding.
3. Pass the processed image to Tesseract with the requested language and page segmentation mode.
4. Display extracted text for human review and offer a text download.

## Privacy and limitations

Identity documents contain highly sensitive information. Use synthetic or redacted sample images for demonstrations. Do not commit document images or extracted personal data to GitHub, and do not use this prototype on a shared or untrusted computer. The app does not intentionally save uploads, but the hosting environment and deployment configuration also matter.

OCR may misread names, numbers, dates, or characters. Always review output manually. Do not use OCR text as proof of identity or as the sole basis for eligibility, financial, legal, or security decisions. A real deployment would require consent, security and legal review, access controls, secure retention policies, and extensive validation.
