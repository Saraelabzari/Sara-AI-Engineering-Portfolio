# ID Document Recognition System

An OCR-based document-processing prototype that extracts text from a user-provided image or scan, then displays the recognized text for review.

> **Status:** Starter implementation. It has not been validated across document types or languages and is not an identity-verification or fraud-detection system.

## Features

- Upload an image of a document
- Image preprocessing for OCR
- Text extraction with Tesseract OCR
- Copyable text output
- No cloud upload or persistent storage by this application

## Requirements

- Python 3.10+
- Tesseract OCR installed separately on your operating system
- Use only test documents or documents you are authorized to process

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Install Tesseract OCR using the official instructions for your operating system and ensure the executable is on your PATH. For Windows, if it is not detected automatically, configure the executable path in the code.

## Privacy and security

Identity documents contain sensitive personal information. Use synthetic or redacted sample images for demonstrations, avoid committing document images or extracted personal data to GitHub, and delete temporary copies when finished. Do not deploy this prototype for real identity verification without security review, consent, legal review, and robust testing.

## Limitations

OCR accuracy depends on image quality, orientation, language, font, and document layout. Extracted text may be incorrect. Always require human review and do not treat OCR output as proof of identity.
