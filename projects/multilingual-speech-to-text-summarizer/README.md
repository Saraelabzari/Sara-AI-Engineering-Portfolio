# Multilingual Speech-to-Text & Summarizer

A portfolio app by **Sara Elabzari** that transcribes audio with OpenAI Whisper and generates multilingual summaries with Hugging Face Transformers.

## Features
- Upload WAV, MP3, M4A, MP4, WEBM, OGG, and FLAC audio
- Automatic speech-language detection or optional language selection
- Transcription in the original language or translation to English
- Optional timestamps, multilingual summaries, and TXT/JSON exports
- Local inference after model downloads; no paid API key required

## Requirements
Python 3.10–3.11 recommended, FFmpeg, internet for first model downloads, and sufficient RAM.

## Windows setup
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Install FFmpeg from https://ffmpeg.org/download.html and ensure `ffmpeg -version` works.

## Run unit tests

Install the development dependencies and run the lightweight helper tests:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

The tests use a fake summarizer for unit tests and do not download model weights. The full app has not yet been validated end-to-end on all operating systems.

## Models
- Whisper: speech recognition and optional translation to English.
- `csebuetnlp/mT5_multilingual_XLSum`: multilingual summarization model.

## Limitations
Model quality varies by language, accent, audio quality, and domain. Action-item detection is a simple keyword heuristic, not a dedicated extraction model. Review outputs and do not upload private recordings without authorization. This is a portfolio prototype, not a validated high-stakes system.
