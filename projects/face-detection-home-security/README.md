# Face Detection Home Security System

A privacy-conscious computer-vision prototype that detects faces from a **local webcam** using OpenCV. It draws bounding boxes and shows a live face count.

> **Status:** Starter implementation. The code has been added to this portfolio but has not yet been tested with a webcam. This is a face **detection** demo, not a facial-recognition or identity-verification system.

## Features

- Live webcam capture
- Face detection with OpenCV's pretrained Haar cascade
- Bounding boxes and face count overlay
- Keyboard controls: press `q` or `Esc` to exit
- No face images saved, no cloud upload, no identification

## Project structure

```text
face-detection-home-security/
├── src/
│   └── detect_faces.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

Use Python 3.10+ and a computer with a webcam.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/detect_faces.py
```

For another camera index:

```bash
python src/detect_faces.py --camera 1
```

## How it works

1. Open the local webcam.
2. Convert frames to grayscale.
3. Run OpenCV's built-in frontal-face Haar cascade detector.
4. Draw bounding boxes around detected faces.
5. Display the annotated live feed without saving video or images.

## Limitations and next steps

Haar cascades may miss faces under poor lighting, occlusion, or non-frontal angles and may produce false positives. The system does not determine whether someone is authorized or whether an intrusion occurred. Future extensions could include an optional, consent-based alert mechanism, evaluation on varied lighting conditions, and a modern face detector.

## Privacy and safety

Only use cameras in spaces where you have permission to record or monitor. This starter implementation processes video locally and does not store frames.
