# Face Detection Home Security Demo

A local computer-vision prototype that detects face-like regions in webcam frames using OpenCV's Haar cascade classifier. It is a detection demo, **not a person-identification or access-control system**.

## Features

- Live webcam preview
- Adjustable detection parameters through command-line options
- Bounding-box overlays and a detected-face count
- Reusable detection and annotation functions
- Unit tests that do not require a webcam
- No image saving or recording by default

## Requirements

- Python 3.10+
- Webcam (for the live demo)
- Windows, macOS, or Linux with camera permissions

## Install

From this project directory:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run

```bash
python src/detect_faces.py
```

Press **Q** or **Esc** to quit. If your webcam is not camera index 0, try:

```bash
python src/detect_faces.py --camera 1
```

Tune the detector if needed:

```bash
python src/detect_faces.py --scale-factor 1.15 --min-neighbors 6 --min-width 40 --min-height 40
```

## Tests

```bash
python -m pytest tests/
```

The tests check cascade loading, empty-frame handling, output structure, and annotation behavior. They have been added to the repository but still need to be run locally.

## How it works

1. OpenCV reads frames from the selected camera.
2. Each frame is converted to grayscale and histogram-equalized.
3. A pretrained Haar cascade proposes face-like regions.
4. Rectangles and a count are drawn on a copy of the frame.
5. The frame is displayed locally until the user exits.

## Limitations and privacy

Haar cascades can miss faces or produce false detections depending on lighting, pose, occlusion, and camera quality. Detection does not establish a person's identity. The demo does not record frames or control locks. Obtain consent before using a camera around people, and do not use this prototype as the sole basis for security decisions.
