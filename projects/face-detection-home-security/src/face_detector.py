"""Reusable OpenCV face detection helpers using a Haar cascade.

This module detects face-like regions only. It does not identify people.
"""
from __future__ import annotations

from pathlib import Path
import cv2


def load_cascade(path: str | Path | None = None):
    """Load OpenCV's frontal-face Haar cascade or raise a useful error."""
    cascade_path = str(path) if path else str(
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Could not load Haar cascade file: {cascade_path}")
    return cascade


def detect_faces(frame, cascade, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple[int, int] = (30, 30)):
    """Return face bounding boxes (x, y, width, height) for an image frame."""
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Input frame is empty.")
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1.0.")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be at least 1.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=min_size
    )
    return [tuple(int(v) for v in face) for face in faces]


def annotate_faces(frame, faces):
    """Return a copy of the frame with rectangles around detected faces."""
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Input frame is empty.")
    output = frame.copy()
    for x, y, width, height in faces:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.putText(output, "Face detected", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    return output
