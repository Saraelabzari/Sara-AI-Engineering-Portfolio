"""Local webcam face detection demo using OpenCV.

This detects faces, not identities. It does not record or upload video.
Press 'q' or Escape to exit.
"""
from __future__ import annotations

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Local webcam face detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    args = parser.parse_args()

    if args.scale_factor <= 1:
        parser.error("--scale-factor must be greater than 1")
    if args.min_neighbors < 0:
        parser.error("--min-neighbors must be nonnegative")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Unable to open camera {args.camera}. Check camera permissions "
            "and whether another app is using it."
        )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Unable to read webcam frame; stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(30, 30),
            )

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2,
            )
            cv2.imshow("Face Detection - press Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
