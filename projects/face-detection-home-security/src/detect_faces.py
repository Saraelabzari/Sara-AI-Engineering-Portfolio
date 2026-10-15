"""Live webcam face detection demo. Press Q or Esc to quit."""
from __future__ import annotations

import argparse
import cv2

from face_detector import annotate_faces, detect_faces, load_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect face-like regions from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-width", type=int, default=30)
    parser.add_argument("--min-height", type=int, default=30)
    args = parser.parse_args()

    cascade = load_cascade()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Check permissions or try --camera 1."
        )

    print("Face detection demo running. Press Q or Esc to quit.")
    print("Frames are processed locally; this demo does not record or identify people.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            faces = detect_faces(frame, cascade, args.scale_factor, args.min_neighbors,
                                 (args.min_width, args.min_height))
            output = annotate_faces(frame, faces)
            cv2.putText(output, f"Detected: {len(faces)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
            cv2.imshow("Local Face Detection Demo", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
