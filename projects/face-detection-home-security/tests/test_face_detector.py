import cv2
import numpy as np
import pytest

from src.face_detector import annotate_faces, detect_faces, load_cascade


def test_load_default_cascade():
    cascade = load_cascade()
    assert not cascade.empty()


def test_detect_faces_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        detect_faces(None, load_cascade())


def test_detect_faces_returns_list_for_blank_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = detect_faces(frame, load_cascade())
    assert isinstance(faces, list)


def test_annotate_does_not_mutate_input():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    original = frame.copy()
    result = annotate_faces(frame, [(10, 10, 20, 20)])
    assert np.array_equal(frame, original)
    assert not np.array_equal(result, original)


def test_invalid_scale_factor():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="scale_factor"):
        detect_faces(frame, load_cascade(), scale_factor=1.0)
