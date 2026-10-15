from pathlib import Path

from PIL import Image
import pytest

from src.pcos_model import validate_dataset_dir


def create_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_validate_dataset_dir_accepts_two_classes(tmp_path):
    for label, color in [("class_a", "white"), ("class_b", "black")]:
        for i in range(5):
            create_image(tmp_path / label / f"{i}.png", color)
    validate_dataset_dir(tmp_path)


def test_validate_dataset_dir_requires_two_classes(tmp_path):
    for i in range(10):
        create_image(tmp_path / "class_a" / f"{i}.png", "white")
    with pytest.raises(ValueError, match="two class"):
        validate_dataset_dir(tmp_path)


def test_validate_dataset_dir_requires_ten_images(tmp_path):
    create_image(tmp_path / "class_a" / "a.png", "white")
    create_image(tmp_path / "class_b" / "b.png", "black")
    with pytest.raises(ValueError, match="at least 10"):
        validate_dataset_dir(tmp_path)


def test_validate_dataset_dir_requires_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_dir(tmp_path / "missing")
