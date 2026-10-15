"""Reusable utilities for experimental ultrasound image classification.

Research only: this package is not a medical diagnostic tool.
"""
from __future__ import annotations

from pathlib import Path

from torchvision import models
from torch import nn


def validate_dataset_dir(data_dir: str | Path) -> None:
    """Validate the ImageFolder directory structure and minimum class count."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    classes = [p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two class subfolders.")
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_count = sum(
        1 for class_dir in classes for file in class_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in supported
    )
    if image_count < 10:
        raise ValueError(f"Need at least 10 supported images; found {image_count}.")


def build_resnet18(num_classes: int, pretrained: bool = True):
    """Return a ResNet-18 with a classification head for the provided labels."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
