"""Reusable transfer-learning utilities for folder-based image classification."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split


def build_transforms(image_size: int = 224):
    """Create consistent training and evaluation transforms."""
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels.")
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_transform, eval_transform


def build_model(num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = True) -> nn.Module:
    """Build a ResNet-18 classifier, replacing its final layer for the dataset."""
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def make_loaders(data_dir: str | Path, batch_size: int = 16,
                 val_fraction: float = 0.2, seed: int = 42,
                 image_size: int = 224, num_workers: int = 0):
    """Load ImageFolder data and create reproducible train/validation subsets.

    The folder must contain one subdirectory per class. Since ImageFolder uses one
    transform per dataset instance, separate dataset instances are used for train
    augmentation and deterministic validation preprocessing.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = ImageFolder(root, transform=train_transform)
    eval_dataset = ImageFolder(root, transform=eval_transform)
    if len(train_dataset.classes) < 2:
        raise ValueError("Dataset must contain at least two class folders.")
    if len(train_dataset) < 4:
        raise ValueError("Dataset must contain at least four images.")
    val_count = max(1, int(round(len(train_dataset) * val_fraction)))
    train_count = len(train_dataset) - val_count
    if train_count < 2:
        raise ValueError("Not enough images for a training subset.")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_count, val_count], generator=generator
    )
    # Point validation subset at deterministic transforms while preserving indices.
    val_subset.dataset = eval_dataset
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, train_dataset.classes


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer=None) -> tuple[float, float]:
    """Run one training or evaluation epoch and return mean loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    if total == 0:
        raise ValueError("DataLoader produced no samples.")
    return total_loss / total, correct / total


def save_checkpoint(path: str | Path, model: nn.Module,
                    class_names: Sequence[str], image_size: int = 224) -> None:
    """Save model weights and metadata without pickling the whole model object."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": image_size,
        "architecture": "resnet18",
    }, destination)
