"""Research starter for ultrasound image classification using transfer learning.

Expected folder layout:
  data/ultrasound/<class_name>/*.png (or other PIL-readable images)

This is not a medical diagnostic system. Use only licensed, de-identified data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from pcos_model import build_resnet18, validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ultrasound"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    validate_dataset_dir(args.data)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if len(dataset) < 10 or len(dataset.classes) < 2:
        raise ValueError("Need at least 10 images and at least 2 class folders for a basic experiment.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Weights are fetched by torchvision if not already cached. For offline use,
    # provide the weights locally or adapt this to a fully offline workflow.
    model = build_resnet18(num_classes=len(dataset.classes), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs} — train loss: {running_loss / len(train_set):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    names = [str(name) for name in dataset.classes]
    print(classification_report(all_labels, all_preds, labels=list(range(len(names))), target_names=names, zero_division=0))
    print("Validation accuracy:", round(accuracy_score(all_labels, all_preds), 4))
    print("Validation macro F1:", round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds, labels=list(range(len(names))), display_labels=names,
        xticks_rotation="vertical", values_format="d"
    )
    plt.title("Ultrasound Image Classification — Experimental Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    torch.save({"state_dict": model.state_dict(), "classes": names}, args.model_dir / "resnet18_classifier.pt")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved model checkpoint to {args.model_dir / 'resnet18_classifier.pt'}")


if __name__ == "__main__":
    main()
