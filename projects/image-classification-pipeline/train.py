"""Train a ResNet-18 transfer-learning baseline on an ImageFolder dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.classifier import build_model, make_loaders, run_epoch, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download pretrained weights")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune all model layers")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")

    torch.manual_seed(args.seed)
    train_loader, val_loader, classes = make_loaders(
        args.data_dir, batch_size=args.batch_size,
        val_fraction=args.val_fraction, seed=args.seed,
        image_size=args.image_size, num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(classes), pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    history = []
    best_val_accuracy = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"

    print(f"Classes: {classes}")
    print(f"Device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_accuracy:.3f} | "
            f"val loss={val_loss:.4f} acc={val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path, model, classes, image_size=args.image_size
            )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
