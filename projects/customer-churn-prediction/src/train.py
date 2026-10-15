"""Train/evaluate the customer churn pipeline and save reproducible artifacts.

Run from this project directory:
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from churn_model import build_pipeline, evaluate_model, load_data, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a customer churn classifier.")
    parser.add_argument("--data", type=Path,
                        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    df = load_data(args.data)
    y = df.pop("Churn")
    X = df.drop(columns=["customerID"], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics, args.output_dir)

    print("\nHeld-out test metrics")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name}: {metrics[name]:.4f}")
    print("\nClassification report")
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    print(report.round(3).to_string())

    matrix = metrics["confusion_matrix"]
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["No churn", "Churn"]
    ).plot(values_format="d")
    plt.title("Customer Churn — Confusion Matrix")
    plt.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"\nSaved model, metrics, and confusion matrix to: {args.output_dir}")


if __name__ == "__main__":
    main()
