"""Train and evaluate the payment fraud model.

Run from this project directory:
python src/train.py --data data/creditcard.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split

from fraud_model import build_model, evaluate_majority_baseline, evaluate_model, load_dataset, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud detection baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--estimators", type=int, default=200)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    baseline = evaluate_majority_baseline(X_train, X_test, y_train, y_test)
    print("Majority-class baseline:")
    print(pd.Series({k: v for k, v in baseline.items() if k != "confusion_matrix"}).round(4))

    model = build_model(n_estimators=args.estimators)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print("\nRandom Forest metrics:")
    print(pd.Series({k: v for k, v in metrics.items()
                     if k not in {"confusion_matrix", "classification_report"}}).round(4))
    print("\nClassification report:")
    print(pd.DataFrame(metrics["classification_report"]).transpose().round(3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=["Legitimate", "Fraud"]
    ).plot(values_format="d")
    plt.title("Fraud Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    from sklearn.metrics import PrecisionRecallDisplay
    _, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Fraud Detection — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(args.output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    save_results(model, metrics, args.model_dir)
    print(f"\nSaved plots to {args.output_dir}; model and metrics to {args.model_dir}")


if __name__ == "__main__":
    main()
