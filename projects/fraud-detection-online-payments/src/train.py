"""Train a fraud-detection baseline on creditcard.csv.

Expected columns: Time, Amount, Class (1 = fraud, 0 = legitimate).
Run: python src/train.py --data data/creditcard.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Download the public creditcard.csv "
            "dataset and pass its path with --data."
        )

    df = pd.read_csv(args.data)
    required = {"Class"}
    if not required.issubset(df.columns):
        raise ValueError("Expected target column 'Class' in the dataset.")
    y = pd.to_numeric(df.pop("Class"), errors="coerce")
    valid = y.isin([0, 1])
    X, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError("Dataset must contain both legitimate and fraudulent examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    print("Majority-class baseline:")
    print(classification_report(y_test, baseline_pred, zero_division=0))

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, score)
    print("\nRandom Forest:")
    print(classification_report(y_test, pred, target_names=["Legitimate", "Fraud"], zero_division=0))
    print(f"Average precision (PR-AUC summary): {ap:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Legitimate", "Fraud"], values_format="d"
    )
    plt.title("Fraud Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud Detection — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(args.output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    dump(model, args.model_dir / "fraud_classifier.joblib")
    print(f"Saved evaluation plots to {args.output_dir} and model to {args.model_dir}")


if __name__ == "__main__":
    main()
