"""Train and evaluate a customer churn classifier.

Expected input: IBM Telco Customer Churn CSV with a 'Churn' target column.
Run: python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download the IBM Telco Customer Churn CSV "
            "and pass its path using --data."
        )
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' was not found.")
    df = df.copy()
    # In the common IBM dataset, TotalCharges can contain blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["Churn"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the Telco Customer Churn CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    df = load_data(args.data)
    y = df.pop("Churn").astype(str).str.strip().map({"Yes": 1, "No": 0})
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)

    # Customer identifiers are not predictive features.
    df = df.drop(columns=["customerID"], errors="ignore")
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = df.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report")
    print(classification_report(y_test, predictions, target_names=["No churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["No churn", "Churn"], values_format="d"
    )
    plt.title("Customer Churn — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
