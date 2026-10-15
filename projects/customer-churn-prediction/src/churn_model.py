"""Reusable preprocessing, training, and evaluation for churn prediction."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
IDENTIFIER_COLUMNS = ("customerID",)


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the IBM Telco CSV and normalize its target and TotalCharges column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download the IBM Telco Customer Churn CSV "
            "and pass its path with --data."
        )
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r} was not found.")
    frame = frame.copy()
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    frame[TARGET] = frame[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    frame = frame.dropna(subset=[TARGET])
    frame[TARGET] = frame[TARGET].astype(int)
    if frame[TARGET].nunique() < 2:
        raise ValueError("The target must contain both churn classes: Yes and No.")
    return frame


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing within the model pipeline to prevent train/test leakage."""
    features = X.drop(columns=list(IDENTIFIER_COLUMNS), errors="ignore")
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns found.")
    preprocess = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return metrics useful for an imbalanced binary classification task."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["No churn", "Churn"], zero_division=0, output_dict=True
        ),
    }


def save_artifacts(model: Pipeline, metrics: dict, output_dir: str | Path) -> None:
    """Save a fitted model and machine-readable metrics locally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "churn_model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
