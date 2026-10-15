"""Reusable training and evaluation functions for transaction fraud classification."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


TARGET = "Class"


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the common creditcard.csv schema and validate binary fraud labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError("Expected target column 'Class' (1 = fraud, 0 = legitimate).")
    y = pd.to_numeric(frame.pop(TARGET), errors="coerce")
    valid = y.isin([0, 1])
    X, y = frame.loc[valid].copy(), y.loc[valid].astype(int)
    if len(y) == 0 or y.nunique() < 2:
        raise ValueError("Dataset must contain both legitimate and fraudulent examples.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All input features must be numeric in this starter pipeline.")
    return X, y


def build_model(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    """Create a pipeline with median imputation and class-weighted Random Forest."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive.")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate ranking and threshold metrics; do not rely on accuracy alone."""
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probability)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, prediction, labels=[0, 1],
            target_names=["Legitimate", "Fraud"], zero_division=0, output_dict=True
        ),
    }


def evaluate_majority_baseline(X_train, X_test, y_train, y_test) -> dict:
    """Evaluate a simple majority-class baseline for comparison."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return {
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
    }


def save_results(model, metrics: dict, output_dir: str | Path) -> None:
    """Save a fitted estimator and JSON metrics to a local output folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fraud_classifier.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
