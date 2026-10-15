"""Reusable classification evaluation and pipeline utilities."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METRIC_COLUMNS = [
    "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC"
]


def validate_dataset(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Validate a dataset and return features plus a clean categorical target."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty or len(data) < 10:
        raise ValueError("Dataset must contain at least 10 rows.")
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    y = data[target]
    valid = y.notna()
    X = data.loc[valid].drop(columns=[target]).copy()
    y = y.loc[valid].astype(str)
    if X.empty or X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    if X.select_dtypes(include=["number"]).shape[1] == 0 and X.select_dtypes(exclude=["number"]).shape[1] == 0:
        raise ValueError("No usable feature columns were found.")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build train-only numeric and categorical preprocessing."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
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
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers)


def default_models() -> dict[str, Any]:
    """Return fresh estimator instances for a fair baseline comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def classification_metrics(y_true, y_pred, y_score=None, classes=None) -> dict[str, float]:
    """Calculate common classification metrics, adding ROC-AUC when supported."""
    result = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "Recall (macro)": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "F1 (macro)": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
    }
    if y_score is not None and classes is not None:
        try:
            if len(classes) == 2:
                result["ROC-AUC"] = float(roc_auc_score(
                    y_true, y_score[:, 1], labels=classes
                ))
            elif y_score.shape[1] == len(classes):
                result["ROC-AUC"] = float(roc_auc_score(
                    y_true, y_score, multi_class="ovr", average="macro", labels=classes
                ))
        except ValueError:
            pass
    return result


def compare_models(data: pd.DataFrame, target: str, test_size: float = 0.25,
                   random_state: int = 42) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], tuple]:
    """Train default classifiers on the same stratified split and return metrics/models."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")
    X, y = validate_dataset(data, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    results = []
    trained: dict[str, dict[str, Any]] = {}
    for name, estimator in default_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        score = None
        if hasattr(pipeline, "predict_proba"):
            score = pipeline.predict_proba(X_test)
        metrics = classification_metrics(
            y_test, prediction, score, getattr(pipeline, "classes_", None)
        )
        results.append({"Model": name, **metrics})
        trained[name] = {
            "pipeline": pipeline,
            "predictions": prediction,
            "y_test": y_test,
            "classes": getattr(pipeline, "classes_", None),
        }
    result_frame = pd.DataFrame(results).set_index("Model")
    split = (X_train, X_test, y_train, y_test)
    return result_frame, trained, split
