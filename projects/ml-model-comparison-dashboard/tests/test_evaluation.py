import pandas as pd
import pytest

from src.evaluation import (
    build_preprocessor,
    classification_metrics,
    compare_models,
    validate_dataset,
)


@pytest.fixture
def data():
    return pd.DataFrame({
        "age": [22, 25, 35, 45, 28, 50, 41, 31, 24, 39, 52, 29, 44, 33, 26, 48, 37, 23, 55, 32],
        "plan": ["basic", "basic", "pro", "pro", "basic", "pro", "pro", "basic", "basic", "pro", "pro", "basic", "pro", "basic", "basic", "pro", "pro", "basic", "pro", "basic"],
        "churn": ["yes", "yes", "no", "no", "yes", "no", "no", "yes", "yes", "no", "no", "yes", "no", "yes", "yes", "no", "no", "yes", "no", "yes"],
    })


def test_validate_dataset_splits_features_and_target(data):
    X, y = validate_dataset(data, "churn")
    assert "churn" not in X.columns
    assert len(X) == len(y)
    assert y.nunique() == 2


def test_missing_target_is_rejected(data):
    with pytest.raises(ValueError, match="not found"):
        validate_dataset(data, "missing")


def test_single_class_target_is_rejected(data):
    data["churn"] = "yes"
    with pytest.raises(ValueError, match="at least two classes"):
        validate_dataset(data, "churn")


def test_preprocessor_handles_numeric_and_categorical(data):
    X, _ = validate_dataset(data, "churn")
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] >= 2


def test_metrics_report_macro_scores():
    result = classification_metrics(
        ["no", "yes", "yes", "no"], ["no", "yes", "no", "no"]
    )
    assert result["Accuracy"] == 0.75
    assert 0 <= result["F1 (macro)"] <= 1


def test_compare_models_returns_three_models(data):
    metrics, models, split = compare_models(data, "churn", test_size=0.25)
    assert len(metrics) == 3
    assert set(models) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert all(0 <= metrics.loc[name, "Accuracy"] <= 1 for name in metrics.index)
    assert len(split[1]) == 5


def test_test_size_is_validated(data):
    with pytest.raises(ValueError, match="test_size"):
        compare_models(data, "churn", test_size=0.8)
