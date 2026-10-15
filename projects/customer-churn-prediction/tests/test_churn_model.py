import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.churn_model import build_pipeline, evaluate_model, load_data


def sample_frame():
    return pd.DataFrame({
        "customerID": [f"C{i:03}" for i in range(40)],
        "tenure": list(range(40)),
        "MonthlyCharges": [25 + (i % 12) * 4 for i in range(40)],
        "Contract": ["Month-to-month" if i % 3 else "One year" for i in range(40)],
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(40)],
    })


def test_load_data_normalizes_target_and_numeric_column(tmp_path):
    path = tmp_path / "telco.csv"
    sample_frame().assign(TotalCharges="").to_csv(path, index=False)
    loaded = load_data(path)
    assert set(loaded["Churn"].unique()) == {0, 1}
    assert loaded["TotalCharges"].isna().all()


def test_load_data_requires_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"tenure": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Churn"):
        load_data(path)


def test_pipeline_fits_and_returns_metrics():
    frame = sample_frame()
    X = frame.drop(columns="Churn")
    y = frame["Churn"].map({"Yes": 1, "No": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
