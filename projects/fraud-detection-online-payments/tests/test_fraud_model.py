import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.fraud_model import build_model, evaluate_model, load_dataset


def synthetic_transactions():
    rows = []
    for i in range(100):
        fraud = int(i % 10 == 0)
        rows.append({"Time": i * 10, "Amount": 900 if fraud else 20 + i % 40,
                     "V1": -4 if fraud else 0.1 * (i % 5), "Class": fraud})
    return pd.DataFrame(rows)


def test_load_dataset(tmp_path):
    path = tmp_path / "creditcard.csv"
    synthetic_transactions().to_csv(path, index=False)
    X, y = load_dataset(path)
    assert "Class" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_load_dataset_requires_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Amount": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Class"):
        load_dataset(path)


def test_pipeline_and_metrics():
    X, y = load_dataset_from_frame(synthetic_transactions())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model(n_estimators=10)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert len(metrics["confusion_matrix"]) == 2


def load_dataset_from_frame(frame):
    y = frame["Class"].astype(int)
    return frame.drop(columns="Class"), y
