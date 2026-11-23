import pandas as pd
import pytest

from src.maintenance import (Config, add_features, add_labels, evaluate, feature_columns,
                             generate_fleet, latest_status, split_by_unit, train_models)


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(n_units=12, sensors=3, seed=1)


def test_generator_shape(fleet):
    assert {"unit", "cycle", "sensor_1"} <= set(fleet.columns)
    assert fleet["unit"].nunique() == 12


def test_labels_rul_and_cap(fleet):
    df = add_labels(fleet, Config(rul_cap=50, alert_horizon=10))
    last = df.groupby("unit").tail(1)
    assert (last["rul"] == 0).all() and (last["fail_soon"] == 1).all()
    assert df["rul"].max() == 50


def test_labels_require_columns():
    with pytest.raises(ValueError, match="unit"):
        add_labels(pd.DataFrame({"cycle": [1]}))


def test_features_no_nans(fleet):
    df = add_features(add_labels(fleet))
    assert not df[feature_columns(df)].isna().any().any()
    assert "sensor_1_slope" in df.columns


def test_split_keeps_units_separate(fleet):
    train, test = split_by_unit(add_labels(fleet))
    assert set(train["unit"]).isdisjoint(test["unit"])


def test_models_beat_baseline(fleet):
    df = add_features(add_labels(fleet))
    train, test = split_by_unit(df)
    reg, clf = train_models(train, n_estimators=30)
    m = evaluate(reg, clf, test, train)
    assert m["rul_mae"] < m["baseline_rul_mae"]
    assert m["alert_recall"] > 0.5
    status = latest_status(reg, clf, test)
    assert len(status) == test["unit"].nunique()
