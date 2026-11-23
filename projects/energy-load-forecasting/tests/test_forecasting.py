import pandas as pd
import pytest

from src.forecasting import (LAGS, evaluate, feature_columns, generate_load, make_features,
                             predictions, time_split, train_model)


@pytest.fixture(scope="module")
def data():
    return generate_load(days=120, seed=5)


def test_generator(data):
    assert len(data) == 120 * 24
    assert data["load_mw"].gt(0).all()


def test_generator_rejects_short():
    with pytest.raises(ValueError):
        generate_load(days=5)


def test_features_have_no_leakage(data):
    df = make_features(data)
    assert not df.isna().any().any()
    # lag_24 must equal the load exactly 24 hours earlier
    row = df.iloc[10]
    earlier = data.set_index("timestamp").loc[row["timestamp"] - pd.Timedelta(hours=24), "load_mw"]
    assert row["lag_24"] == pytest.approx(earlier)
    assert all(f"lag_{l}" in df for l in LAGS)


def test_missing_column():
    with pytest.raises(ValueError, match="load_mw"):
        make_features(pd.DataFrame({"timestamp": ["2025-01-01"]}))


def test_time_split_is_chronological(data):
    train, test = time_split(make_features(data), 14)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_model_beats_seasonal_naive(data):
    train, test = time_split(make_features(data), 14)
    model = train_model(train)
    m = evaluate(model, test)
    assert m["mae_mw"] < m["seasonal_naive_mae_mw"]
    assert len(predictions(model, test)) == len(test)
