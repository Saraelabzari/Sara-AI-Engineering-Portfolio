"""Short-term electricity load forecasting with calendar, lag and weather features.

Hourly data with a timestamp, a load column and (optionally) temperature.
A synthetic generator with daily/weekly seasonality, holidays and a
temperature-driven cooling load is included for reproducible runs and tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

TS, Y, TEMP = "timestamp", "load_mw", "temperature_c"
LAGS = (24, 48, 168)          # same hour yesterday, 2 days ago, last week


def generate_load(days: int = 365, seed: int = 3, start: str = "2025-01-01") -> pd.DataFrame:
    """Synthetic hourly load (MW) for a mid-size industrial/urban grid."""
    if days < 30:
        raise ValueError("days must be at least 30")
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    doy, hour, dow = idx.dayofyear.values, idx.hour.values, idx.dayofweek.values
    temp = 22 + 9 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24) \
        + rng.normal(0, 1.5, len(idx))
    daily = 1 + 0.25 * np.sin(2 * np.pi * (hour - 7) / 24) + 0.12 * np.sin(4 * np.pi * (hour - 3) / 24)
    weekly = np.where(dow == 4, 0.82, np.where(dow == 5, 0.9, 1.0))   # Friday/Saturday weekend
    cooling = np.clip(temp - 24, 0, None) * 9
    load = 420 * daily * weekly + cooling + rng.normal(0, 10, len(idx))
    return pd.DataFrame({TS: idx, Y: load.round(2), TEMP: temp.round(1)})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + lag + rolling features. Rows without full lag history are dropped."""
    for col in (TS, Y):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    out[TS] = pd.to_datetime(out[TS])
    out = out.sort_values(TS).reset_index(drop=True)
    t = out[TS].dt
    out["hour"], out["dayofweek"], out["month"] = t.hour, t.dayofweek, t.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    for lag in LAGS:
        out[f"lag_{lag}"] = out[Y].shift(lag)
    out["roll_mean_24"] = out[Y].shift(24).rolling(24).mean()
    return out.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TS, Y)]


def time_split(df: pd.DataFrame, test_days: int = 28):
    """Chronological split: the last `test_days` are the test period (no shuffling)."""
    cutoff = df[TS].max() - pd.Timedelta(days=test_days)
    return df[df[TS] <= cutoff], df[df[TS] > cutoff]


def train_model(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed)
    return model.fit(train[feature_columns(train)], train[Y])


def evaluate(model, test: pd.DataFrame) -> dict:
    pred = model.predict(test[feature_columns(test)])
    naive = test["lag_168"]                       # seasonal-naive: same hour last week
    return {
        "mae_mw": float(mean_absolute_error(test[Y], pred)),
        "mape_pct": float(100 * mean_absolute_percentage_error(test[Y], pred)),
        "seasonal_naive_mae_mw": float(mean_absolute_error(test[Y], naive)),
        "seasonal_naive_mape_pct": float(100 * mean_absolute_percentage_error(test[Y], naive)),
        "peak_error_mw": float(abs(test[Y].max() - pred.max())),
    }


def predictions(model, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({TS: test[TS].values, "actual": test[Y].values,
                         "forecast": model.predict(test[feature_columns(test)]).round(2),
                         "seasonal_naive": test["lag_168"].values})
