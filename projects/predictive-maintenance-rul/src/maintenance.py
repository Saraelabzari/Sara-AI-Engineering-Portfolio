"""Predictive maintenance: Remaining Useful Life (RUL) regression and
failure-risk classification for rotating industrial equipment.

The module is dataset-agnostic. It expects run-to-failure records with one row
per (unit, cycle) and numeric sensor columns. A synthetic generator is included
so the full pipeline can be run and tested without a downloaded dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)

ID_COL, TIME_COL = "unit", "cycle"


@dataclass(frozen=True)
class Config:
    window: int = 5            # rolling-window length (cycles)
    rul_cap: int = 125         # piecewise-linear RUL cap (common for turbofan-style data)
    alert_horizon: int = 30    # "will fail within N cycles" label


def generate_fleet(n_units: int = 40, sensors: int = 6, seed: int = 7) -> pd.DataFrame:
    """Create a synthetic run-to-failure fleet with noisy, degrading sensors."""
    if n_units < 2:
        raise ValueError("n_units must be at least 2")
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 300))
        drift = rng.uniform(0.6, 1.4, sensors)
        for cycle in range(1, life + 1):
            wear = (cycle / life) ** 2.2
            values = 1.0 + drift * wear * 3 + rng.normal(0, 0.08, sensors)
            rows.append([unit, cycle, *values])
    cols = [ID_COL, TIME_COL] + [f"sensor_{i + 1}" for i in range(sensors)]
    return pd.DataFrame(rows, columns=cols)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in (ID_COL, TIME_COL, "rul", "fail_soon")]
    if not cols:
        raise ValueError("No sensor columns found")
    return cols


def add_labels(df: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Add capped RUL and a binary 'fail_soon' label (run-to-failure data only)."""
    for col in (ID_COL, TIME_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    last = out.groupby(ID_COL)[TIME_COL].transform("max")
    out["rul"] = (last - out[TIME_COL]).clip(upper=cfg.rul_cap)
    out["fail_soon"] = (last - out[TIME_COL] <= cfg.alert_horizon).astype(int)
    return out


def add_features(df: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Rolling mean / std and cycle-over-cycle slope per unit and sensor."""
    if cfg.window < 2:
        raise ValueError("window must be at least 2")
    out = df.sort_values([ID_COL, TIME_COL]).copy()
    grouped = out.groupby(ID_COL)
    for col in sensor_columns(df):
        roll = grouped[col].rolling(cfg.window, min_periods=1)
        out[f"{col}_mean"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{col}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        out[f"{col}_slope"] = grouped[col].diff().fillna(0.0)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (ID_COL, "rul", "fail_soon")]


def split_by_unit(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Split whole units (not rows) so no machine appears in both sets."""
    units = np.array(sorted(df[ID_COL].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = max(1, int(round(len(units) * test_size)))
    test_units = set(units[:n_test])
    mask = df[ID_COL].isin(test_units)
    return df[~mask], df[mask]


def train_models(train: pd.DataFrame, n_estimators: int = 200, seed: int = 42):
    X = train[feature_columns(train)]
    reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3,
                                random_state=seed, n_jobs=-1).fit(X, train["rul"])
    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3,
                                 class_weight="balanced", random_state=seed,
                                 n_jobs=-1).fit(X, train["fail_soon"])
    return reg, clf


def evaluate(reg, clf, test: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Metrics for both models plus a mean-RUL baseline for context."""
    X = test[feature_columns(test)]
    pred_rul = reg.predict(X)
    baseline = np.full(len(test), train["rul"].mean())
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = test["fail_soon"]
    return {
        "rul_mae": float(mean_absolute_error(test["rul"], pred_rul)),
        "rul_rmse": float(np.sqrt(mean_squared_error(test["rul"], pred_rul))),
        "baseline_rul_mae": float(mean_absolute_error(test["rul"], baseline)),
        "alert_precision": float(precision_score(y, pred, zero_division=0)),
        "alert_recall": float(recall_score(y, pred, zero_division=0)),
        "alert_f1": float(f1_score(y, pred, zero_division=0)),
        "alert_avg_precision": float(average_precision_score(y, proba)),
    }


def latest_status(reg, clf, df_features: pd.DataFrame) -> pd.DataFrame:
    """Predicted RUL and failure risk at the latest cycle of each unit."""
    last = df_features.sort_values(TIME_COL).groupby(ID_COL).tail(1)
    X = last[feature_columns(last)]
    out = last[[ID_COL, TIME_COL]].copy()
    out["predicted_rul"] = reg.predict(X).round(1)
    out["failure_risk"] = clf.predict_proba(X)[:, 1].round(3)
    out["status"] = pd.cut(out["failure_risk"], [-0.01, 0.3, 0.7, 1.0],
                           labels=["Healthy", "Watch", "Maintain now"])
    return out.sort_values("failure_risk", ascending=False).reset_index(drop=True)
