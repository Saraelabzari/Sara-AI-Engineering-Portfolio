"""Train the load-forecasting model.

python src/train.py                        # synthetic year of hourly data
python src/train.py --data data/load.csv   # CSV with timestamp, load_mw[, temperature_c]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecasting import evaluate, generate_load, make_features, predictions, time_split, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--test-days", type=int, default=28)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    raw = pd.read_csv(args.data) if args.data else generate_load()
    df = make_features(raw)
    train, test = time_split(df, args.test_days)
    model = train_model(train)
    metrics = evaluate(model, test)
    print(pd.Series(metrics).round(2).to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    p = predictions(model, test).set_index("timestamp").iloc[:24 * 7]
    ax = p.plot(figsize=(11, 4), title="Hourly load — last week of test period")
    ax.set_ylabel("MW"); plt.tight_layout(); plt.savefig(args.out / "forecast_week.png", dpi=120)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, args.out / "load_model.joblib")
    print(f"\nSaved plot, metrics and model to {args.out}/")


if __name__ == "__main__":
    main()
