"""Train and evaluate the predictive-maintenance models.

python src/train.py                       # synthetic fleet
python src/train.py --data data/fleet.csv # your own run-to-failure CSV (unit, cycle, sensors...)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from maintenance import (Config, add_features, add_labels, evaluate, generate_fleet,
                         split_by_unit, train_models)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="CSV with unit, cycle and sensor columns")
    ap.add_argument("--estimators", type=int, default=200)
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    raw = pd.read_csv(args.data) if args.data else generate_fleet()
    cfg = Config()
    df = add_features(add_labels(raw, cfg), cfg)
    train, test = split_by_unit(df)
    reg, clf = train_models(train, n_estimators=args.estimators)
    metrics = evaluate(reg, clf, test, train)
    print(pd.Series(metrics).round(3).to_string())

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"regressor": reg, "classifier": clf, "config": cfg}, args.model_dir / "maintenance.joblib")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to {args.model_dir}/")


if __name__ == "__main__":
    main()
