# Predictive Maintenance — Remaining Useful Life (RUL)

Industrial AI project that predicts **how many operating cycles a machine has left** and **whether it is likely to fail soon**, from multi-sensor run-to-failure data. It is built for rotating equipment such as motors, fans, pumps and turbines, where planned maintenance is far cheaper than an unplanned stop.

## What it does

- **RUL regression:** a Random Forest predicts remaining cycles, using the piecewise-linear capped RUL common in turbofan benchmarks.
- **Failure-risk alert:** a class-weighted Random Forest predicts "fails within *N* cycles" and turns it into Healthy / Watch / Maintain now status.
- **Leak-free evaluation:** the split is by machine, not by row, so no unit appears in both training and test data. Results are compared with a mean-RUL baseline.
- **Feature engineering:** per-unit rolling mean, rolling standard deviation and cycle-to-cycle slope for every sensor.
- **Fleet dashboard (Streamlit):** KPIs, a ranked maintenance list and sensor trends for each unit.
- **Works on its own:** a synthetic fleet generator lets the whole pipeline run without a download. You can also bring your own CSV (`unit, cycle, sensor_1 … sensor_n`), for example NASA C-MAPSS.

## Results on the synthetic fleet

Reproducible with `python src/train.py` (seeded, 40 units, 25 % of units held out):

| Metric | Value |
|---|---|
| RUL MAE | **4.2 cycles** (mean-RUL baseline: 36.8) |
| RUL RMSE | 6.5 cycles |
| Alert precision / recall | 0.86 / 0.92 |
| Alert average precision | 0.97 |

> These numbers come from synthetic data and show the pipeline works. They are not a claim about real equipment. Real sensors have noise, operating regimes and sensor failures that need extra validation.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

python src/train.py              # train + evaluate on the synthetic fleet
streamlit run app.py             # interactive fleet-health dashboard
python -m pytest tests/          # 6 tests
```

Use your own data: `python src/train.py --data data/fleet.csv`

## Project structure

```
src/maintenance.py   data generation, labelling, features, unit-wise split, models, evaluation
src/train.py         CLI: train, print metrics, save models/maintenance.joblib + metrics.json
app.py               Streamlit dashboard
tests/               pytest suite (labels, features, leakage-free split, beats baseline)
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · joblib · pytest

## Next steps

- Benchmark on NASA C-MAPSS FD001–FD004 and compare against an LSTM / 1D-CNN.
- Add conformal prediction intervals so maintenance planners get an uncertainty band.
- Add a cost-based alert threshold (cost of an unplanned stop vs. early maintenance).
