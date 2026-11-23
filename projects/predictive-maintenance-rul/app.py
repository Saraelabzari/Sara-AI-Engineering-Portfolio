"""Streamlit fleet-health dashboard: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.maintenance import (Config, add_features, add_labels, evaluate, generate_fleet,
                             latest_status, split_by_unit, train_models)

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")
st.title("🛠️ Predictive Maintenance — Remaining Useful Life")
st.caption("Random Forest models predict remaining cycles and the risk of failure within the alert horizon.")

with st.sidebar:
    units = st.slider("Synthetic fleet size", 10, 80, 40)
    horizon = st.slider("Alert horizon (cycles)", 10, 60, 30)
    upload = st.file_uploader("…or upload run-to-failure CSV (unit, cycle, sensors)", type="csv")

cfg = Config(alert_horizon=horizon)


@st.cache_resource
def fit(data: pd.DataFrame, cfg: Config):
    df = add_features(add_labels(data, cfg), cfg)
    train, test = split_by_unit(df)
    reg, clf = train_models(train, n_estimators=150)
    return df, test, reg, clf, evaluate(reg, clf, test, train)


raw = pd.read_csv(upload) if upload else generate_fleet(units)
df, test, reg, clf, m = fit(raw, cfg)

c = st.columns(4)
c[0].metric("RUL MAE (cycles)", f"{m['rul_mae']:.1f}", f"{m['rul_mae'] - m['baseline_rul_mae']:.1f} vs baseline", delta_color="inverse")
c[1].metric("RUL RMSE", f"{m['rul_rmse']:.1f}")
c[2].metric("Alert recall", f"{m['alert_recall']:.0%}")
c[3].metric("Alert precision", f"{m['alert_precision']:.0%}")

st.subheader("Held-out units — current status")
# Truncate test units at a random point to simulate machines still in service.
snap = test.groupby("unit", group_keys=False).apply(lambda g: g.head(max(5, int(len(g) * 0.8))))
st.dataframe(latest_status(reg, clf, snap), use_container_width=True)

unit = st.selectbox("Inspect unit", sorted(test["unit"].unique()))
g = test[test["unit"] == unit]
st.line_chart(g.set_index("cycle")[[c for c in g.columns if c.startswith("sensor_") and c.count("_") == 1]])
