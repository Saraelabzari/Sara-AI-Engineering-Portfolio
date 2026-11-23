"""Streamlit load-forecasting dashboard: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.forecasting import evaluate, generate_load, make_features, predictions, time_split, train_model

st.set_page_config(page_title="Energy Load Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ Short-Term Energy Load Forecasting")
st.caption("Gradient-boosted trees on calendar, lag and temperature features vs. a seasonal-naive baseline.")

with st.sidebar:
    test_days = st.slider("Test period (days)", 7, 60, 28)
    upload = st.file_uploader("…or upload hourly CSV (timestamp, load_mw[, temperature_c])", type="csv")


@st.cache_resource
def run(data: pd.DataFrame, test_days: int):
    df = make_features(data)
    train, test = time_split(df, test_days)
    model = train_model(train)
    return evaluate(model, test), predictions(model, test)


m, pred = run(pd.read_csv(upload) if upload else generate_load(), test_days)
c = st.columns(3)
c[0].metric("MAE", f"{m['mae_mw']:.1f} MW", f"{m['mae_mw'] - m['seasonal_naive_mae_mw']:.1f} vs naive", delta_color="inverse")
c[1].metric("MAPE", f"{m['mape_pct']:.2f} %")
c[2].metric("Peak error", f"{m['peak_error_mw']:.1f} MW")
days = st.slider("Days to show", 1, test_days, 7)
st.line_chart(pred.set_index("timestamp").iloc[: days * 24])
