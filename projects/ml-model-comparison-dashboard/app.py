"""Streamlit dashboard for comparing classification baselines."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluation import compare_models, validate_dataset

DEFAULT_DATA = Path(__file__).resolve().parent / "sample_data.csv"


def main() -> None:
    st.set_page_config(page_title="ML Model Comparison", page_icon="📊", layout="wide")
    st.title("Machine Learning Model Comparison Dashboard")
    st.caption("Compare baseline classifiers on the same stratified hold-out split.")

    uploaded = st.file_uploader("Optional: upload a CSV dataset", type=["csv"])
    try:
        df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DEFAULT_DATA)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
    if uploaded is None:
        st.info("Using a small synthetic demo dataset. Do not interpret its scores as a real benchmark.")
    if df.empty or len(df) < 10:
        st.error("Please provide a dataset with at least 10 rows.")
        st.stop()
    if df.columns.duplicated().any():
        st.error("The CSV has duplicate column names. Rename them before continuing.")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns) - 1)
    test_size = st.slider("Test set fraction", 0.1, 0.5, 0.25, 0.05)
    st.caption("Features are preprocessed inside each pipeline to avoid fitting imputers/scalers on test data.")

    try:
        X, y = validate_dataset(df, target)
    except (TypeError, ValueError) as exc:
        st.warning(str(exc))
        st.stop()
    st.write(f"**Rows available:** {len(y):,} · **Features:** {X.shape[1]} · **Classes:** {y.nunique()}")
    st.dataframe(
        y.value_counts().rename_axis("Class").reset_index(name="Rows"),
        use_container_width=True,
    )

    if st.button("Train and compare models", type="primary"):
        try:
            with st.spinner("Training three baseline models…"):
                metrics, trained, split = compare_models(df, target, test_size=test_size)
        except (TypeError, ValueError) as exc:
            st.error(f"Training could not start: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

        st.session_state["comparison_metrics"] = metrics
        st.session_state["comparison_models"] = trained
        st.session_state["comparison_split"] = split

    if "comparison_metrics" not in st.session_state:
        st.info("Choose a target column and train the models to see results.")
        return

    metrics = st.session_state["comparison_metrics"]
    trained = st.session_state["comparison_models"]
    st.subheader("Held-out evaluation metrics")
    st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
    chart_columns = [column for column in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC"] if column in metrics.columns]
    st.bar_chart(metrics[chart_columns])
    st.caption(
        "Macro precision/recall/F1 weight each class equally. ROC-AUC is omitted when it cannot be computed. "
        "Scores depend on the data, split, and defaults; they are not production guarantees."
    )

    selected = st.selectbox("Confusion matrix model", list(trained.keys()))
    record = trained[selected]
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        record["y_test"], record["predictions"], ax=ax, xticks_rotation="vertical",
        colorbar=False,
    )
    ax.set_title(f"{selected} — Confusion Matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
