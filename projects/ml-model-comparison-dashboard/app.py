"""Interactive classification model comparison dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA = Path(__file__).resolve().parent / "sample_data.csv"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical),
    ])


def main() -> None:
    st.set_page_config(page_title="ML Model Comparison", page_icon="📊", layout="wide")
    st.title("Machine Learning Model Comparison Dashboard")
    st.caption("Compare classification models on a consistent held-out split.")
    uploaded = st.file_uploader("Optional: upload a CSV dataset", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(DEFAULT_DATA)
        st.info("Using a small synthetic demo dataset. Upload your own CSV for a meaningful experiment.")

    if len(df) < 10:
        st.error("Please use at least 10 rows.")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns) - 1)
    y = df[target].astype(str)
    X = df.drop(columns=[target]).copy()
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() < 2:
        st.error("The target must contain at least two classes.")
        st.stop()

    test_size = st.slider("Test set fraction", 0.2, 0.4, 0.25, 0.05)
    if st.button("Train and compare models"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as exc:
            st.error(f"Could not create a stratified split: {exc}")
            st.stop()

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        results = []
        predictions = {}
        for name, estimator in models.items():
            pipeline = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            predictions[name] = pred
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            }
            try:
                if len(pipeline.classes_) == 2:
                    row["ROC-AUC"] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
            except (ValueError, AttributeError):
                pass
            results.append(row)

        results_df = pd.DataFrame(results).set_index("Model")
        st.subheader("Evaluation metrics")
        st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)
        st.bar_chart(results_df[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]])
        selected = st.selectbox("Confusion matrix model", list(predictions.keys()))
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, predictions[selected], ax=ax, xticks_rotation="vertical")
        ax.set_title(f"{selected} — Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
