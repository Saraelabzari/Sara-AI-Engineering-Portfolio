# Machine Learning Model Comparison Dashboard

An interactive Streamlit dashboard that trains several classification models on a selected dataset and compares their performance using consistent train/test splits and evaluation metrics.

> **Status:** Starter project. It uses a bundled, synthetic demonstration dataset by default. Results are educational and should not be interpreted as production model benchmarks.

## Features

- Dataset preview and target-column selection
- Numeric and categorical preprocessing with scikit-learn pipelines
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Accuracy, precision, recall, F1-score, and ROC-AUC where applicable
- Confusion matrix and metric comparison chart
- Reproducible train/test split

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Methodology notes

- The target column should be categorical or binary for this starter dashboard.
- The dashboard uses a stratified split when possible.
- Preprocessing is fitted inside each model pipeline to reduce data leakage.
- Model metrics depend on dataset choice and class balance; review precision and recall in addition to accuracy.
- For a rigorous comparison, use cross-validation, confidence intervals, tuned thresholds, and a held-out final test set.

## Next improvements

Add regression support, cross-validation, hyperparameter tuning, experiment export, and a model card with intended use and limitations.
