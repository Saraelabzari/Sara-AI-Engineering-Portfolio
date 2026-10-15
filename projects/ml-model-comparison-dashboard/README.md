# Machine Learning Model Comparison Dashboard

An interactive Streamlit application for comparing three baseline classification algorithms on the same held-out split. It supports numeric and categorical features and displays macro-averaged metrics and a confusion matrix.

> **Status:** Portfolio prototype. The bundled dataset is synthetic and intentionally small. Its scores are illustrative only, not meaningful evidence of model quality on real business data.

## Features

- Upload a CSV or use the included synthetic sample
- Select the target column
- Validate class labels and feature columns
- Preprocess numeric and categorical data inside each scikit-learn pipeline
- Compare Logistic Regression, Random Forest, and Gradient Boosting
- Display accuracy, macro precision, macro recall, macro F1, and ROC-AUC when supported
- Inspect class distribution and a model-specific confusion matrix
- Reusable evaluation functions and unit tests

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The tests are committed but have not been executed in this environment. Run them locally before presenting the project as verified.

## Expected CSV format

- One row per observation
- One target column with at least two classes
- At least one feature column
- Numeric and categorical features are supported
- Missing target rows are excluded; missing feature values are imputed within the training pipeline

Avoid target leakage: do not include columns that encode the answer or information unavailable at prediction time.

## Methodology and limitations

- Uses a stratified train/test split with a fixed random seed.
- Preprocessing is fitted only on training data through scikit-learn pipelines.
- Macro metrics give each class equal weight, which can be useful when classes are imbalanced.
- ROC-AUC is shown only when it can be calculated from model probability estimates.
- The dashboard uses default hyperparameters and one split; results can vary with data and random seed.

For stronger evidence, add cross-validation, confidence intervals, threshold tuning, calibration, and a final untouched test set. Compare models on a task-appropriate metric rather than choosing the highest accuracy by default.
