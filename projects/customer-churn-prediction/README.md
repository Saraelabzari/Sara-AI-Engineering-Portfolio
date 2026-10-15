# Customer Churn Prediction

A reproducible machine-learning pipeline for predicting whether a subscription customer is likely to churn. It includes preprocessing, a Logistic Regression baseline, held-out evaluation, saved model artifacts, and unit tests.

> **Status:** Code and tests have been added to the portfolio repository. The pipeline has not yet been run against the full IBM dataset, so no performance result is claimed.

## Features

- Loads and validates the IBM Telco Customer Churn CSV
- Converts `TotalCharges` to numeric values and handles missing values in the preprocessing pipeline
- Excludes customer identifiers from model features
- Uses stratified train/test splitting and preprocessing fitted only on training data
- Reports accuracy, precision, recall, F1-score, ROC-AUC, and a classification report
- Saves the trained model, metrics JSON, and confusion-matrix image

## Project structure

```text
customer-churn-prediction/
├── data/                  # Local dataset; excluded from Git
├── reports/               # Generated model and metrics; excluded from Git
├── src/
│   ├── churn_model.py     # Reusable preprocessing, training utilities, evaluation
│   └── train.py           # Command-line training entry point
├── tests/
│   └── test_churn_model.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Dataset

Use the public IBM Telco Customer Churn dataset. Place the CSV at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The expected target column is `Churn`, with values `Yes` and `No`. Do not commit customer data or data you are not licensed to share.

## Setup — Windows

Run these commands from the project folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Train and evaluate

```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script writes `reports/churn_model.joblib`, `reports/metrics.json`, and `reports/confusion_matrix.png`. These are generated locally and intentionally excluded from Git.

## Run tests

```bash
python -m pytest tests/
```

Tests use a small synthetic dataset and do not require downloading the IBM dataset. They have been added but still need to be executed in a local environment.

## Tech stack

Python, pandas, scikit-learn, Matplotlib, joblib, pytest.

## Responsible use

Churn probabilities are estimates, not certainty about individual customers. Review precision and recall together, check for data leakage and group-level performance differences, and use predictions to improve customer support—not to unfairly restrict service. This project is a portfolio prototype, not a production decision system.
