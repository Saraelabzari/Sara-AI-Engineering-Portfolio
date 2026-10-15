# Fraud Detection for Online Payments

A reproducible machine-learning project for identifying potentially fraudulent transactions in a highly imbalanced dataset. It includes a majority-class baseline, a class-weighted Random Forest pipeline, precision-recall evaluation, model export, and unit tests.

> **Status:** Full starter code and tests have been added. The real public dataset has not yet been used to run the training pipeline, so this repository does not claim a specific model score.

## Features

- Validates the expected `Class` target column (1 = fraud, 0 = legitimate)
- Handles missing numeric feature values with median imputation inside the pipeline
- Uses a stratified train/test split
- Compares a majority-class baseline with a class-weighted Random Forest
- Reports precision, recall, F1, average precision, ROC-AUC, and a classification report
- Saves a confusion matrix, precision-recall curve, trained model, and JSON metrics
- Includes synthetic-data unit tests that don't need the full dataset

## Dataset

This project expects the public European credit-card fraud dataset commonly distributed as `creditcard.csv`, with numeric feature columns and a `Class` target. Obtain the file from a trusted source under its applicable terms and place it here:

`data/creditcard.csv`

The dataset is highly imbalanced. Accuracy alone can look excellent even when a model detects few or no fraudulent transactions. Do not commit the dataset to GitHub.

## Setup — Windows

From this project directory:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Train

```bash
python src/train.py --data data/creditcard.csv
```

Optional: reduce the number of trees for a quicker development run:

```bash
python src/train.py --data data/creditcard.csv --estimators 30
```

Generated outputs include `reports/confusion_matrix.png`, `reports/precision_recall_curve.png`, `models/fraud_classifier.joblib`, and `models/metrics.json`. These files are generated locally and excluded from Git.

## Run tests

```bash
python -m pytest tests/
```

The tests use a small synthetic dataset and do not require the full credit-card dataset. They have been added but still need to be executed locally.

## Tech stack

Python, pandas, scikit-learn, Matplotlib, joblib, pytest.

## Responsible use

This is a decision-support prototype, not a production payment-blocking system. Choose thresholds based on the costs of false positives and missed fraud, validate against time-based splits when appropriate, monitor data drift, and provide human review for consequential decisions. Never put real card numbers, customer records, API keys, or credentials in this repository.
