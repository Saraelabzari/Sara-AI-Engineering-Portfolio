# Customer Churn Prediction

A machine-learning project to identify customers who may leave a subscription-based service. The goal is to support retention teams with risk scores and interpretable model evaluation.

## Project status

**Starter implementation / in progress.** This repository contains the project plan and training script structure. Results should only be reported after running the pipeline on the dataset.

## Objectives

- Explore customer demographics, account details, service usage, and churn labels.
- Build a reproducible preprocessing and classification pipeline.
- Compare a simple baseline with a tree-based model.
- Evaluate using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Review feature importance to help explain predictions.

## Dataset

Use the public **IBM Telco Customer Churn** dataset. Download the CSV from a trusted source and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Do not commit private customer information or data that you are not licensed to share.

Expected target column: `Churn`.

## Project structure

```text
customer-churn-prediction/
├── data/                 # Add dataset locally; not committed
├── src/
│   └── train.py           # Preprocessing, training and evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script reports evaluation metrics on a held-out test set. It does not claim a specific accuracy in advance; performance depends on the dataset and split.

## Tech stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib

## Responsible use

Churn scores are decision-support signals, not certainty about an individual customer. Check for data leakage, monitor performance across customer groups, and use predictions to improve service rather than to unfairly restrict access.

## Next improvements

- Add exploratory data analysis and visualizations.
- Compare Logistic Regression, Random Forest, and Gradient Boosting.
- Tune the decision threshold around retention-team capacity.
- Add cross-validation, calibration, and a model card.
- Build a small Streamlit dashboard after validating the model.
