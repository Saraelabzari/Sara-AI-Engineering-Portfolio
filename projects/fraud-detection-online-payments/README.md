# Fraud Detection for Online Payments

A machine-learning starter project for classifying potentially fraudulent payment transactions. The pipeline emphasizes class imbalance, reproducible evaluation, and avoiding data leakage.

> **Status:** Starter implementation. Run the script on the public dataset to produce real metrics before describing performance in a CV or interview.

## Objectives

- Explore a labeled transaction dataset.
- Compare a majority-class baseline with a balanced Random Forest classifier.
- Evaluate fraud detection using precision, recall, F1-score, PR-AUC, and a confusion matrix.
- Save a trained model for later experimentation.
- Discuss false positives and false negatives in the context of payment review.

## Dataset

This project is configured for the public European credit-card fraud dataset commonly distributed as `creditcard.csv`. It includes anonymized transaction features, `Time`, `Amount`, and `Class` (1 = fraud, 0 = legitimate). Obtain it from a trusted, properly licensed source and place it at `data/creditcard.csv`. The dataset is highly imbalanced, so accuracy alone is misleading.

## Project structure

```text
fraud-detection-online-payments/
├── data/                  # local dataset, not committed
├── src/train.py
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
python src/train.py --data data/creditcard.csv
```

## Metrics to review

- **Precision:** How many flagged transactions were actually fraudulent?
- **Recall:** How many fraudulent transactions were detected?
- **F1-score:** Balance of precision and recall.
- **Average precision / PR-AUC:** Especially useful when fraud is rare.
- **Confusion matrix:** Shows false positives and missed fraud.

## Responsible use

This model is a prototype for analyst decision support, not an automatic final decision system. Evaluate thresholds, operational costs, data drift, and fairness before any real deployment. Never use this repository to process real payment-card data or credentials.
