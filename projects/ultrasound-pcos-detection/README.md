# PCOS Prediction from Clinical Data & Ultrasound Image Classification

This portfolio folder documents two complementary research directions for PCOS-related classification:

1. **Clinical/tabular data baseline (original FYP notebook):** data inspection, missing-value handling, stratified train/test split, a scikit-learn preprocessing pipeline, and comparison of Logistic Regression, Random Forest, and XGBoost using ROC-AUC.
2. **Ultrasound image classification prototype:** a separate PyTorch/ResNet-18 transfer-learning starter for labeled image folders.

> **Medical limitation:** These are academic research prototypes, not validated diagnostic tools. They must not be used to diagnose PCOS, guide treatment, or replace a qualified clinician. No clinical performance claim is made here.

## Clinical/tabular ML workflow

The original notebook supplied for this project includes:

- Loading the tabular dataset from an Excel workbook
- Reviewing columns, target-label counts, and missing values
- Removing unnamed columns and normalizing column names
- Stratified train/test split for the `PCOS (Y/N)` target
- Median imputation and feature standardization in a scikit-learn pipeline
- Class-weighted Logistic Regression baseline
- Random Forest and XGBoost model comparison
- Classification report, ROC-AUC, and five-fold ROC-AUC cross-validation

The notebook's code compares models; its scores should be rerun and checked before reporting them publicly. Do not interpret ROC-AUC alone as clinical validity.

## Ultrasound image-classification prototype

The image pipeline in this folder uses PyTorch and ResNet-18 transfer learning. It expects a folder-per-class image dataset and can report accuracy, macro F1, a classification report, and a confusion matrix.

Example layout:

```text
data/ultrasound/
├── dataset_label_a/
│   ├── image_001.png
│   └── image_002.png
└── dataset_label_b/
    ├── image_003.png
    └── image_004.png
```

Use only appropriately licensed, de-identified data with documented label definitions. The dataset is not included in this repository.

## Install and run the image prototype

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python src/train.py --data data/ultrasound --epochs 5 --batch-size 16
```

Follow the training script's arguments and output paths for the current image-classification implementation.

## Tests

```bash
python -m pytest tests/
```

Tests are included but have not been executed in this environment.

## Research limitations

- The original tabular notebook uses a held-out split and cross-validation, but its metrics must be reproduced before making claims.
- The image prototype currently uses an image-level split. If multiple images come from one patient, split by patient to reduce data leakage.
- A small or unrepresentative dataset can produce misleading results.
- Further work should include a separate held-out test set, per-class error analysis, calibration, subgroup evaluation, and external validation.
- Never present this experimental work as a diagnosis or substitute for medical care.
