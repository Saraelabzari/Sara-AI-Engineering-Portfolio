# Ultrasound Image Classification Research Prototype

A deep-learning image-classification starter built with PyTorch and a pretrained ResNet-18. The pipeline expects dataset-provided class labels and reports experimental validation metrics.

> **Medical limitation:** This is not a validated diagnostic tool. It does not establish PCOS status, provide medical advice, or support clinical decisions. No dataset has been included and no real-dataset performance has been measured.

## Features

- Validates a folder-per-class image dataset
- Resizes images and converts grayscale images to three channels
- Transfer learning using ResNet-18
- Training loop with AdamW and cross-entropy loss
- Classification report, accuracy, macro F1, and confusion matrix
- Saves a model checkpoint with class names
- Unit tests for dataset validation

## Dataset layout

Use only appropriately licensed and de-identified data with clear label definitions:

```text
data/ultrasound/
├── dataset_label_a/
│   ├── image_001.png
│   └── image_002.png
└── dataset_label_b/
    ├── image_003.png
    └── image_004.png
```

The class folder names must reflect the dataset's documented labels. Do not assume an ultrasound image alone confirms PCOS or another diagnosis. The dataset is not included in this repository.

## Install — Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The pretrained ResNet-18 weights may be downloaded by torchvision on first use. This requires internet access unless the weights are already cached.

## Train

```bash
python src/train.py --data data/ultrasound --epochs 5 --batch-size 16
```

Outputs are saved locally to `reports/confusion_matrix.png` and `models/resnet18_classifier.pt`; these folders are excluded from Git.

## Tests

```bash
python -m pytest tests/
```

The tests generate temporary sample images and do not require a clinical dataset. They have been added but not yet executed in this environment.

## Important research limitations

- The current split is image-level, not patient-level. If multiple images come from one patient, a patient-level split is essential to reduce leakage.
- A small or unrepresentative dataset can produce misleading results.
- Report uncertainty and assess class balance, subgroup performance, calibration, and external validation before drawing research conclusions.
- Never present this experimental model as a diagnosis or substitute for a qualified clinician.
