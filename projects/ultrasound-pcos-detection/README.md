# Ultrasound-Based PCOS Detection — Deep Learning

A research-oriented computer-vision project template for investigating whether deep-learning models can classify ultrasound images into dataset-provided categories.

> **Important status and medical limitation:** This is a starter project, not a validated diagnostic tool. No ultrasound dataset has been included, and no model performance has been measured. PCOS is a clinical condition and should not be diagnosed from an experimental image classifier alone. The project must not be used to make clinical decisions.

## Goals

- Establish a reproducible image classification workflow.
- Keep patient data private and use only appropriately licensed, de-identified images.
- Compare a pretrained image model with a simple baseline.
- Evaluate per-class precision, recall, macro F1, confusion matrix, and calibration.
- Document dataset limitations, possible bias, and the difference between research classification and clinical diagnosis.

## Dataset format

The training script expects a folder layout with one subfolder per class:

```text
data/ultrasound/
├── class_a/
│   ├── image_001.png
│   └── image_002.png
└── class_b/
    ├── image_003.png
    └── image_004.png
```

Replace the example class names with labels explicitly defined by the dataset's documentation. Do not assume ultrasound images alone establish PCOS status. Use only data with a clear legal/ethical basis and suitable consent or authorization.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data/ultrasound
```

## Method

The starter uses transfer learning with a pretrained ResNet-18, a held-out split, class-aware sampling, and basic metrics. A real research study should prefer patient-level separation to avoid images from the same patient appearing in both training and test sets.

## Responsible research checklist

- Verify provenance, label definitions, sample size, and class balance.
- Split by patient, not by image, whenever multiple images per patient exist.
- Keep an untouched test set and report confidence intervals where possible.
- Compare performance across clinically relevant subgroups when ethically and statistically appropriate.
- Do not present model scores as a diagnosis, screening recommendation, or substitute for a qualified clinician.
