# Image Classification Pipeline (ML + Deep Learning)

A reusable image-classification training starter using PyTorch and ResNet-18 transfer learning. It reads labeled images from class-named folders, applies train/evaluation transforms, trains a classification head (or optionally fine-tunes the backbone), and saves the best validation checkpoint plus training history.

> **Status:** General-purpose portfolio starter. No dataset or trained weights are included, and no accuracy claim is made. Results depend on dataset quality, class balance, leakage, and split design.

## Features

- Folder-based image dataset via `torchvision.datasets.ImageFolder`
- ResNet-18 transfer learning with optional pretrained weights
- Train-time augmentation and deterministic evaluation transforms
- Reproducible random train/validation split
- CPU or CUDA training
- Loss and accuracy history in JSON
- Best-validation checkpoint with class-name metadata
- Unit tests for transforms, model construction, training loop, and checkpoint output

## Dataset layout

Create a local dataset folder like this (do not commit private or licensed datasets):

```text
my_dataset/
├── cats/
│   ├── cat_001.jpg
│   └── cat_002.jpg
└── dogs/
    ├── dog_001.jpg
    └── dog_002.jpg
```

Each immediate subfolder is treated as a class label. Use enough examples per class to support a meaningful split. For medical or repeated-subject datasets, split by patient/person or source group rather than randomly by image to reduce leakage.

## Install and train

Python 3.10+ is recommended. Install the PyTorch and Torchvision builds appropriate for your CPU/GPU from the official PyTorch installation selector, then install this project's requirements.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py --data-dir /path/to/my_dataset --epochs 5 --batch-size 16
```

Pretrained weights are enabled by default and may be downloaded on first use. For a fully random initialization that avoids the weights download, pass `--no-pretrained`.

Useful options:

```bash
python train.py --data-dir ./my_dataset --epochs 10 --image-size 224 --learning-rate 0.001
python train.py --data-dir ./my_dataset --unfreeze-backbone --output-dir outputs_finetuned
```

Artifacts are written to `outputs/` by default: `best_model.pth` and `training_history.json`. These files are ignored by Git to avoid committing large generated artifacts.

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Tests are included but have not been executed in this environment. Model tests may download no weights because they explicitly use `pretrained=False`.

## Method and limitations

- Uses ImageNet normalization for ResNet-18 inputs.
- Validation accuracy is used to select the checkpoint; it is not a final unbiased test score.
- No separate test set, calibration, or class-imbalance strategy is implemented in this starter.
- If images are duplicated, come from the same patient, or are near-identical, random image-level splits can overestimate generalization.
- Review per-class precision/recall, confusion matrices, and error examples before considering deployment.
- This is a training starter, not a production inference service. Add a separate held-out test set and domain-specific validation for serious use.
