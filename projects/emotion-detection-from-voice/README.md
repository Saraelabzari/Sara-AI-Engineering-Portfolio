# Emotion Detection from Voice

A speech-emotion-recognition research prototype that extracts MFCC statistics from labeled audio and trains a Random Forest baseline to predict the labels in a dataset.

> **Status:** The implementation and tests have been added. It has not yet been trained on a full labeled dataset, so no performance result is claimed.

## Features

- CSV manifest validation for audio paths and emotion labels
- Relative and absolute audio path support
- MFCC mean and standard-deviation feature extraction
- Imputation and a class-weighted Random Forest classifier
- Stratified train/test split and macro F1/classification report
- Confusion matrix and saved model
- Unit tests using a generated tone and temporary manifest files

## Dataset format

Create a local CSV at `data/labels.csv` with these columns:

```csv
path,label
audio/sample_001.wav,happy
audio/sample_002.wav,sad
audio/sample_003.wav,neutral
audio/sample_004.wav,angry
```

Relative audio paths are resolved from the manifest directory. Use a dataset whose licence allows your intended use; RAVDESS is one commonly used academic dataset. Do not upload recordings unless redistribution is permitted.

## Setup — Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Train

```bash
python src/train.py --manifest data/labels.csv
```

Optional settings:

```bash
python src/train.py --manifest data/labels.csv --trees 100
```

Generated files are written to `reports/confusion_matrix.png` and `models/emotion_classifier.joblib`; generated outputs are excluded from Git.

## Tests

```bash
python -m pytest tests/
```

The tests use a generated audio tone and do not need a downloaded emotion dataset. They have been added but still need to be run locally.

## Tech stack

Python, librosa, NumPy, pandas, scikit-learn, Matplotlib, SoundFile, pytest.

## Limitations and responsible use

Vocal emotion is subjective, culturally dependent, and not a reliable direct measurement of a person's internal state. Predictions may vary across languages, accents, speakers, microphones, and contexts. Do not use the output as a diagnosis or as the sole basis for employment, education, policing, or other consequential decisions.
