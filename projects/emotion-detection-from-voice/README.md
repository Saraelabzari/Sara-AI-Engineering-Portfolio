# Emotion Detection from Voice

A speech-emotion-recognition starter project that extracts acoustic features from audio and trains a classifier to predict emotion labels from a labeled speech dataset.

> **Status:** Starter implementation. It has not yet been trained or evaluated on a dataset, so no accuracy or performance claim is made.

## Objectives

- Load labeled speech recordings.
- Extract MFCC-based acoustic features.
- Train a baseline classifier.
- Evaluate with macro F1-score, a classification report, and a confusion matrix.
- Consider speaker-independent splits to reduce speaker leakage.

## Dataset

This project expects a folder of audio files and a CSV manifest with columns:

- `path`: path to an audio file, relative to the manifest directory or absolute
- `label`: emotion class, such as happy, sad, angry, neutral, or fearful

Use a dataset whose terms of use permit your intended use. For example, the RAVDESS emotional speech dataset is commonly used for academic experimentation. Download it from its official source and create a manifest locally at `data/labels.csv`. Do not commit audio recordings unless the dataset licence explicitly permits redistribution.

## Setup and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --manifest data/labels.csv
```

## Method

1. Read each audio file.
2. Resample to a consistent sampling rate.
3. Extract MFCC statistics from the audio.
4. Encode the emotion labels.
5. Split the dataset into training and test subsets.
6. Train a Random Forest baseline and report macro F1-score and per-class metrics.

## Limitations and responsible use

Vocal emotion is subjective and culturally dependent. Audio models can be inaccurate across languages, accents, recording conditions, and demographic groups. Do not use predicted emotion as a diagnosis or as the sole basis for employment, education, policing, or other consequential decisions. The model predicts dataset labels, not a person's true internal state.
