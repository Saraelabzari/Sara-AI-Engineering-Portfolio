"""Train a baseline speech-emotion classifier from a CSV manifest.

Manifest columns:
  path  - audio file path (relative to manifest folder or absolute)
  label - emotion category

Example:
  python src/train.py --manifest data/labels.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


def extract_features(audio_path: Path, sample_rate: int = 22050) -> np.ndarray:
    """Extract mean and standard deviation of MFCCs from one audio file."""
    signal, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest = pd.read_csv(args.manifest)
    if not {"path", "label"}.issubset(manifest.columns):
        raise ValueError("Manifest must include 'path' and 'label' columns.")
    manifest = manifest.dropna(subset=["path", "label"])
    if manifest["label"].nunique() < 2:
        raise ValueError("At least two distinct emotion labels are required.")

    base = args.manifest.parent
    features, labels, kept = [], [], []
    for row in manifest.itertuples(index=False):
        audio_path = Path(str(row.path))
        if not audio_path.is_absolute():
            audio_path = base / audio_path
        if not audio_path.exists():
            print(f"Skipping missing audio file: {audio_path}")
            continue
        try:
            features.append(extract_features(audio_path))
            labels.append(str(row.label))
            kept.append(str(audio_path))
        except Exception as exc:
            print(f"Skipping unreadable audio file {audio_path}: {exc}")

    if len(features) < 4 or len(set(labels)) < 2:
        raise ValueError("Not enough valid audio samples across at least two labels.")

    X = np.vstack(features)
    y = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Macro F1: {f1_score(y_test, pred, average='macro'):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation="vertical")
    plt.title("Speech Emotion Recognition — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    # Persist the pipeline and the label classes learned by the classifier.
    from joblib import dump
    dump(model, args.model_dir / "emotion_classifier.joblib")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved model to {args.model_dir / 'emotion_classifier.joblib'}")


if __name__ == "__main__":
    main()
