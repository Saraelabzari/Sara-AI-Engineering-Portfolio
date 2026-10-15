"""Train a baseline emotion classifier from a local audio manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split

from emotion_model import build_classifier, extract_features, load_manifest, resolve_audio_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train speech emotion recognition baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    features, labels = [], []
    for row in manifest.itertuples(index=False):
        audio_path = resolve_audio_path(row.path, args.manifest.parent)
        try:
            features.append(extract_features(audio_path))
            labels.append(row.label)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Skipping {audio_path}: {exc}")
    if len(features) < 4 or len(set(labels)) < 2:
        raise ValueError("Not enough readable audio files across at least two labels.")
    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        raise ValueError("Each emotion label needs at least two readable examples for stratified splitting.")

    X = np.vstack(features)
    y = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_classifier(n_estimators=args.trees)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(classification_report(y_test, prediction, zero_division=0))
    print(f"Macro F1: {f1_score(y_test, prediction, average='macro'):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, xticks_rotation="vertical")
    plt.title("Speech Emotion Recognition — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    dump(model, args.model_dir / "emotion_classifier.joblib")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved model to {args.model_dir / 'emotion_classifier.joblib'}")


if __name__ == "__main__":
    main()
