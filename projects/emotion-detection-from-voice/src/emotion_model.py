"""Reusable audio feature extraction and speech-emotion model helpers."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def extract_features(audio_path: str | Path, sample_rate: int = 22050,
                     n_mfcc: int = 20) -> np.ndarray:
    """Extract MFCC mean/std statistics from a mono audio clip."""
    audio_path = Path(audio_path)
    if sample_rate <= 0 or n_mfcc <= 0:
        raise ValueError("sample_rate and n_mfcc must be positive.")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    signal, actual_rate = librosa.load(audio_path, sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    mfcc = librosa.feature.mfcc(y=signal, sr=actual_rate, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def resolve_audio_path(value: str, manifest_dir: str | Path) -> Path:
    """Resolve absolute paths unchanged and relative paths from the manifest folder."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(manifest_dir) / path


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV manifest containing path and label columns."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("Manifest must include 'path' and 'label' columns.")
    frame = frame.dropna(subset=["path", "label"]).copy()
    frame["path"] = frame["path"].astype(str)
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[frame["label"] != ""]
    if frame["label"].nunique() < 2:
        raise ValueError("At least two distinct emotion labels are required.")
    return frame.reset_index(drop=True)


def build_classifier(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    """Create a baseline classifier pipeline."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive.")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        )),
    ])
