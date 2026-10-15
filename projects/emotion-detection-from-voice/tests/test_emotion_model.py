import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.emotion_model import build_classifier, extract_features, load_manifest, resolve_audio_path


def test_extract_features_from_generated_tone(tmp_path):
    sample_rate = 22050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 440 * time)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sample_rate)
    features = extract_features(path)
    assert features.shape == (40,)
    assert np.isfinite(features).all()


def test_extract_features_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_features(tmp_path / "missing.wav")


def test_load_manifest_requires_columns(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"filename": ["a.wav"], "emotion": ["happy"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="path"):
        load_manifest(path)


def test_resolve_audio_path(tmp_path):
    assert resolve_audio_path("sample.wav", tmp_path) == tmp_path / "sample.wav"


def test_classifier_rejects_invalid_tree_count():
    with pytest.raises(ValueError):
        build_classifier(n_estimators=0)
