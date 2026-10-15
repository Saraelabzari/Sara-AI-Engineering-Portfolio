import pytest

from src.audio_utils import make_transcript, split_text, timestamp


def test_timestamp_formats_seconds():
    assert timestamp(0) == "00:00:00"
    assert timestamp(65) == "00:01:05"
    assert timestamp(3661) == "01:01:01"
    assert timestamp(-5) == "00:00:00"


def test_make_transcript_with_segments():
    result = {"text": "Hello world", "segments": [{"start": 0, "end": 2.5, "text": " Hello world"}]}
    assert make_transcript(result) == "[00:00:00–00:00:02] Hello world"
    assert make_transcript(result, include_timestamps=False) == "Hello world"


def test_make_transcript_falls_back_to_text():
    assert make_transcript({"text": "  Plain transcript  ", "segments": []}) == "Plain transcript"


def test_split_text_respects_limit():
    chunks = split_text("word " * 1000, max_chars=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_text_empty_and_invalid_limit():
    assert split_text("  ") == []
    with pytest.raises(ValueError):
        split_text("hello", max_chars=0)
