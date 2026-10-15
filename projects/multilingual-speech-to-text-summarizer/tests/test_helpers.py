from app import timestamp, make_transcript, split_text


def test_timestamp():
    assert timestamp(0) == "00:00:00"
    assert timestamp(65) == "00:01:05"
    assert timestamp(3661) == "01:01:01"


def test_make_transcript():
    result = {"text": "Hello world", "segments": [{"start": 0, "end": 2.5, "text": " Hello world"}]}
    assert make_transcript(result, True) == "[00:00:00–00:00:02] Hello world"
    assert make_transcript(result, False) == "Hello world"


def test_split_text():
    chunks = split_text("This is a sentence. " * 300, max_chars=2200)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2200 for chunk in chunks)
