import json

import pytest

from src.retrieval import FALLBACK, FAQRetriever, load_knowledge_base, validate_entries


@pytest.fixture
def entries():
    return [
        {"question": "How do I reset my password?", "answer": "Use the reset link.", "category": "Account"},
        {"question": "Where can I track my order?", "answer": "Open the Orders page.", "category": "Orders"},
        {"question": "What is your refund policy?", "answer": "Review the published refund policy.", "category": "Refunds"},
    ]


def test_exact_match_returns_answer_and_source(entries):
    result = FAQRetriever(entries).answer("How do I reset my password?")
    assert result["answer"] == "Use the reset link."
    assert result["source"] == "How do I reset my password?"
    assert result["category"] == "Account"
    assert result["matched"] is True
    assert result["score"] >= 0.18


def test_unknown_question_uses_fallback(entries):
    result = FAQRetriever(entries).answer("How is the weather on Mars?", threshold=0.25)
    assert result["answer"] == FALLBACK
    assert result["source"] is None
    assert result["matched"] is False


def test_blank_query_uses_fallback(entries):
    result = FAQRetriever(entries).answer("   ")
    assert result["answer"] == FALLBACK
    assert result["score"] is None
    assert result["matched"] is False


def test_threshold_validation(entries):
    with pytest.raises(ValueError):
        FAQRetriever(entries).answer("password", threshold=1.2)


def test_empty_or_invalid_entries_are_rejected():
    with pytest.raises(ValueError):
        validate_entries([])
    with pytest.raises(ValueError):
        validate_entries([{"question": " ", "answer": "answer"}])
    with pytest.raises(ValueError):
        validate_entries([{"question": "question", "answer": ""}])


def test_load_knowledge_base_from_json(tmp_path, entries):
    path = tmp_path / "faq.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    loaded = load_knowledge_base(path)
    assert len(loaded) == 3
    assert loaded[1]["category"] == "Orders"


def test_empty_knowledge_base_file_is_rejected(tmp_path):
    path = tmp_path / "faq.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge_base(path)
