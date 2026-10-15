"""Validated FAQ loading and confidence-aware TF-IDF retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK = (
    "I couldn't find a reliable answer in the current FAQ. "
    "Please contact the official support team for help."
)


def validate_entries(entries: Any) -> list[dict[str, str]]:
    """Validate and normalize FAQ records before indexing."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("Knowledge base must be a non-empty list.")
    cleaned: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"FAQ entry {index + 1} must be an object.")
        question = entry.get("question")
        answer = entry.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"FAQ entry {index + 1} needs a non-empty question.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"FAQ entry {index + 1} needs a non-empty answer.")
        cleaned.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "category": str(entry.get("category", "General")).strip() or "General",
        })
    return cleaned


def load_knowledge_base(path: str | Path) -> list[dict[str, str]]:
    """Read and validate a JSON FAQ file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_entries(json.load(handle))


class FAQRetriever:
    """Small, dependency-light FAQ retriever with a conservative fallback."""

    def __init__(self, entries: list[dict[str, str]]) -> None:
        self.entries = validate_entries(entries)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        try:
            self.question_matrix = self.vectorizer.fit_transform(
                [entry["question"] for entry in self.entries]
            )
        except ValueError as exc:
            raise ValueError(
                "FAQ questions need at least one searchable word."
            ) from exc

    def answer(self, query: str, threshold: float = 0.18) -> dict[str, Any]:
        """Return answer metadata; low-confidence queries receive a safe fallback."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if not isinstance(query, str) or not query.strip():
            return {
                "answer": FALLBACK, "source": None, "category": None,
                "score": None, "matched": False,
            }
        try:
            query_vector = self.vectorizer.transform([query.strip()])
        except ValueError:
            return {
                "answer": FALLBACK, "source": None, "category": None,
                "score": 0.0, "matched": False,
            }
        scores = cosine_similarity(query_vector, self.question_matrix).ravel()
        best_idx = int(scores.argmax())
        score = float(scores[best_idx])
        if score < threshold:
            return {
                "answer": FALLBACK, "source": None, "category": None,
                "score": score, "matched": False,
            }
        entry = self.entries[best_idx]
        return {
            "answer": entry["answer"],
            "source": entry["question"],
            "category": entry["category"],
            "score": score,
            "matched": True,
        }
