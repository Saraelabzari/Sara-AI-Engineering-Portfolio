"""Summarization utilities; model loading is kept separate for easier testing."""
from __future__ import annotations

from src.audio_utils import split_text

SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"


def summarize_text(text: str, summarizer, max_tokens: int = 120, chunk_chars: int = 2200) -> str:
    """Summarize long transcripts in chunks and optionally condense the result."""
    if not text.strip():
        return ""
    if max_tokens < 10:
        raise ValueError("max_tokens must be at least 10")

    summaries = []
    for chunk in split_text(text, max_chars=chunk_chars):
        output = summarizer(
            chunk,
            max_new_tokens=max_tokens,
            min_length=min(20, max(5, max_tokens // 4)),
            do_sample=False,
            truncation=True,
        )
        if output and output[0].get("generated_text", "").strip():
            summaries.append(output[0]["generated_text"].strip())
    combined = " ".join(summaries).strip()
    if len(summaries) > 1 and len(combined) > 500:
        output = summarizer(
            combined[:5000], max_new_tokens=max_tokens, do_sample=False, truncation=True
        )
        if output and output[0].get("generated_text", "").strip():
            return output[0]["generated_text"].strip()
    return combined
