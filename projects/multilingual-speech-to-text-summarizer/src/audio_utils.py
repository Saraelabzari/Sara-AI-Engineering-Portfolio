"""Small, dependency-light utilities for transcript formatting and chunking."""
from __future__ import annotations


def timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, clamping negative values to zero."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_transcript(result: dict, include_timestamps: bool = True) -> str:
    """Build readable transcript text from a Whisper-style result dictionary."""
    segments = result.get("segments") or []
    if not include_timestamps or not segments:
        return (result.get("text") or "").strip()
    lines = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if text:
            start = timestamp(segment.get("start", 0))
            end = timestamp(segment.get("end", 0))
            lines.append(f"[{start}–{end}] {text}")
    return "\n".join(lines) or (result.get("text") or "").strip()


def split_text(text: str, max_chars: int = 2200) -> list[str]:
    """Split long text near sentence boundaries while respecting max_chars."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    # Preserve common sentence punctuation, including Arabic question marks.
    parts = []
    current = []
    for word in normalized.split(" "):
        if len(word) > max_chars:
            if current:
                parts.append(" ".join(current))
                current = []
            parts.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
        if current and current[-1][-1:] in {".", "!", "?", "؟"} and len(" ".join(current)) >= max_chars * 0.6:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts
