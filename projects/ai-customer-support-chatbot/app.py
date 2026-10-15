"""FAQ retrieval chatbot demo using Streamlit and TF-IDF."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"
FALLBACK = (
    "I couldn't find a reliable answer in the current FAQ. "
    "Please contact the official support team for help."
)


@st.cache_data
def load_knowledge_base(path: str = str(KB_PATH)) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Knowledge base must be a non-empty list.")
    for entry in entries:
        if not entry.get("question") or not entry.get("answer"):
            raise ValueError("Each FAQ entry must have a question and an answer.")
    return entries


def retrieve_answer(query: str, entries: list[dict[str, str]], threshold: float = 0.18):
    """Return best FAQ answer and similarity score, or fallback if below threshold."""
    query = query.strip()
    if not query:
        return FALLBACK, None, None
    questions = [entry["question"] for entry in entries]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(questions + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score < threshold:
        return FALLBACK, None, best_score
    return entries[best_idx]["answer"], entries[best_idx]["question"], best_score


def main() -> None:
    st.set_page_config(page_title="AI Customer Support Chatbot", page_icon="💬")
    st.title("AI Customer Support Chatbot")
    st.caption("FAQ retrieval demo — replace the sample FAQs with verified company information.")
    entries = load_knowledge_base()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(f"FAQ source: {message['source']}")

    prompt = st.chat_input("Ask a question about orders, refunds, or account help…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer, source, score = retrieve_answer(prompt, entries)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if source:
                st.caption(f"FAQ source: {source} · similarity {score:.2f}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "source": source}
        )


if __name__ == "__main__":
    main()
