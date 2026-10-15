"""Streamlit interface for a grounded FAQ customer-support chatbot."""
from pathlib import Path

import streamlit as st

from src.retrieval import FAQRetriever, load_knowledge_base

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"


@st.cache_data
def cached_knowledge_base(path: str) -> list[dict[str, str]]:
    return load_knowledge_base(path)


@st.cache_resource
def cached_retriever(entries: list[dict[str, str]]) -> FAQRetriever:
    return FAQRetriever(entries)


def main() -> None:
    st.set_page_config(page_title="AI Customer Support Chatbot", page_icon="💬")
    st.title("AI Customer Support Chatbot")
    st.caption(
        "A local FAQ retrieval demo. Answers come from the configured knowledge base; "
        "it does not generate or verify company policy."
    )

    with st.sidebar:
        st.subheader("Settings")
        threshold = st.slider(
            "Answer confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
            help="Raise this value to make the bot more cautious about answering.",
        )
        show_score = st.checkbox("Show retrieval details", value=False)
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    try:
        entries = cached_knowledge_base(str(KB_PATH))
        retriever = cached_retriever(entries)
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the FAQ knowledge base: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(f"FAQ source: {message['source']}")
            if show_score and message.get("score") is not None:
                st.caption(f"Similarity score: {message['score']:.3f}")

    prompt = st.chat_input("Ask about passwords, orders, refunds, or account help…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = retriever.answer(prompt, threshold=threshold)
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result["source"]:
                st.caption(
                    f"FAQ source: {result['source']} · {result['category']}"
                )
            if show_score and result["score"] is not None:
                st.caption(f"Similarity score: {result['score']:.3f}")
            if not result["matched"]:
                st.info("This response is a fallback because no FAQ passed the threshold.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "source": result["source"],
            "score": result["score"],
        })


if __name__ == "__main__":
    main()
