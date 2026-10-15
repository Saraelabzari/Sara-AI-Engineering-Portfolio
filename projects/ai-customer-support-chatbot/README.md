# AI Chatbot for Customer Support

A retrieval-augmented customer-support chatbot starter. It searches a small knowledge base for relevant passages and uses them as context for a response, with a fallback when it cannot find supporting information.

> **Status:** Starter app. Add your own verified FAQs before use. This is not connected to a live customer system and has not been production-tested.

## Features

- Simple chat interface built with Streamlit
- Local FAQ/knowledge-base retrieval using TF-IDF
- Shows the source FAQ used for the response
- Fallback when no relevant answer is found
- No API key required for the baseline retrieval version

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Customize the knowledge base

Edit `knowledge_base.json`. Each entry must contain `question` and `answer`. Use only accurate, approved customer-support information; do not add private customer conversations or secrets.

## How it works

1. The app loads FAQ question-answer pairs.
2. TF-IDF vectorization represents FAQ questions as weighted text features.
3. Cosine similarity finds the closest FAQ to a user question.
4. The app displays the answer only if similarity exceeds a configurable threshold; otherwise, it asks the user to contact support.

This is retrieval, not a generative language model. It is intentionally designed to avoid inventing unsupported answers.

## Improvements to explore

- Add evaluation questions and measure top-1 retrieval accuracy.
- Use semantic embeddings for paraphrase handling.
- Add a carefully grounded LLM response layer with citations.
- Add multilingual support, authentication, logging controls, and human handoff before deployment.
