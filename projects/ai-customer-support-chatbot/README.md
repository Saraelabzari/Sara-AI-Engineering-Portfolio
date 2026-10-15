# AI Chatbot for Customer Support

A small, explainable FAQ retrieval chatbot built with Streamlit and TF-IDF. It answers from a curated JSON knowledge base, shows the matching FAQ, and uses a conservative fallback when a query does not meet a configurable similarity threshold.

> **Status:** Portfolio prototype. The included FAQs are examples, not verified company policies. It is not connected to a live customer account system and has not been production-tested.

## Features

- Streamlit chat interface with conversation history and a clear-chat control
- Reusable retrieval logic separated from the UI in `src/retrieval.py`
- TF-IDF with unigrams and bigrams, ranked by cosine similarity
- Configurable confidence threshold and optional similarity details
- FAQ source/category shown for matched answers
- Input validation and fallback for blank, unknown, or low-similarity questions
- No external LLM API key required

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Tests are included in the repository, but the project should be tested in your own environment before presenting it as verified.

## Configure the FAQ knowledge base

Edit `knowledge_base.json`. Each record must include a non-empty `question` and `answer`; `category` is optional.

```json
{
  "question": "How can I reset my password?",
  "answer": "Use the approved password reset process.",
  "category": "Account"
}
```

Only add accurate, approved information. Do not place personal data, credentials, private customer conversations, or confidential company policies in the repository.

## Architecture

1. **Load and validate:** parse the JSON and validate each FAQ record.
2. **Vectorize:** build a TF-IDF representation of FAQ questions.
3. **Rank:** compare the user query to FAQ vectors using cosine similarity.
4. **Apply threshold:** return the best FAQ answer only when its score reaches the selected threshold.
5. **Fallback:** otherwise recommend contacting official support.

This is a retrieval-based system, not a generative LLM or a full RAG pipeline: it does not compose new answers from documents. TF-IDF similarity is a heuristic, not a calibrated probability of correctness. Tune the threshold using a representative evaluation set before real use.

## Limitations and next steps

- Sample FAQs must be replaced with verified organization-specific content.
- TF-IDF can miss paraphrases and multilingual queries; semantic embeddings and language-aware retrieval could improve recall.
- Evaluate with held-out queries, top-1 accuracy, fallback rate, and human review of unsafe or incorrect matches.
- Production deployment would need authentication, privacy-safe logging, rate limits, monitoring, escalation to a human, and security review.
