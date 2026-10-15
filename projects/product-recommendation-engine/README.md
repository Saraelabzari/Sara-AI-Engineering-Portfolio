# Product Recommendation Engine

A content-based recommendation starter that suggests products similar to a selected item using text features from product titles and descriptions.

> **Status:** Starter implementation. The included catalog is fictional demonstration data. Recommendations have not been evaluated with real user interaction data.

## Features

- Small editable sample product catalog
- TF-IDF representation of product text
- Cosine-similarity recommendations
- Streamlit interface for selecting a product and viewing similar items
- No user tracking or personal data required

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Load product titles, categories, and descriptions from `products.csv`.
2. Combine product text and vectorize it with TF-IDF.
3. Calculate cosine similarity between products.
4. Return the top matching items, excluding the selected product itself.

## Limitations and future improvements

This is a content-based baseline, so it cannot learn from user behavior and may over-recommend items with similar descriptions. Future work could add collaborative filtering, offline ranking metrics, popularity baselines, and a feedback loop with consented, anonymized interaction data.

## Privacy and responsible use

The demo does not collect user profiles. If adding behavioral recommendations, clearly disclose data use, minimize retained information, and provide meaningful user controls.
