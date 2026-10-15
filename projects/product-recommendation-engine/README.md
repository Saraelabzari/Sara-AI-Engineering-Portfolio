# Product Recommendation Engine

A content-based recommendation demo that ranks similar products from titles, categories, and descriptions using TF-IDF features and cosine similarity.

> **Status:** Portfolio prototype. The bundled catalog is fictional demonstration data. Similarity scores are ranking signals, not calibrated probabilities or evidence of user preference.

## Features

- Streamlit interface for selecting a product
- Reusable recommendation logic in `src/recommender.py`
- TF-IDF with unigrams and bigrams, plus cosine similarity
- Optional category filter and configurable recommendation count
- Catalog validation for missing fields, empty IDs, and duplicate IDs
- Unit tests for ranking, filtering, and invalid inputs
- No user tracking or personal data required

## Run locally

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

Tests are committed but have not been executed in this environment; run them locally before describing the project as verified.

## Customize the catalog

Replace `products.csv` with a catalog containing these columns:

| Column | Purpose |
|---|---|
| `product_id` | Unique product identifier |
| `title` | Product name |
| `category` | Product category |
| `description` | Short product description |

Keep the catalog free of personal data. Ensure IDs are unique and products have meaningful text fields.

## How it works

1. Validate and normalize the catalog.
2. Combine each product's title, category, and description.
3. Convert product text to TF-IDF vectors.
4. Rank the other products by cosine similarity to the selected item.
5. Optionally restrict the results to a category.

## Limitations and next steps

This baseline cannot learn from clicks, purchases, inventory, price, or individual preferences. Short or generic descriptions may lead to weak recommendations. Evaluate with curated relevance judgments and ranking metrics such as Precision@K, Recall@K, MAP, or NDCG before deployment. Future work could compare semantic embeddings and collaborative filtering against this baseline.

If behavioral data is added, obtain appropriate consent, minimize retention, document the purpose, and provide user controls.
