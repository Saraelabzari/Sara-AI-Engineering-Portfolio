"""Reusable content-based product recommendation utilities."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {"product_id", "title", "category", "description"}


def validate_products(products: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a cleaned copy of the catalog."""
    if not isinstance(products, pd.DataFrame):
        raise TypeError("products must be a pandas DataFrame.")
    missing = REQUIRED_COLUMNS.difference(products.columns)
    if missing:
        raise ValueError(f"Products data is missing required columns: {sorted(missing)}")
    cleaned = products.copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    if cleaned.empty:
        raise ValueError("Product catalog must contain at least one product.")
    if (cleaned["product_id"] == "").any():
        raise ValueError("Every product must have a non-empty product_id.")
    if cleaned["product_id"].duplicated().any():
        raise ValueError("product_id values must be unique.")
    if not (cleaned["title"] != "").any():
        raise ValueError("At least one product must have a title.")
    return cleaned.reset_index(drop=True)


def build_product_text(products: pd.DataFrame) -> pd.Series:
    """Combine relevant text fields into a searchable representation."""
    cleaned = validate_products(products)
    return (
        cleaned["title"] + " " + cleaned["category"] + " " + cleaned["description"]
    ).str.strip()


class ProductRecommender:
    """Rank products by TF-IDF cosine similarity to a selected catalog item."""

    def __init__(self, products: pd.DataFrame) -> None:
        self.products = validate_products(products)
        text = (
            self.products["title"] + " " + self.products["category"] + " "
            + self.products["description"]
        ).str.strip()
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(text)

    def recommend(self, product_id: str, top_n: int = 5,
                  category: str | None = None) -> pd.DataFrame:
        """Return the top similar items, optionally constrained to a category."""
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        matches = self.products.index[
            self.products["product_id"] == str(product_id)
        ].tolist()
        if not matches:
            raise ValueError(f"Unknown product_id: {product_id}")
        selected_index = matches[0]
        scores = cosine_similarity(
            self.matrix[selected_index], self.matrix
        ).ravel()
        candidates: Iterable[int] = [
            idx for idx in range(len(self.products)) if idx != selected_index
        ]
        if category is not None:
            candidates = [
                idx for idx in candidates
                if self.products.iloc[idx]["category"].casefold() == category.casefold()
            ]
        ranked = sorted(candidates, key=lambda idx: (-scores[idx], idx))[:top_n]
        result = self.products.iloc[ranked].copy()
        result["similarity"] = [float(scores[idx]) for idx in ranked]
        return result.reset_index(drop=True)


def recommend_products(products: pd.DataFrame, product_id: str,
                       top_n: int = 5) -> pd.DataFrame:
    """Convenience function retained for scripts and unit tests."""
    return ProductRecommender(products).recommend(product_id, top_n=top_n)
