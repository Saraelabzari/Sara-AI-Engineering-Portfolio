"""Content-based product recommendation demo using TF-IDF."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent / "products.csv"


def recommend_products(products: pd.DataFrame, product_id: str, top_n: int = 5) -> pd.DataFrame:
    required = {"product_id", "title", "category", "description"}
    if not required.issubset(products.columns):
        raise ValueError(f"Products CSV must contain columns: {sorted(required)}")
    if products["product_id"].duplicated().any():
        raise ValueError("product_id values must be unique.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    matches = products.index[products["product_id"].astype(str) == str(product_id)].tolist()
    if not matches:
        raise ValueError(f"Unknown product_id: {product_id}")

    text = (
        products["title"].fillna("") + " " +
        products["category"].fillna("") + " " +
        products["description"].fillna("")
    )
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(text)
    selected_index = matches[0]
    scores = cosine_similarity(matrix[selected_index], matrix).ravel()
    ranked = scores.argsort()[::-1]
    ranked = [idx for idx in ranked if idx != selected_index][:top_n]
    result = products.iloc[ranked].copy()
    result["similarity"] = scores[ranked]
    return result.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Product Recommendation Engine", page_icon="🛍️")
    st.title("Product Recommendation Engine")
    st.caption("Content-based recommendations using product text.")
    products = pd.read_csv(DATA_PATH)
    selected_id = st.selectbox(
        "Choose a product",
        products["product_id"].astype(str).tolist(),
        format_func=lambda pid: products.loc[
            products["product_id"].astype(str) == pid, "title"
        ].iloc[0],
    )
    top_n = st.slider("Number of recommendations", min_value=1, max_value=8, value=5)
    if st.button("Recommend similar products"):
        selected = products.loc[products["product_id"].astype(str) == selected_id].iloc[0]
        st.subheader(f"Selected: {selected['title']}")
        results = recommend_products(products, selected_id, top_n)
        if results.empty:
            st.info("No similar products were found.")
        else:
            for row in results.itertuples(index=False):
                st.markdown(f"**{row.title}** · {row.category}")
                st.write(row.description)
                st.caption(f"Text similarity: {row.similarity:.2f}")


if __name__ == "__main__":
    main()
