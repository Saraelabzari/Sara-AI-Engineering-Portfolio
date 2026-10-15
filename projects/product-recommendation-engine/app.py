"""Streamlit UI for a content-based product recommendation demo."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import ProductRecommender, validate_products

DATA_PATH = Path(__file__).resolve().parent / "products.csv"


@st.cache_data
def load_catalog(path: str) -> pd.DataFrame:
    return validate_products(pd.read_csv(path))


@st.cache_resource
def build_recommender(products: pd.DataFrame) -> ProductRecommender:
    return ProductRecommender(products)


def main() -> None:
    st.set_page_config(page_title="Product Recommendation Engine", page_icon="🛍️")
    st.title("Product Recommendation Engine")
    st.caption("Content-based recommendations using product text features.")

    try:
        products = load_catalog(str(DATA_PATH))
        recommender = build_recommender(products)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the product catalog: {exc}")
        st.stop()

    options = products["product_id"].tolist()
    selected_id = st.selectbox(
        "Choose a product",
        options,
        format_func=lambda pid: products.loc[
            products["product_id"] == pid, "title"
        ].iloc[0],
    )
    top_n = st.slider(
        "Number of recommendations", min_value=1,
        max_value=max(1, min(8, len(products) - 1)), value=min(5, max(1, len(products) - 1))
    )
    categories = ["Any category"] + sorted(
        category for category in products["category"].unique() if category
    )
    category_choice = st.selectbox("Optional category filter", categories)

    if st.button("Recommend similar products", type="primary"):
        selected = products.loc[products["product_id"] == selected_id].iloc[0]
        st.subheader(f"Selected: {selected['title']}")
        category = None if category_choice == "Any category" else category_choice
        results = recommender.recommend(selected_id, top_n=top_n, category=category)
        if results.empty:
            st.info("No other products match the selected category.")
        else:
            for row in results.itertuples(index=False):
                with st.container(border=True):
                    st.markdown(f"**{row.title}**")
                    st.caption(f"{row.category} · Product ID: {row.product_id}")
                    st.write(row.description)
                    st.progress(min(1.0, max(0.0, float(row.similarity))))
                    st.caption(
                        f"Text similarity: {row.similarity:.3f} "
                        "(a ranking score, not a purchase probability)"
                    )


if __name__ == "__main__":
    main()
