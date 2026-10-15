import pandas as pd
import pytest

from src.recommender import ProductRecommender, recommend_products, validate_products


@pytest.fixture
def products():
    return pd.DataFrame([
        {"product_id": "1", "title": "Wireless Headphones", "category": "Audio", "description": "Bluetooth headphones with noise cancellation"},
        {"product_id": "2", "title": "Bluetooth Speaker", "category": "Audio", "description": "Wireless speaker with rich sound"},
        {"product_id": "3", "title": "Garden Shovel", "category": "Garden", "description": "Hand tool for digging soil"},
    ])


def test_does_not_recommend_selected_item(products):
    results = recommend_products(products, "1", top_n=2)
    assert "1" not in results["product_id"].tolist()
    assert len(results) <= 2


def test_related_product_ranks_ahead_of_unrelated_product(products):
    results = ProductRecommender(products).recommend("1", top_n=2)
    assert results.iloc[0]["product_id"] == "2"
    assert results.iloc[0]["similarity"] >= results.iloc[1]["similarity"]


def test_unknown_id_raises(products):
    with pytest.raises(ValueError, match="Unknown product_id"):
        recommend_products(products, "missing")


def test_invalid_top_n_raises(products):
    model = ProductRecommender(products)
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.recommend("1", top_n=value)


def test_duplicate_ids_are_rejected(products):
    duplicated = pd.concat([products, products.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_products(duplicated)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_products(pd.DataFrame([{"product_id": "1"}]))


def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError, match="at least one product"):
        validate_products(pd.DataFrame(columns=[
            "product_id", "title", "category", "description"
        ]))


def test_category_filter_returns_only_requested_category(products):
    results = ProductRecommender(products).recommend("1", category="audio")
    assert results["category"].str.casefold().eq("audio").all()
    assert "1" not in results["product_id"].tolist()
