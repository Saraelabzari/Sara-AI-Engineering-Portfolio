import unittest

import pandas as pd

from app import recommend_products


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.products = pd.DataFrame([
            {"product_id": "1", "title": "Wireless Headphones", "category": "Audio", "description": "Bluetooth headphones"},
            {"product_id": "2", "title": "Bluetooth Speaker", "category": "Audio", "description": "Wireless speaker sound"},
            {"product_id": "3", "title": "Garden Shovel", "category": "Garden", "description": "Tool for digging soil"},
        ])

    def test_does_not_recommend_selected_item(self):
        results = recommend_products(self.products, "1", top_n=2)
        self.assertNotIn("1", results["product_id"].tolist())
        self.assertLessEqual(len(results), 2)

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            recommend_products(self.products, "missing")


if __name__ == "__main__":
    unittest.main()
