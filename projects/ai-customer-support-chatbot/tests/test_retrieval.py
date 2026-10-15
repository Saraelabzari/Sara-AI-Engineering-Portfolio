import json
import unittest
from pathlib import Path

from app import FALLBACK, retrieve_answer


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"question": "How do I reset my password?", "answer": "Use the reset link."},
            {"question": "Where is my order?", "answer": "Check the Orders page."},
        ]

    def test_matching_question_returns_answer_and_source(self):
        answer, source, score = retrieve_answer("How do I reset my password?", self.entries)
        self.assertEqual(answer, "Use the reset link.")
        self.assertIsNotNone(source)
        self.assertGreaterEqual(score, 0.18)

    def test_empty_question_returns_fallback(self):
        answer, source, score = retrieve_answer("   ", self.entries)
        self.assertEqual(answer, FALLBACK)
        self.assertIsNone(source)
        self.assertIsNone(score)

    def test_unknown_question_returns_fallback(self):
        answer, source, score = retrieve_answer("What is the weather on Mars?", self.entries)
        self.assertEqual(answer, FALLBACK)
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()
