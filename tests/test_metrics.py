import unittest
from src.evaluation.metrics import compute_exact_match, compute_generation_metrics

class TestMetrics(unittest.TestCase):

    def test_exact_match(self):
        preds = ["A", "B", "C", "D"]
        refs = ["A", "B", "A", "D"]
        acc = compute_exact_match(preds, refs)
        self.assertEqual(acc, 75.0)

    def test_generation_metrics(self):
        preds = ["The quick brown fox jumps over the lazy dog."]
        refs = ["The quick brown fox jumps over a lazy dog."]
        metrics = compute_generation_metrics(preds, refs)

        self.assertIn("rouge1", metrics)
        self.assertIn("rougeL", metrics)
        self.assertIn("bleu", metrics)
        self.assertGreater(metrics["rouge1"], 50.0)

if __name__ == "__main__":
    unittest.main()
