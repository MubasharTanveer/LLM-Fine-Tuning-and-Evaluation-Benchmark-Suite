import os
import tempfile
import unittest
from src.data.dataset_loader import load_instruction_dataset, format_chatml_prompt
from src.data.data_generator import SyntheticDataGenerator

class TestDataset(unittest.TestCase):

    def test_prompt_formatting(self):
        sample = {
            "instruction": "Solve X",
            "input": "X = 5",
            "output": "X is 5"
        }
        formatted = format_chatml_prompt(sample, prompt_style="alpaca")
        self.assertIn("### Instruction:\nSolve X", formatted)
        self.assertIn("### Input:\nX = 5", formatted)
        self.assertIn("### Response:\nX is 5", formatted)

    def test_synthetic_data_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "synthetic_test.json")
            dataset = SyntheticDataGenerator.generate_dataset(num_samples=5, output_path=out_file)
            self.assertEqual(len(dataset), 5)
            self.assertTrue(os.path.exists(out_file))

            ds_dict = load_instruction_dataset(out_file, validation_split_pct=20)
            self.assertIn("train", ds_dict)

if __name__ == "__main__":
    unittest.main()
