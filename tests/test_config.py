import os
import unittest
from src.config import AppConfig

class TestConfig(unittest.TestCase):

    def test_default_config_loading(self):
        config_path = "./configs/default_config.yaml"
        self.assertTrue(os.path.exists(config_path), "default_config.yaml does not exist.")

        config = AppConfig.from_yaml(config_path)
        self.assertEqual(config.project.name, "llm-finetune-eval-suite")
        self.assertEqual(config.lora.r, 16)
        self.assertTrue(config.quantization.load_in_4bit)
        self.assertIn("q_proj", config.lora.target_modules)

    def test_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            AppConfig.from_yaml("non_existent_config.yaml")

if __name__ == "__main__":
    unittest.main()
