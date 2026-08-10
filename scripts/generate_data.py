#!/usr/bin/env python3
"""
CLI script to generate synthetic instruction datasets for fine-tuning.
Usage: python scripts/generate_data.py --num-samples 50 --output data/synthetic_data.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_generator import SyntheticDataGenerator
from src.utils.logger import logger, setup_logger

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LLM instruction dataset.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of instruction samples to generate.")
    parser.add_argument("--output", type=str, default="./data/sample_instruction_data.json", help="Path to save output JSON file.")
    args = parser.parse_args()

    setup_logger()
    logger.info("Initializing Data Generator...")
    SyntheticDataGenerator.generate_dataset(num_samples=args.num_samples, output_path=args.output)
    logger.info("Data generation complete!")

if __name__ == "__main__":
    main()
