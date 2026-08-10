#!/usr/bin/env python3
"""
CLI script to run QLoRA fine-tuning.
Usage: python scripts/train.py --config configs/default_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.utils.logger import logger, setup_logger
from src.utils.seed import set_seed
from src.training.qlora_trainer import QLoRATrainer
from src.training.unsloth_trainer import UnslothTrainerWrapper

def main():
    parser = argparse.ArgumentParser(description="LLM QLoRA Fine-Tuning CLI")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()

    setup_logger()
    logger.info(f"Loading configuration from: {args.config}")
    config = AppConfig.from_yaml(args.config)

    set_seed(config.project.seed)

    if config.model.use_unsloth:
        logger.info("Initializing Unsloth fine-tuning wrapper...")
        trainer = UnslothTrainerWrapper(config)
    else:
        logger.info("Initializing standard QLoRA SFT trainer...")
        trainer = QLoRATrainer(config)

    trainer.train()
    logger.info("Training script execution finished.")

if __name__ == "__main__":
    main()
