#!/usr/bin/env python3
"""
End-to-End Execution Script for LLM Fine-Tuning and Evaluation Suite.
Usage:
  python scripts/run_full_pipeline.py --dry-run
  python scripts/run_full_pipeline.py --config configs/default_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.utils.logger import logger, setup_logger
from src.data.data_generator import SyntheticDataGenerator
from src.data.dataset_loader import load_instruction_dataset
from src.evaluation.metrics import compute_exact_match, compute_generation_metrics
from src.evaluation.report_builder import EvaluationReportBuilder

def run_dry_run():
    """Runs a simulated end-to-end dry run to verify pipelines, metrics, dataset formats, and reports without needing GPU weights."""
    logger.info("[bold yellow]Running Pipeline Dry-Run Validation (CPU compatible)...[/bold yellow]")

    # 1. Dataset generation test
    gen_data = SyntheticDataGenerator.generate_dataset(num_samples=10, output_path="./data/sample_instruction_data.json")
    logger.info(f"[OK] Synthetic Data Generation: {len(gen_data)} items generated.")

    # 2. Dataset loading and prompt formatting test
    ds = load_instruction_dataset("./data/sample_instruction_data.json", validation_split_pct=20)
    logger.info(f"[OK] Instruction Tokenization & Formatting: Train={len(ds['train'])}, Validation={len(ds['validation'])}")

    # 3. Metric computation validation test
    mock_preds = ["B", "To simulate a larger effective batch size across multiple micro-batches without exceeding VRAM."]
    mock_refs = ["B", "To simulate a larger effective batch size."]
    em = compute_exact_match(mock_preds, mock_refs)
    metrics = compute_generation_metrics(mock_preds, mock_refs)

    logger.info(f"[OK] Metric Engine Validation: ExactMatch={em}%, ROUGE-L={metrics['rougeL']}, BLEU-4={metrics['bleu']}")

    # 4. Report Builder validation
    mock_results = {
        "num_samples": 4,
        "base_model": {
            "mmlu_accuracy_pct": 50.0,
            "perplexity": 14.82,
            "rouge1": 42.1,
            "rouge2": 21.5,
            "rougeL": 38.4,
            "bleu": 18.2,
            "sample_outputs": [{"question": "What is QLoRA?", "gold_answer": "B", "generated": "A standard tuning framework."}]
        },
        "fine_tuned_model": {
            "mmlu_accuracy_pct": 100.0,
            "perplexity": 4.12,
            "rouge1": 88.5,
            "rouge2": 76.2,
            "rougeL": 85.0,
            "bleu": 64.1,
            "sample_outputs": [{"question": "What is QLoRA?", "gold_answer": "B", "generated": "B) Quantized 4-bit NF4 adaptation."}]
        }
    }

    report_builder = EvaluationReportBuilder(output_dir="./eval_results")
    report_builder.print_summary_table(mock_results)
    report_builder.generate_markdown_report(mock_results)
    report_builder.save_json(mock_results)

    logger.info("[bold green]Dry-run verification completed successfully! Architecture is fully operational.[/bold green]")

def main():
    parser = argparse.ArgumentParser(description="Run Full Fine-Tuning and Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml", help="Path to config file.")
    parser.add_argument("--dry-run", action="store_true", help="Execute dry-run validation without loading heavy weights.")
    args = parser.parse_args()

    setup_logger()

    if args.dry_run:
        run_dry_run()
    else:
        logger.info(f"Starting Full Production Pipeline with config: {args.config}")
        # Execute train CLI logic
        from src.training.qlora_trainer import QLoRATrainer
        config = AppConfig.from_yaml(args.config)
        trainer = QLoRATrainer(config)
        trainer.train()

        # Execute evaluation CLI logic
        from src.evaluation.evaluator import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator(
            base_model_name=config.model.base_model_name_or_path,
            peft_model_path=config.project.output_dir
        )
        results = evaluator.evaluate_benchmark(config.data.eval_file)
        report_builder = EvaluationReportBuilder(output_dir="./eval_results")
        report_builder.print_summary_table(results)
        report_builder.generate_markdown_report(results)
        report_builder.save_json(results)

if __name__ == "__main__":
    main()
