#!/usr/bin/env python3
"""
CLI script to evaluate fine-tuned model against base model on standardized benchmark harness.
Usage: python scripts/evaluate.py --base-model meta-llama/Meta-Llama-3-8B-Instruct --peft-model outputs/qlora_model --benchmark data/sample_benchmark_mmlu.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import logger, setup_logger
from src.evaluation.evaluator import BenchmarkEvaluator
from src.evaluation.report_builder import EvaluationReportBuilder

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Evaluation CLI")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face model ID or local base model path.")
    parser.add_argument("--peft-model", type=str, default="./outputs/qlora_model", help="Path to fine-tuned PEFT adapter directory.")
    parser.add_argument("--benchmark", type=str, default="./data/sample_benchmark_mmlu.json", help="Path to MMLU benchmark JSON dataset.")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Directory to save evaluation reports.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum benchmark items to evaluate.")
    args = parser.parse_args()

    setup_logger()
    logger.info("Initializing LLM Benchmark Evaluator...")

    evaluator = BenchmarkEvaluator(
        base_model_name=args.base_model,
        peft_model_path=args.peft_model
    )

    results = evaluator.evaluate_benchmark(
        benchmark_file=args.benchmark,
        max_samples=args.max_samples
    )

    report_builder = EvaluationReportBuilder(output_dir=args.output_dir)
    report_builder.print_summary_table(results)
    report_builder.generate_markdown_report(results)
    report_builder.save_json(results)
    report_builder.log_to_wandb(results)

    logger.info("Evaluation workflow complete!")

if __name__ == "__main__":
    main()
