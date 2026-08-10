import os
import json
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table

try:
    import wandb
except ImportError:
    wandb = None

from ..utils.logger import logger

class EvaluationReportBuilder:
    """Formats, logs, and exports comparative evaluation benchmark results."""

    def __init__(self, output_dir: str = "./eval_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.console = Console()

    def print_summary_table(self, results: Dict[str, Any]):
        """Displays side-by-side comparison table in console using Rich."""
        table = Table(title="[bold yellow]Benchmark Evaluation Comparison: Base Model vs Fine-Tuned (PEFT)[/bold yellow]")

        table.add_column("Evaluation Metric", style="cyan", no_wrap=True)
        table.add_column("Base Model", style="magenta")
        table.add_column("Fine-Tuned (QLoRA)", style="green")
        table.add_column("Delta / Improvement", style="bold yellow")

        base = results.get("base_model", {})
        peft = results.get("fine_tuned_model", {})

        metrics_map = [
            ("MMLU Accuracy (%)", "mmlu_accuracy_pct", True),
            ("Perplexity (PPL)", "perplexity", False), # lower is better
            ("ROUGE-1 Score", "rouge1", True),
            ("ROUGE-2 Score", "rouge2", True),
            ("ROUGE-L Score", "rougeL", True),
            ("BLEU-4 Score", "bleu", True)
        ]

        for label, key, higher_is_better in metrics_map:
            b_val = base.get(key, 0.0)
            p_val = peft.get(key, 0.0)

            if p_val != 0.0:
                diff = p_val - b_val
                sign = "+" if diff > 0 else ""
                color = "green" if (diff > 0 if higher_is_better else diff < 0) else "red"
                delta_str = f"[{color}]{sign}{diff:.2f}[/{color}]"
            else:
                delta_str = "N/A"

            table.add_row(label, f"{b_val:.2f}", f"{p_val:.2f}" if p_val != 0.0 else "N/A", delta_str)

        self.console.print(table)

    def generate_markdown_report(self, results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Generates a markdown benchmark summary report."""
        path = filepath or os.path.join(self.output_dir, "benchmark_report.md")
        base = results.get("base_model", {})
        peft = results.get("fine_tuned_model", {})

        md = [
            "# LLM Fine-Tuning & Evaluation Benchmark Report\n",
            f"**Total Benchmark Samples Evaluated**: `{results.get('num_samples', 0)}`\n",
            "## Metric Summary\n",
            "| Metric | Base Model | Fine-Tuned Model (QLoRA) | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **MMLU Subset Accuracy (%)** | `{base.get('mmlu_accuracy_pct', 0):.2f}%` | `{peft.get('mmlu_accuracy_pct', 0):.2f}%` | `+{peft.get('mmlu_accuracy_pct', 0) - base.get('mmlu_accuracy_pct', 0):.2f}%` |",
            f"| **Perplexity (PPL)** | `{base.get('perplexity', 0):.2f}` | `{peft.get('perplexity', 0):.2f}` | `{peft.get('perplexity', 0) - base.get('perplexity', 0):.2f}` |",
            f"| **ROUGE-1 Score** | `{base.get('rouge1', 0):.2f}` | `{peft.get('rouge1', 0):.2f}` | `+{peft.get('rouge1', 0) - base.get('rouge1', 0):.2f}` |",
            f"| **ROUGE-2 Score** | `{base.get('rouge2', 0):.2f}` | `{peft.get('rouge2', 0):.2f}` | `+{peft.get('rouge2', 0) - base.get('rouge2', 0):.2f}` |",
            f"| **ROUGE-L Score** | `{base.get('rougeL', 0):.2f}` | `{peft.get('rougeL', 0):.2f}` | `+{peft.get('rougeL', 0) - base.get('rougeL', 0):.2f}` |",
            f"| **BLEU-4 Score** | `{base.get('bleu', 0):.2f}` | `{peft.get('bleu', 0):.2f}` | `+{peft.get('bleu', 0) - base.get('bleu', 0):.2f}` |\n",
            "## Qualitative Sample Comparison\n"
        ]

        base_samples = base.get("sample_outputs", [])
        peft_samples = peft.get("sample_outputs", [])

        for idx, b_samp in enumerate(base_samples):
            p_samp = peft_samples[idx] if idx < len(peft_samples) else {}
            md.append(f"### Sample {idx+1}: {b_samp.get('question', '')[:80]}...")
            md.append(f"**Gold Target**: `{b_samp.get('gold_answer')}`")
            md.append(f"- **Base Model Generation**: {b_samp.get('generated')}")
            if p_samp:
                md.append(f"- **Fine-Tuned Generation**: {p_samp.get('generated')}")
            md.append("\n---\n")

        content = "\n".join(md)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved benchmark markdown report to: [bold cyan]{path}[/bold cyan]")
        return path

    def save_json(self, results: Dict[str, Any], filepath: Optional[str] = None):
        """Saves evaluation results dictionary as formatted JSON file."""
        path = filepath or os.path.join(self.output_dir, "benchmark_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark JSON data to: {path}")

    def log_to_wandb(self, results: Dict[str, Any], wandb_run: Optional[Any] = None):
        """Logs benchmark comparison summary table and metrics to active W&B run."""
        if wandb is None:
            logger.info("Weights & Biases (`wandb`) package not installed. Skipping W&B report upload.")
            return

        if wandb_run is None and wandb.run is not None:
            wandb_run = wandb.run

        if wandb_run is not None:
            base = results.get("base_model", {})
            peft = results.get("fine_tuned_model", {})

            wandb_run.log({
                "eval/base_mmlu_acc": base.get("mmlu_accuracy_pct", 0),
                "eval/peft_mmlu_acc": peft.get("mmlu_accuracy_pct", 0),
                "eval/base_perplexity": base.get("perplexity", 0),
                "eval/peft_perplexity": peft.get("perplexity", 0),
                "eval/peft_rougeL": peft.get("rougeL", 0),
                "eval/peft_bleu": peft.get("bleu", 0)
            })

            # Create W&B Table
            wb_table = wandb.Table(columns=["Model", "MMLU Acc (%)", "Perplexity", "ROUGE-L", "BLEU-4"])
            wb_table.add_data("Base Model", base.get("mmlu_accuracy_pct", 0), base.get("perplexity", 0), base.get("rougeL", 0), base.get("bleu", 0))
            if peft:
                wb_table.add_data("Fine-Tuned (QLoRA)", peft.get("mmlu_accuracy_pct", 0), peft.get("perplexity", 0), peft.get("rougeL", 0), peft.get("bleu", 0))

            wandb_run.log({"benchmark_summary_table": wb_table})
            logger.info("Logged benchmark evaluation artifacts to Weights & Biases!")
