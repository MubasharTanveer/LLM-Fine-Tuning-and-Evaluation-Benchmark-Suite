import os
import json
from typing import Dict, Any, List, Optional
from ..utils.logger import logger
from .metrics import compute_perplexity, compute_exact_match, compute_generation_metrics

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None

class BenchmarkEvaluator:
    """
    Evaluator engine comparing base model vs fine-tuned adapter model performance
    on MMLU QA subsets and text generation benchmarks.
    """

    def __init__(self, base_model_name: str, peft_model_path: Optional[str] = None, device: str = "cuda"):
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.device = device if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, is_peft: bool = False):
        """Loads base model or PEFT adapter model."""
        logger.info(f"Loading {'PEFT Fine-Tuned' if is_peft else 'Base'} Model: {self.base_model_name}")

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_peft and self.peft_model_path and os.path.exists(self.peft_model_path):
            logger.info(f"Attaching PEFT adapters from: {self.peft_model_path}")
            model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        else:
            model = base_model

        model.eval()
        return model, tokenizer

    def evaluate_benchmark(self, benchmark_file: str, max_samples: int = 100) -> Dict[str, Any]:
        """
        Runs comprehensive benchmark evaluation for both Base Model and Fine-Tuned Model.
        """
        if not os.path.exists(benchmark_file):
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

        with open(benchmark_file, "r", encoding="utf-8") as f:
            items = json.load(f)[:max_samples]

        logger.info(f"Running benchmark evaluation on {len(items)} items...")

        # Evaluate Base Model
        base_model, base_tok = self.load_model_and_tokenizer(is_peft=False)
        base_results = self._evaluate_model(base_model, base_tok, items)

        # Free GPU VRAM memory
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate Fine-Tuned PEFT Model if path exists
        peft_results = {}
        if self.peft_model_path and os.path.exists(self.peft_model_path):
            peft_model, peft_tok = self.load_model_and_tokenizer(is_peft=True)
            peft_results = self._evaluate_model(peft_model, peft_tok, items)

            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.warning("PEFT adapter model path not found. Skipping fine-tuned model comparative run.")

        return {
            "num_samples": len(items),
            "base_model": base_results,
            "fine_tuned_model": peft_results
        }

    def _evaluate_model(self, model, tokenizer, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Internal worker evaluating a single model instance against benchmark items."""
        predictions = []
        references = []
        raw_outputs = []
        eval_texts = []

        for item in items:
            question = item.get("question", "")
            choices = "\n".join(item.get("choices", []))
            prompt = f"Question: {question}\n{choices}\nAnswer with option letter (A, B, C, D) and explanation:\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            predictions.append(gen_text)
            ref_answer = item.get("answer", "")
            references.append(ref_answer)
            raw_outputs.append({
                "id": item.get("id"),
                "question": question,
                "gold_answer": ref_answer,
                "generated": gen_text
            })
            eval_texts.append(f"{question} {gen_text}")

        # Compute Metrics
        em_acc = compute_exact_match(predictions, references)
        gen_metrics = compute_generation_metrics(predictions, [item.get("reference_output", "") for item in items])
        ppl = compute_perplexity(model, tokenizer, eval_texts, device=self.device)

        return {
            "mmlu_accuracy_pct": em_acc,
            "perplexity": ppl,
            "rouge1": gen_metrics["rouge1"],
            "rouge2": gen_metrics["rouge2"],
            "rougeL": gen_metrics["rougeL"],
            "bleu": gen_metrics["bleu"],
            "sample_outputs": raw_outputs[:5]
        }
