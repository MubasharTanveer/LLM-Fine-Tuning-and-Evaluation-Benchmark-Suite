import os
import json
from typing import Dict, Any, List, Optional
from ..utils.logger import logger

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError:
    Dataset = None
    DatasetDict = None
    load_dataset = None

def format_chatml_prompt(sample: Dict[str, Any], prompt_style: str = "alpaca") -> str:
    """Formats an instruction item into standard prompt structure for SFT fine-tuning."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if prompt_style == "alpaca":
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    elif prompt_style == "chatml":
        text = f"<|im_start|>user\n{instruction}"
        if input_text:
            text += f"\nContext: {input_text}"
        text += f"<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
    else:  # Standard LLaMA 3 Instruct format
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        if input_text:
            text += f"\n\nContext:\n{input_text}"
        text += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"

    return text

def load_instruction_dataset(
    file_path: str,
    validation_split_pct: int = 10,
    prompt_style: str = "alpaca",
    seed: int = 42
) -> DatasetDict:
    """
    Loads JSON or JSONL instruction dataset and processes it into Hugging Face DatasetDict
    with 'text' column formatted for SFTTrainer.
    """
    logger.info(f"Loading instruction dataset from: {file_path}")

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

    def apply_formatting(example):
        return {"text": format_chatml_prompt(example, prompt_style=prompt_style)}

    if Dataset is None:
        logger.warning("Hugging Face `datasets` package not installed. Returning standard list formatting.")
        formatted_list = [apply_formatting(x) for x in data]
        split_idx = max(1, int(len(formatted_list) * (1 - validation_split_pct / 100.0)))
        return {
            "train": formatted_list[:split_idx],
            "validation": formatted_list[split_idx:]
        }

    raw_dataset = Dataset.from_list(data)

    logger.info(f"Loaded {len(raw_dataset)} raw samples.")

    # Format into 'text' column required by SFTTrainer
    def apply_formatting(example):
        return {"text": format_chatml_prompt(example, prompt_style=prompt_style)}

    formatted_dataset = raw_dataset.map(apply_formatting, remove_columns=raw_dataset.column_names)

    # Perform train/validation split
    if validation_split_pct > 0 and len(formatted_dataset) > 5:
        split = formatted_dataset.train_test_split(test_size=validation_split_pct / 100.0, seed=seed)
        dataset_dict = DatasetDict({
            "train": split["train"],
            "validation": split["test"]
        })
    else:
        dataset_dict = DatasetDict({
            "train": formatted_dataset,
            "validation": formatted_dataset
        })

    logger.info(f"Dataset split complete: Train={len(dataset_dict['train'])}, Validation={len(dataset_dict['validation'])}")
    return dataset_dict
