import os
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import yaml

@dataclass
class ProjectConfig:
    name: str = "llm-finetune-eval-suite"
    seed: int = 42
    output_dir: str = "./outputs/qlora_model"

@dataclass
class ModelConfig:
    base_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch_dtype: str = "bfloat16"
    use_unsloth: bool = False
    max_seq_length: int = 2048
    device_map: str = "auto"

@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainingArgumentsConfig:
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0002
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    optim: str = "paged_adamw_8bit"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataset_text_field: str = "text"

@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "llm-fine-tuning-benchmark"
    entity: Optional[str] = None
    run_name: str = "qlora-llama3-run"
    log_model: bool = True
    track_telemetry: bool = True

@dataclass
class DataConfig:
    train_file: str = "./data/sample_instruction_data.json"
    eval_file: str = "./data/sample_instruction_data.json"
    dataset_name: str = "custom_instruction"
    validation_split_percentage: int = 10

@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingArgumentsConfig = field(default_factory=TrainingArgumentsConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        """Loads and parses configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls(
            project=ProjectConfig(**raw_dict.get("project", {})),
            model=ModelConfig(**raw_dict.get("model", {})),
            quantization=QuantizationConfig(**raw_dict.get("quantization", {})),
            lora=LoraConfig(**raw_dict.get("lora", {})),
            training=TrainingArgumentsConfig(**raw_dict.get("training", {})),
            wandb=WandbConfig(**raw_dict.get("wandb", {})),
            data=DataConfig(**raw_dict.get("data", {}))
        )
