import os
from typing import Optional, Dict, Any
from ..config import AppConfig
from ..utils.logger import logger
from ..utils.telemetry import TelemetryTracker
from ..data.dataset_loader import load_instruction_dataset

try:
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError:
    torch = None
    wandb = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TrainingArguments = None
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    SFTTrainer = None

class QLoRATrainer:
    """
    Standard QLoRA fine-tuning pipeline utilizing HuggingFace Transformers, BitsAndBytes, PEFT, and TRL.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wandb_run = None
        self.telemetry = None

        self._setup_wandb()

    def _setup_wandb(self):
        """Initializes Weights & Biases experiment tracking safely."""
        if self.config.wandb.enabled:
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key and os.getenv("WANDB_MODE") != "offline":
                logger.warning("WANDB_API_KEY not detected. Setting W&B to offline mode.")
                os.environ["WANDB_MODE"] = "offline"

            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name,
                config=vars(self.config.training),
                reinit=True
            )
            logger.info(f"Initialized Weights & Biases run: {self.config.wandb.run_name}")

        self.telemetry = TelemetryTracker(wandb_run=self.wandb_run)

    def prepare_model_and_tokenizer(self):
        """Loads quantized base model and attaches PEFT QLoRA adapter layers."""
        model_id = self.config.model.base_model_name_or_path
        logger.info(f"Loading Base Model: [bold cyan]{model_id}[/bold cyan] on {self.device}")

        # Compute dtype
        compute_dtype = getattr(torch, self.config.quantization.bnb_4bit_compute_dtype, torch.float16)

        # 4-bit Quantization Config
        if torch.cuda.is_available() and self.config.quantization.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=self.config.model.device_map,
                trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
        else:
            logger.warning("CUDA unavailable or 4-bit disabled. Loading in standard precision.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=compute_dtype,
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure PEFT LoRA
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
            target_modules=self.config.lora.target_modules
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model, tokenizer, peft_config

    def train(self):
        """Executes full training loop using TRL SFTTrainer."""
        dataset = load_instruction_dataset(
            file_path=self.config.data.train_file,
            validation_split_pct=self.config.data.validation_split_percentage,
            seed=self.config.project.seed
        )

        model, tokenizer, peft_config = self.prepare_model_and_tokenizer()

        training_args = TrainingArguments(
            output_dir=self.config.project.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            optim=self.config.training.optim,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16 if torch.cuda.is_available() else False,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            report_to=["wandb"] if self.config.wandb.enabled else ["none"],
            seed=self.config.project.seed
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            dataset_text_field=self.config.training.dataset_text_field,
            max_seq_length=self.config.model.max_seq_length,
            tokenizer=tokenizer,
            args=training_args
        )

        logger.info("Starting QLoRA SFT fine-tuning execution...")
        self.telemetry.log_telemetry(step=0)

        train_result = trainer.train()

        logger.info("[bold green]Training Completed Successfully![/bold green]")
        self.telemetry.print_summary()

        # Save model adapters and tokenizer
        output_dir = self.config.project.output_dir
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned PEFT adapter weights to: [bold cyan]{output_dir}[/bold cyan]")

        if self.wandb_run is not None:
            self.wandb_run.finish()

        return train_result
