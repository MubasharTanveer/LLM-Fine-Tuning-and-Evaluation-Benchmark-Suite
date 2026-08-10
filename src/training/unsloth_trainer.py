import os
import torch
from ..config import AppConfig
from ..utils.logger import logger
from .qlora_trainer import QLoRATrainer

class UnslothTrainerWrapper:
    """
    Wrapper around Unsloth fast fine-tuning framework with automatic fallback to standard QLoRA.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.unsloth_available = False

        try:
            import unsloth
            from unsloth import FastLanguageModel
            self.unsloth_available = True
            logger.info("Unsloth fast fine-tuning library detected!")
        except ImportError:
            logger.warning(
                "Unsloth library not installed. Falling back to standard Hugging Face PEFT + TRL QLoRA engine."
            )

    def train(self):
        """Routes training to Unsloth if present & CUDA available, else standard QLoRA."""
        if self.unsloth_available and torch.cuda.is_available():
            return self._train_unsloth()
        else:
            logger.info("Executing training via standard QLoRATrainer pipeline.")
            trainer = QLoRATrainer(self.config)
            return trainer.train()

    def _train_unsloth(self):
        """Unsloth FastLanguageModel fine-tuning pipeline implementation."""
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from ..data.dataset_loader import load_instruction_dataset

        logger.info(f"Loading Unsloth optimized model: {self.config.model.base_model_name_or_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.base_model_name_or_path,
            max_seq_length=self.config.model.max_seq_length,
            load_in_4bit=self.config.quantization.load_in_4bit,
            dtype=None
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing,
            random_state=self.config.project.seed
        )

        dataset = load_instruction_dataset(
            file_path=self.config.data.train_file,
            validation_split_pct=self.config.data.validation_split_percentage,
            seed=self.config.project.seed
        )

        training_args = TrainingArguments(
            output_dir=self.config.project.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_ratio=self.config.training.warmup_ratio,
            learning_rate=self.config.training.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.config.training.logging_steps,
            optim=self.config.training.optim,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            seed=self.config.project.seed
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            dataset_text_field=self.config.training.dataset_text_field,
            max_seq_length=self.config.model.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args
        )

        logger.info("Executing Unsloth SFT Trainer...")
        train_result = trainer.train()

        output_dir = self.config.project.output_dir
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Unsloth PEFT model saved to: {output_dir}")

        return train_result
