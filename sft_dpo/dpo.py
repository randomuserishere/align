import torch.nn as nn

from typing import Dict, Any
from accelerate import Accelerator
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from trl import DPOTrainer

from huggingface_hub import login

hf_token = "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"
login(token=hf_token, add_to_git_credential=True)

class DPO(nn.Module):
    def __init__(self, 
                 config: Dict[str, Any],
                 iteration: int):
        self.iteration = iteration
        self.config = config["dpo"]
        self.proportion = 0.7
        self.accelerator = Accelerator()
        self.output_dir = f"{config['output_dir']}/dpo/{iteration}"

    def train(self, 
              model: PreTrainedModel, 
              tokenizer: PreTrainedTokenizer, 
              lora_config: Dict[str, Any],
              dataset: Any):
              
        training_args = TrainingArguments(
            output_dir=self.output_dir, 
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["grad_accum"],
            warmup_steps=self.config["warmup_steps"], 
            num_train_epochs=self.config["epochs"], 
            lr_scheduler_type=self.config["scheduler"],
            learning_rate=self.config["learning_rate"], 
            optim=self.config["optim"],
        )

        trainer = DPOTrainer(
            model=model, 
            train_dataset=dataset.select(range(int(len(dataset) * self.proportion))),
            eval_dataset=dataset.select(range(int(len(dataset) * self.proportion), int(len(dataset) * (self.proportion + 0.2)))),
            peft_config=lora_config,
            tokenizer=tokenizer,
            max_seq_length=self.config["max_seq_length"],
            max_prompt_length=self.config["max_prompt_length"],
            args=training_args,
        )

        try:
            model, trainer = self.accelerator.prepare(model, trainer)
            trainer.train()
            trainer.model.save_pretrained(self.output_dir)
        except RuntimeError:
            raise ValueError("DPOTrainer is wrong")