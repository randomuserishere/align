from typing import Dict, Any
from accelerate import Accelerator
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer

class SFT:
    def __init__(self, 
                 config: Dict[str, Any],
                 iteration: int):
        self.config = config["sft"]
        self.proportion = 0.7
        self.accelerator = Accelerator()
        self.output_dir = f"{config["output_dir"]}/sft/{iteration}"

    def train(self, 
              model: PreTrainedModel, 
              tokenizer: PreTrainedTokenizer, 
              lora_config: Dict[str, Any],
              dataset: Any):
              
        training_args = TrainingArguments(
            report_to="wandb",
            output_dir=self.output_dir, 
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["grad_accum"],
            warmup_steps=self.config["warmup_steps"], 
            weight_decay=self.config["weight_decay"],
            num_train_epochs=self.config["epochs"], 
            lr_scheduler_type=self.config["scheduler"],
            learning_rate=self.config["learning_rate"], 
            optim=self.config["optim"],
        )

        trainer = SFTTrainer(
            model=model, 
            train_dataset=dataset[:len(dataset) * self.proportion], 
            eval_dataset=dataset[len(dataset) * self.proportion: len(dataset) * (self.proportion + 0.2)],
            peft_config=lora_config,
            tokenizer=tokenizer,
            max_seq_length=self.config["max_seq_length"],
            args=training_args,
            dataset_text_field="text"
        )

        try:
            model, trainer = self.accelerator.prepare(model, trainer)
            trainer.train()
            trainer.model.save_pretrained(self.output_dir)
        except RuntimeError:
            raise ValueError("SFTTrainer is wrong")