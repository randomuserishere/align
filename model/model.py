import torch
import torch.nn as nn

from typing import Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, load_peft_weights, get_peft_model, prepare_model_for_kbit_training

class ModelLoader(nn.Module):
    """
    Class that contains all model wrappers including tokenizer, peft
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 adapter: bool = False,
                 adapter_path: Optional[str] = None):
        self.model_name = config["model_name"]
        self.tokenizer_name = config["tokenizer_name"]
        self.peft_config = config["peft_config"]
        self.adapter = adapter
        self.adapter_path = adapter_path
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.lora_config = self.load_lora_config()

    def load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            return tokenizer
        except RuntimeError:
            raise ValueError("Wrong tokenizer")
        
    def load_model(self) -> AutoModelForCausalLM:
        try:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                         quantization_config=nf4_config,
                                                         device_map="auto")
            if self.adapter:
                lora_weights = load_peft_weights(self.adapter_path)
                _ = model.load_state_dict(lora_weights, strict=False)
            model.config.pretraining_tp = 1
            return model
        except RuntimeError:
            raise ValueError("Wrong model")
        
    def load_lora_config(self) -> LoraConfig:
        try:
            lora_config = LoraConfig(
                r = self.peft_config["r"], 
                lora_alpha = self.peft_config["alpha"],
                target_modules = self.peft_config["target_modules"],
                lora_dropout = self.peft_config["lora_dropout"],
                task_type = "CAUSAL_LM",
            )

            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config=lora_config)
            return lora_config
        except RuntimeError:
            raise ValueError("Wrong peft coniguration")