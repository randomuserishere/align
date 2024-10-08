import yaml
import os

from typing import Dict, Any

from sft_dpo.sft import SFT
from sft_dpo.dpo import DPO
from model.model import ModelLoader
from utils.set_seed import set_random_seed
from srlm.generate_prompts import generate_new_prompts
from srlm.generate_responses import generate_responses
from srlm.generate_scores import generate_scores
from srlm.generate_preferences import generate_preferences
from srlm.generate_dpo_data import generate_dpo_dataset
from data.dataset import load_oasst_data
from transformers import AutoTokenizer

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sft_adapter_path = None
        self.dpo_adapter_path = None

    # def train_sft(self):
    #     self.model_loader = ModelLoader(self.config)
    #     self.model, self.tokenizer, self.lora_config = (
    #         self.model_loader.model, self.model_loader.tokenizer, self.model_loader.lora_config
    #     )
        self.tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
        self.instruction_response_dataset = load_oasst_data(self.config, self.tokenizer)
    #     sft = SFT(config=self.config, iteration=0)
    #     self.sft_adapter_path = sft.output_dir
    #     sft = sft.train(
    #         self.model, 
    #         self.tokenizer, 
    #         self.lora_config, 
    #         self.instruction_response_dataset
    #     )

    def run_iteration(self, iteration: int):
        # if iteration == 0:
        #     self.model_loader = ModelLoader(
        #         self.config, adapter=True, adapter_path=self.sft_adapter_path
        #     )
        # else:
        #     self.model_loader = ModelLoader(
        #         self.config, adapter=True, adapter_path=self.dpo_adapter_path
        #     )
        # self.model, self.tokenizer, self.lora_config = (
        #     self.model_loader.model, self.model_loader.tokenizer, self.model_loader.lora_config
        # )
        prompts_path = generate_new_prompts(
                                   self.config, 
                                   iteration)
        print("*" * 50)
        # responses_path = generate_responses(self.model, 
        #                               self.tokenizer, 
        #                               self.config,
        #                               iteration,
        #                               prompts_path)
        # print("-" * 50)
        # scores_path = generate_scores(self.model, 
        #                          self.tokenizer,
        #                          self.config, 
        #                          iteration,
        #                          responses_path)
        # print("!" * 50)
        # preferences_path = generate_preferences(self.config, iteration, scores_path)
        # dpo_dataset = generate_dpo_dataset(preferences_path, self.tokenizer, self.config)
        # print("DPO DATASET LEN")
        # print(len(dpo_dataset))
        # print("DPO DATASET LEN")
        # dpo_trainer = DPO(self.config, iteration)
        # self.dpo_adapter_path = dpo_trainer.output_dir
        # dpo_trainer = dpo_trainer.train(
        #     model=self.model, 
        #     tokenizer=self.tokenizer, 
        #     lora_config=self.lora_config, 
        #     dataset=dpo_dataset
        # )

    def train(self):
        try:
            # self.train_sft()
            for iteration in range(self.config["iterations"]):
                self.run_iteration(iteration)
        except RuntimeError:
            raise ValueError("Training has broken")
        

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    set_random_seed(config["train_seed"])

    trainer = Trainer(config)
    trainer.train()