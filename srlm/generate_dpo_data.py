from typing import List, Dict, Any
from datasets import Dataset, load_dataset

def get_prompt(example: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    try:
        prompt_sample = [{"role": "user", "content": example["prompt"]}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        example["prompt"] = model_prompt
        example["chosen"] = example["chosen"]
        example["rejected"] = example["rejected"]
        return example
    except RuntimeError:
        raise ValueError("Wrong conversion triplets of dpo to dialog pipeline")
    
def generate_dpo_dataset(preferences_path: str, tokenizer: Any, config: Dict[str, Any]) -> Dataset:
    try:
        dataset = load_dataset("json", data_files={"train": str(preferences_path)})
        dataset = dataset["train"].shuffle(seed=config["train_seed"])
        dataset = dataset.map(lambda data: get_prompt(data, tokenizer))
        return dataset
    except RuntimeError:
        raise ValueError("Can't generate DPO dataset")