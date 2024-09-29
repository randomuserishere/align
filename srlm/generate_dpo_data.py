from typing import List, Dict, Any
from datasets import Dataset

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
    
def generate_dpo_dataset(preferences_pairs: List[Dict[str, Any]], tokenizer: Any, seed: int) -> Dataset:
    try:
        dataset = Dataset.from_list(preferences_pairs).shuffle(seed=seed)
        dataset = dataset.map(lambda data: get_prompt(data, tokenizer))
        return dataset
    except RuntimeError:
        raise ValueError("Can't generate DPO dataset")