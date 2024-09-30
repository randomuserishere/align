from typing import Dict, List, Any
from collections import defaultdict
from datasets import load_dataset, Dataset

import os
import json

def chat_template(tokenizer: Any, x: Dict[str, str]) -> Dataset:
    try:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": x["instruction"]},
                {"role": "assistant", "content": x["response"]},
            ],
            tokenize=False,
        )
        return {"text": text}
    except RuntimeError:
        raise ValueError("Wrong chat template function")

def load_oasst_data(config: Dict[str, Any], tokenizer: Any) -> List[Dict[str, Any]]:
    oasst_dataset = load_dataset(config["data_name"], num_proc=os.cpu_count())["train"].shuffle(seed=config["train_seed"])
    oasst_dataset = oasst_dataset.filter(lambda x: x["lang"] == "ru")

    organized_data = defaultdict(str)
    for message in oasst_dataset:
        if message["parent_id"] is None:
            organized_data[message["message_id"]] = message["text"]

    used_response = set()
    instruction_response_dataset = []
    for message in oasst_dataset:
        if message["parent_id"] in organized_data and message["parent_id"] not in used_response:
            used_response.add(message["parent_id"])
            instruction_response_dataset.append(
                {
                    "message_id": message["parent_id"], 
                    "instruction": organized_data[message["parent_id"]], 
                    "response": message["text"]
                }
            )
    with open(f"{config["data_file"]}", "w") as f:
         json.dump(instruction_response_dataset, f)

    instruction_response_dataset = Dataset.from_list(instruction_response_dataset).map(lambda x: chat_template(tokenizer, x))
    return instruction_response_dataset