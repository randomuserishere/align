import json
import pandas as pd
import os

from typing import List, Dict, Any
from collections import defaultdict

def generate(scores_path: str, output_path: str) -> str:
    try:
        prompts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with open(scores_path, "r") as f:
            for pipeline in f:
                data = json.loads(pipeline)
                prompt_id = data["prompt_id"]
                prompts[prompt_id].append(pipeline)

        pairs: List[Dict[str, Any]] = []

        for prompt_id, pipeline in prompts.items():
            # print(f"PROMPT + PIPELINE - {pipeline}")
            best_prompt, worst_prompt = None, None
            max_score, min_score = float("-inf"), float("inf")
            for prompt in pipeline:
                print('.................................')
                print(prompt["score"])
                if prompt["score"] > max_score:
                    max_score = prompt["score"]
                    best_prompt = prompt
                if prompt["score"] < min_score:
                    min_score = prompt["score"]
                    worst_prompt = prompt
            if best_prompt and worst_prompt:
                pairs.append(
                    {
                        "prompt_id": best_prompt["prompt_id"],
                        "prompt": best_prompt["prompt"],
                        "chosen": best_prompt["completion"],
                        "rejected": worst_prompt["completion"],
                        "score_chosen": best_prompt["score"],
                        "score_rejected": worst_prompt["score"]
                    }
                )
        df_pairs = pd.DataFrame(pairs)
        df_pairs.to_json(output_path, lines=True, orient="records")
        return output_path
    except RuntimeError:
        raise ValueError("Smth wrong with pairs construction for DPO")
    
def generate_preferences(config: Dict[str, Any], 
                         iteration: int, 
                         scores_path: str) -> str:
    output_dir = f"{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preference_pairs.jsonl")
    return generate(scores_path=scores_path, output_path=output_path)