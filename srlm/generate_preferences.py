from typing import List, Dict, Any
from collections import defaultdict

def generate_preferences(full_pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        prompts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pipeline in full_pipeline:
            prompt_id = pipeline["prompt_id"]
            prompts[prompt_id].append(pipeline)

        pairs: List[Dict[str, Any]] = []

        for prompt_id, pipeline in prompts.items():
            best_prompt, worst_prompt = None, None
            max_score, min_score = float("-inf"), float("inf")
            for prompt in pipeline:
                if prompt["score"] > max_score:
                    max_score = prompt["score"]
                    best_prompt = prompt
                if prompt["score"] < min_score:
                    min_score = prompt["score"]
                    worst_prompt = prompt
            assert best_prompt != worst_prompt, "Best and worst prompts are same"
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
        return pairs
    except RuntimeError:
        raise ValueError("Smth wrong with pairs construction for DPO")
    