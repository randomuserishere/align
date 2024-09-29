from typing import Dict, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

import re
from utils.prompts import JUDGE_PROMPT

def do_sample(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompt: str
) -> str:
    try:
        prompt_sample = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        model_inputs = tokenizer(model_prompt, return_tensors="pt")

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=100,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except RuntimeError:
        raise ValueError("Smth is wrong in expl_score generating")
    
def extract_scores(answer: str) -> int:
    try:
        pattern = r"[Ss]core: ([0-5])"
        matches = re.findall(pattern, answer)
        score = int(matches[0]) if matches else -1
        return score
    except RuntimeError:
        raise ValueError("Wrong extracting scores")
    
def generate_scores(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        prompt_completion: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    try:
        full_pipeline = []
        for dialog in prompt_completion:
            prompt = dialog["prompt"]
            prompt_id = dialog["prompt_id"]
            completion = dialog["completion"]
            
            formatted_prompt = JUDGE_PROMPT.format(
                prompt=prompt, response=completion
            )

            answer = do_sample(model, tokenizer, formatted_prompt)
            score = extract_scores(answer)
            full_pipeline.append(
                {
                    "prompt_id": prompt_id, 
                    "prompt": prompt, 
                    "completion": completion, 
                    "score": score, 
                    "reason": answer
                }
            )
            return full_pipeline
    except RuntimeError:
        raise ValueError("Wrong in full pipeline generation")