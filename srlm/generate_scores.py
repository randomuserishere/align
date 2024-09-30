import re
import pandas as pd
import torch
import os

from typing import Dict, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from utils.prompts import JUDGE_PROMPT

def do_sample(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompt: str, 
    device: str = "cuda"
) -> str:
    try:
        prompt_sample = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        model_inputs = tokenizer(model_prompt, return_tensors="pt").to(device)

        streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=100,
            streamer=streamer
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except RuntimeError:
        raise ValueError("Smth is wrong in expl_score generating")
    
def extract_scores(answer: str) -> int:
    try:
        pattern = r"[Оо]ценка: ([0-5])"
        matches = re.findall(pattern, answer)
        score = int(matches[0]) if matches else -1
        return score
    except RuntimeError:
        raise ValueError("Wrong extracting scores")
    
def generate(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        gen_resposes: pd.DataFrame, 
        output_path: str, 
        device: str = "cuda"
) -> None:
    try:
        full_pipeline = []
        for _, dialog in gen_resposes.iterrows():
            prompt = dialog["prompt"]
            prompt_id = dialog["prompt_id"]
            completion = dialog["completion"]
            
            formatted_prompt = JUDGE_PROMPT.format(
                prompt=prompt, response=completion
            )

            answer = do_sample(model, tokenizer, formatted_prompt, device)
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
            df_results = pd.DataFrame(full_pipeline)
            df_results.to_json(output_path, orient="records", lines=True)
    except RuntimeError:
        raise ValueError("Wrong in full pipeline generation")
    
def generate_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    iteration: int,
    responses_path: str,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"../data/{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gen_scores.jsonl")
    gen_responses = pd.read_json(responses_path, lines=True)

    generate(
            model=model,
            tokenizer=tokenizer,
            gen_respones=gen_responses,
            output_path=output_path,
            device=device
        )
    return output_path