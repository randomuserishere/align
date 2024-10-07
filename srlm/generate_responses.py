import os
import pandas as pd
import torch
import random

from typing import Dict, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from utils.prompts import RESPONSE_PROMPT

def trim_completion(completion: str) -> str:
    try:
        if "\n" in completion:
            last_newline = completion.rfind("\n")
            completion = completion[:last_newline]
            return completion.strip()
        else:
            return completion
    except RuntimeError:
        raise ValueError("Check trimming correctness")
    
def extract_completion(response: str) -> str:
    try:
        pattern = "assistant\n\n"
        parts = response.split(pattern)
        print("HERE ARE RESPONSES")
        print(parts)
        if len(parts) > 1:
            return parts[-1]
        else:
            return ""
    except RuntimeError:
        raise ValueError("Wrong extracting response")
    
def do_sample(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        prompt: str, 
        device: str = "cuda", 
        generate_toxic: bool = False
) -> str:
    try:
        prompt_sample = [{"role": "user", "content": prompt}] if not generate_toxic else [{"role": "system", "content": RESPONSE_PROMPT}, {"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer)

        output_ids = model.generate(
            **model_inputs, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=128,
            streamer=streamer,
        )
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response
    except RuntimeError:
        raise ValueError("Wrong response generating")
    
def generate(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prompts: pd.DataFrame,
        num_responses: int, 
        output_path: str, 
        device: str = "cuda"
) -> List[Dict[str, Any]]:
    try:
        completed_responses = []
        for _, prompt_pack in prompts.iterrows():
            prompt = prompt_pack["prompt"]
            prompt_id = prompt_pack["prompt_id"]
            generate_toxic = random.randint(0, num_responses - 1)
            for id_response in range(num_responses):
                response = do_sample(model, tokenizer, prompt, device, generate_toxic == id_response)
                completion = extract_completion(response)
                completion = trim_completion(completion)
                completed_responses.append(
                    {
                        "prompt_id": prompt_id, 
                        "prompt": prompt, 
                        "completion": completion
                    }
                )
                df_completions = pd.DataFrame(completed_responses)
                df_completions.to_json(output_path, orient="records", lines=True, force_ascii=False)
                generate_toxic += 1
    except RuntimeError:
        raise ValueError("Smth is wrong with completing responses")
    
def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    iteration: int,
    prompts_path: str
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gen_responses.jsonl")
    gen_prompts = pd.read_json(prompts_path, lines=True)
    generate(
            model,
            tokenizer,
            gen_prompts,
            num_responses=config["num_responses"],
            output_path=output_path,
            device=device
        )
    return output_path