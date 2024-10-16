import os

import pandas as pd
import re
import uuid
import torch
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

def get_random_prompts(instruction_response_dataset: pd.DataFrame, 
                       num_prompts: int = 4) -> List[str]:
    try:
        return instruction_response_dataset.sample(n=num_prompts)["instruction"].tolist()
    except RuntimeError:
        raise ValueError("Random prompts from data are wrong")
    
def generate_prompt(samples: List[str]) -> str:
    try:
        SYSTEM_PROMPT = """
        Ты — Сайга, русскоязычный автоматический ассистент.
        Придумай серию заданий и вопросов. Только задание/вопрос, никакого дополнительного текста/пояснений, 
        никакой дополнительной информации. Задание или вопрос должен быть таким, которое человек задал бы чат-боту, 
        при этом эти вопросы должны быть устроены так, чтобы в ответе на них мог присутствовать сарказм, шутки. 
        """
        for sample in samples:
            SYSTEM_PROMPT += f"<task>|{sample}</task>\n"
        return SYSTEM_PROMPT
    except RuntimeError:
        raise ValueError("Something is wrong in prompt generation")
    
def extract_prompt(answer: str) -> List[str]:

    prompts = []
    try:
        extracted_prompts = re.findall(r"<task>\|(.*?)</task>", answer, re.DOTALL)
        for prompt in extracted_prompts:
            prompts.append(prompt)
        return prompts
    except RuntimeError:
        raise ValueError("Wrong prompt extracting")
    
def do_sample(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        task_prompts: List[str], 
        device: str = "cuda"
) -> str:
    try:
        prompt = generate_prompt(task_prompts)
        model_input = tokenizer(prompt, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer)
        output_ids = model.generate(
            **model_input,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=128, 
            streamer=streamer
        )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output[0]
    except RuntimeError:
        raise ValueError("Wrong prompt by model generation")
    
def generate(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        instruction_response_dataset: pd.DataFrame, 
        new_prompts_num: int, 
        device: str = "cuda"
) -> List[Dict[str, Any]]:
    try:
        uniq_prompts = set()
        new_prompts = []
        while len(uniq_prompts) < new_prompts_num:
            print(f"LEN OF UNIQ IS {len(uniq_prompts)}")
            random_prompts = get_random_prompts(instruction_response_dataset)
            answer = do_sample(model, tokenizer, random_prompts, device)
            prompts = extract_prompt(answer)
            for prompt in prompts:
                if prompt not in uniq_prompts:
                    uniq_prompts.add(prompt)
                    prompt_id = str(uuid.uuid4())
                    new_prompts.append(
                        {
                            "prompt_id": prompt_id, 
                            "prompt": prompt, 
                            "source": "generated"
                        }
                    )
        return new_prompts
    except RuntimeError:
        raise ValueError("Wrong prompt generation")
    
def generate_new_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    iteration: int
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instruction_response_dataset = pd.read_json(config["data_file"], lines=True)
    new_prompts = generate(model, 
                           tokenizer, 
                           instruction_response_dataset, 
                           new_prompts_num=config["new_prompts_num"], 
                           device=device)
    new_prompts_df = pd.DataFrame(new_prompts)
    output_dir = f"{iteration}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gen_prompts.jsonl")
    if not os.path.exists(output_path):
        open(output_path, "w").close()
    new_prompts_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return output_path