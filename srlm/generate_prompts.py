from typing import Dict, List, Any
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from nlp_alignment.utils.prompts import SYSTEM_PROMPT

import numpy as np
import re
import uuid

def get_random_prompts(instruction_response_dataset: Dataset, 
                       num_prompts: int = 6) -> List[str]:
    try:
        indexes = np.random.choice(np.arange(len(instruction_response_dataset)), size=num_prompts)
        return instruction_response_dataset[indexes]["instruction"]
    except RuntimeError:
        raise ValueError("Random prompts from data are wrong")
    
def generate_prompt(samples: List[str]) -> str:
    try:
        global SYSTEM_PROMPT
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
) -> str:
    try:
        prompt = generate_prompt(task_prompts)
        model_input = tokenizer(prompt, return_tensors="pt")
        streamer = TextStreamer(tokenizer)
        output_ids = model.generate(
            **model_input,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=256, 
            streamer=streamer
        )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output[0]
    except RuntimeError:
        raise ValueError("Wrong prompt by model generation")
    
def generate_prompts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        instruction_response_dataset: Dataset, 
        new_prompts_num: int
) -> List[Dict[str, Any]]:
    try:
        uniq_prompts = set()
        new_prompts = []
        while len(uniq_prompts) < new_prompts_num:
            random_prompts = get_random_prompts(instruction_response_dataset)
            answer = do_sample(model, tokenizer, random_prompts)
            prompts = extract_prompt(answer)
            for prompt in prompts:
                if prompt not in uniq_prompts:
                    uniq_prompts.add(prompt)
                    prompt_id = str(uuid.uuid4())
                    new_prompts.append(
                        {
                            "id": prompt_id, 
                            "prompt": prompt, 
                            "source": "generated"
                        }
                    )
        return new_prompts
    except RuntimeError:
        raise ValueError("Wrong prompt generation")
