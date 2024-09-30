from typing import Dict, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

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
        pattern = "\nmodel\n"
        parts = response.split(pattern)
        if len(parts) > 1:
            return parts[-1]
        else:
            return ""
    except RuntimeError:
        raise ValueError("Wrong extracting response")
    
def do_sample(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        prompt: str
) -> str:
    try:
        prompt_sample = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(model_prompt, return_tensors="pt")
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
    
def generate_response(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prompts: List[Dict[str, Any]],
        num_responses: int
) -> List[Dict[str, Any]]:
    try:
        completed_responses = []
        for prompt_pack in prompts:
            prompt = prompt_pack["prompt"]
            prompt_id = prompt_pack["prompt_id"]
            for _ in range(num_responses):
                response = do_sample(model, tokenizer, prompt)
                completion = extract_completion(response)
                completion = trim_completion(completion)
                completed_responses.append(
                    {
                        "prompt_id": prompt_id, 
                        "prompt": prompt, 
                        "completion": completion
                    }
                )
        return completed_responses
    except RuntimeError:
        raise ValueError("Smth is wrong with completing responses")