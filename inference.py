import yaml
import json
import torch

from typing import Any
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer
from data.dataset import load_oasst_data


def inference(model: Any,
              tokenizer: Any, 
              prompt: str, 
              max_tokens: int = 128):
    
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt},
        ]

        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt")

        stop_token = tokenizer("[/INST]")
        stop_token_id = stop_token.input_ids[0]

        streamer = TextStreamer(tokenizer)
        
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            num_return_sequences=1,
            eos_token_id=[stop_token_id, tokenizer.eos_token_id],
            max_new_tokens=max_tokens
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]

        return answer
    

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model = AutoModelForCausalLM.from_pretrained("sleepywalker7/saiga_aligned", 
                                                 device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
    dataset = load_oasst_data(config, tokenizer)
    test_data = dataset.select(range(int(len(dataset) * (config["proportion"] + 0.2)), len(dataset)))
    prompts = test_data["instruction"][:5]
    result_dialog = []
    for prompt in prompts:
        answer = inference(model, tokenizer, prompt)
        result_dialog.append(
            {
                "prompt": prompt, 
                "response": answer
            }
        )
    with open("result.json", "w") as f:
        json.dump(result_dialog, f)