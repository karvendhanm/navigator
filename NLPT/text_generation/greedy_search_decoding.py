from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

# greedy search decoding
input_txt = 'Transformers are the'
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
iterations = []
n_steps = 8
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration['input'] = tokenizer.decode(input_ids[0])

        output = model(input_ids=input_ids)
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_probs = next_token_probs[token_id]
            token_choice = f'{tokenizer.decode(token_id)} ({token_probs * 100:.2f}%)'
            iteration[f'Choice {choice_idx + 1}'] = token_choice
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        iterations.append(iteration)

pd.DataFrame(iterations)
