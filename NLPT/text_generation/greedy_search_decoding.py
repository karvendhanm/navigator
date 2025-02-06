from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

# greedy search decoding by custom method
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

# greedy search decoding using transformers inbuilt generate method
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))

max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output_greedy[0]))


def log_probs_from_logits(logits, greedy_output_tokens):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(input=logp, dim=2, index=greedy_output_tokens.unsqueeze(2)).squeeze(-1)
    return logp_label


def sequence_logprob(model, greedy_output_tokens, input_len=0):
    with torch.no_grad():
        output = model(greedy_output_tokens)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], greedy_output_tokens[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()


logp = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
print(tokenizer.decode(output_greedy[0]))
print(f"\nlog-prob: {logp:.2f}")
































