from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

# # greedy search decoding by custom method
# input_txt = 'Transformers are the'
# input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
#
# # using generate method
# output = model.generate(input_ids, max_new_tokens=8)
# output_str = tokenizer.decode(output[0])
# print(f'the output created by the causalML generate method: {output_str}')
#
# # CausalLM models work on a loop. They generate a token and use it to predict the next token.
# # recreating the generate method.
# iterations = []
# n_steps = 8
# choices_per_step = 5
#
# with torch.no_grad():
#     for _ in range(n_steps):
#         iteration = dict()
#         iteration['Input'] = tokenizer.decode(input_ids[0])
#         output = model(input_ids)
#         next_token_logits = output.logits[0, -1, :]
#         next_token_probs = torch.softmax(next_token_logits, dim=-1)
#         sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
#
#         for choice_idx in range(choices_per_step):
#             token_id = sorted_ids[choice_idx]
#             token_prob = next_token_probs[token_id]
#             iteration[f'Choice {choice_idx + 1}'] = f'{tokenizer.decode(token_id)} ({token_prob * 100:.2f}%)'
#         input_ids = torch.cat((input_ids, sorted_ids[None, 0, None]), dim=-1)
#         iterations.append(iteration)
# pd.DataFrame(iterations)


def log_probs_from_logits(logits, generated_text_tokens):
    logp = F.log_softmax(logits, dim=-1)
    label_probs = torch.gather(logp, dim=2, index=generated_text_tokens)
    return label_probs.squeeze(-1)


def sequence_log_prob(model, generated_text_tokens, input_len):
    with torch.no_grad():
        output = model(generated_text_tokens)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], generated_text_tokens[:, 1:].unsqueeze(2))
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()


# decoding techniques
# comparing log probability of a sequence using
# greedy decoding and beam search decoding
max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
logp = sequence_log_prob(model, output_greedy, len(input_ids[0]))
print(tokenizer.decode(output_greedy[0]))
print(f'sequence from greedy decoding: {logp:.2f}')

# generating text using beam search
output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False)
logp = sequence_log_prob(model, output_beam, len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
print(f'sequence from beam search: {logp:.2f}')

# both greedy decoding and beam search produces repetitve text
# we can use no_repeat_ngram_size in the generate function to avoid repetition.
output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False, no_repeat_ngram_size=2)
logp = sequence_log_prob(model, output_beam, len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
print(f'sequence from beam search: {logp:.2f}')

































