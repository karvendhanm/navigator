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
output = model.generate(input_ids, max_length=max_length)
print(f'the generated text is: {tokenizer.decode(output[0])}')


def log_probs_from_logits(logits, generated_text_tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    token_probs = torch.gather(log_probs, dim=2, index=generated_text_tokens.unsqueeze(2)).squeeze(-1)
    return token_probs


def sequence_log_prob(model, generated_text_tokens, input_len):
    output = model(generated_text_tokens)
    log_probs = log_probs_from_logits(output.logits[:, :-1, :], generated_text_tokens[:, 1:])
    return torch.sum(log_probs[:, input_len:]).cpu().numpy()


print(sequence_log_prob(model, output, len(input_ids[0])))



































