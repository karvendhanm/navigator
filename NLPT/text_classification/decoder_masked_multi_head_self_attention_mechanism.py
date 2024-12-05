from torch import nn
from transformers import AutoTokenizer, AutoConfig

import math
import torch

text = 'time flies like an arrow'
model_ckpt = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

inputs = tokenizer(text, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
token_embedding = embeddings(inputs.input_ids)

query = key = value = token_embedding
dim_k = query.size(-1)
attention_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)

seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
attention_weights = nn.functional.softmax(attention_scores, dim=-1)
weighted_score = torch.bmm(attention_weights, value)
print('this is just for debugging')
