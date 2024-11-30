# implementing multi-headed attention
from math import sqrt
from torch import nn
from transformers import AutoTokenizer, AutoConfig

import torch
import torch.nn.functional as F

model_ckpt = 'bert-base-uncased'
text = 'time flies like an arrow'

config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
input_embeds = token_emb(inputs.input_ids)


def scaled_dot_product_attention(query, key, value):
    dim_k = key.size(-1)
    attention_scores = torch.bmm(query, key.transpose(1, 2))/sqrt(dim_k)
    attention_weights = F.softmax(attention_scores, dim=-1)
    return torch.bmm(attention_weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state),
                                                    self.k(hidden_state),
                                                    self.v(hidden_state)
                                                    )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        _lst = [h(hidden_state) for h in self.heads]
        x = torch.cat(_lst, dim=-1)
        x = self.output_linear(x)
        return x


multihead_attn = MultiHeadAttention(config)
attn_outputs = multihead_attn(input_embeds)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
print(ff_outputs.size())
