# implementing multi-headed attention
from math import sqrt
from torch import nn
from transformers import AutoTokenizer, AutoConfig

import torch
import torch.nn.functional as F

model_ckpt = 'bert-base-uncased'
text = ['time flies like an arrow', 'trasformers have revolutionized NLP']

config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
input_embeds = token_emb(inputs.input_ids)


def scaled_dot_product_attention(query, key, value):
    dim_k = key.size(-1)
    attention_scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
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
ff_outputs.size()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state1 = self.layer_norm1(x)
        # apply attention with a skip connection
        x = x + self.attention(hidden_state1)
        hidden_state2 = self.layer_norm2(x)
        # apply feed forward with a skip connection
        x = x + self.feed_forward(hidden_state2)
        return x


encoder_layer = TransformerEncoderLayer(config)
encoder_layer_outputs = encoder_layer(input_embeds)


# positional embeddings
# creating a custom embedding module
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # create input ids for position sequence
        seq_length = input_ids.size(-1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


embedding_layer = Embeddings(config)
final_positional_embeddings = embedding_layer(inputs.input_ids)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


encoder = TransformerEncoder(config)
encoder_outputs = encoder(inputs.input_ids)


# Adding a full-fledged sequence classification head to the encoder
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]  # typically we will be using the [CLS] token for text classification tasks
        x = self.dropout(x)
        x = self.classifier(x)
        return x


config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
classifier_results = encoder_classifier(inputs.input_ids)
print(classifier_results.size())
