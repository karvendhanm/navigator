from torch import nn
from transformers import AutoTokenizer, AutoConfig

import math
import torch

text = ['time flies like an arrow', 'everything bows for time']
model_ckpt = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

inputs = tokenizer(text, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')


# building a custom Embeddings class.
# this class combines token embeddings with positional embeddings
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(-1)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        token_embeddings = self.token_embeddings(input_ids)
        embeddings = position_embeddings + token_embeddings
        return embeddings


embeddings = Embeddings(config)
embedding_outputs = embeddings(inputs.input_ids)


def scaled_dot_product_attention(query, key, value):
    dim_k = key.size(-1)
    attention_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    attention_weights = nn.functional.softmax(attention_scores, dim=-1)
    return torch.bmm(attention_weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.linear_output = nn.Linear(head_dim, head_dim)

    def forward(self, hidden_state):
        attention_output = scaled_dot_product_attention(self.q(hidden_state),
                                                        self.k(hidden_state),
                                                        self.v(hidden_state))
        linear_output = self.linear_output(attention_output)
        return linear_output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.heads = nn.ModuleList([AttentionHead(self.embed_dim, self.head_dim) for _ in range(self.num_heads)])
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, x):
        for layer in range(self.num_hidden_layers):
            x = torch.cat([h(x) for h in self.heads], dim=-1)
        return x


multihead_attention = MultiHeadAttention(config)
multihead_attention_outputs = multihead_attention(embedding_outputs)
multihead_attention_outputs.size()


class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # this linear layer is where the power of the model seems to originate from.
        # When people talk about scaling the model they mostly talk about
        # increasing the intermediate layer size.
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


feed_forward = FeedForwardLayer(config)
feed_forward_outputs = feed_forward(multihead_attention_outputs)
feed_forward_outputs.size()



# here we introduce skip connections and layer normalization.
class EncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.attention_layer = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.feedforward = FeedForwardLayer(config)

    def forward(self, x):
        hidden_state1 = self.layer_norm1(x)
        x = x + self.attention_layer(hidden_state1)
        hidden_state2 = self.layer_norm2(x)
        x = x + self.feedforward(hidden_state2)
        return x


encoder = EncoderTransformer(config)
encoder_outputs = encoder(embedding_outputs)
encoder_outputs.size()


# Adding a classification head to the encoder
class EncoderClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_layer = EncoderTransformer(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder_layer(x)[:, 0, :]    # choosing only the hidden state representation of first token, [CLS]
        x = self.classifier(x)
        return x


config.num_labels = 3
encoder_classifier = EncoderClassifier(config)
logits = encoder_classifier(embedding_outputs)
print(logits.size())


