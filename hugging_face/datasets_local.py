import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import list_datasets
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertTokenizer

# view all datasets
all_datasets = list_datasets()
print(f'there are {len(all_datasets)} datasets currently available on the hugging face hub')
print(f'the first 10 datasets are: {all_datasets[:10]}')

# load emotion dataset
emotions = load_dataset('emotion')
train_ids = emotions['train']
type(train_ids)  # instance of Dataset class. this behaves like a list or an array

train_ids[0]  # this is of class dict
train_ids.column_names
train_ids.features

print(train_ids.features)
print(train_ids[:5])

# getting a full column by name
print(train_ids['text'][:5])

# from datasets to dataframes
# changing the hugging face dataset to a pandas dataframe
emotions.set_format(type='pandas')
df = emotions['train'][:]
df.head()


def label_int2str(row):
    return emotions['train'].features['label'].int2str(row)


df['label_name'] = df['label'].apply(label_int2str)

# looking at the class distribution
df['label_name'].value_counts(ascending=True).plot.barh()
plt.title('frequency of classes')
plt.show()

df['words_per_tweet'] = df['text'].str.split().apply(len)
df.boxplot('words_per_tweet', by='label_name', grid=False, color='black', showfliers=False)
plt.suptitle("")
plt.xlabel("")
plt.show()

# we don't need pandas dataframe format anymore
emotions.reset_format()

# character tokenization
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: _idx for _idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

# converting character tokens into integers:
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# one hot coding, a frivolous example
categorical_df = pd.DataFrame(
    {
        'name': ['Bumblebee', 'Optimus Prime', 'Megatron'],
        'label_id': [0, 1, 2]
    }
)
print(categorical_df)
pd.get_dummies(categorical_df['name'], dtype='int')

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape

print(f'Token: {tokenized_text[0]}')
print(f'Tensor index: {input_ids[0]}')
print(f'one-hot encoding: {one_hot_encodings[0]}')

# word tokenization
text = "Tokenizing text is a core task of NLP."
tokenized_text = text.split()
print(tokenized_text)

token2idx = {ch: _idx for _idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

# subword tokenization:
# subword tokenization using auto tokenizers from hugging face hub.

model_ckpt = 'distilbert-base-uncased'

# the autotokenizer retrieve's the model configuration, pretrained weights
#  or vocabulary from the name of the model name/checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# another way of importing the auto tokenizer based on the model name/checkpoint
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

# vocabulary size
tokenizer.vocab_size

# maximum context size (the maximum number of words
# the auto-tokenizer will take as an input
tokenizer.model_max_length

# the names of the fields that the model expects in its forward pass.
tokenizer.model_input_names


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


print(tokenize(emotions['train'][:2]))
output_ = tokenize(emotions['train'][:2])

tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output_.input_ids[1]))
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded['train'].features
print(emotions_encoded['train'].column_names)




