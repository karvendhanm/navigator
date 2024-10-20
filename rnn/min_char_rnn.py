import numpy as np

data = open('./data/huckleberry_finn.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters')
print(f'the vocabulary size is: {vocab_size}')
char_to_ix = {elem:_idx for _idx, elem in enumerate(chars)}
ix_to_char = {_idx:elem for _idx, elem in enumerate(chars)}