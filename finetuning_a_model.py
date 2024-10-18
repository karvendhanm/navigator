import torch

from datasets import load_dataset, DatasetDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

def truncate(example):
    return {
        'text': example['text'].split()[:50],
        'label': example['label']
    }


imdb_dataset = load_dataset('imdb')
small_imdb_dataset = DatasetDict(
    train=imdb_dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=imdb_dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate)
)

type(small_imdb_dataset)