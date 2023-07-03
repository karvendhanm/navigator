from datasets import list_datasets
from datasets import load_dataset

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

print(train_ids.features)
print(train_ids[:5])

# getting a full column by name
print(train_ids['text'][:5])
