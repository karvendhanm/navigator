from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0')
print(f'features of the dataset: {dataset["train"].column_names}')

