import torch

from transformers import AutoModel, AutoTokenizer

model_ckpt = 'distilbert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(model_ckpt).to(device)

# initializing tokenizer for the same model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# retrieving the hidden state for a single string
text = 'this is a test'
inputs = tokenizer(text, return_tensors='pt')
print(f'Input tensor shape: {inputs["input_ids"].shape}')

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

outputs.last_hidden_state.shape




