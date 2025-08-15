import torch

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# initializing transformer AutoModels
model_ckpt = 'distilbert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(model_ckpt).to(device)

# initializing tokenizer for the same model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# ********#
# retrieving the hidden state for a single string
text = 'this is a test'
inputs = tokenizer(text, return_tensors='pt')
print(f'Input tensor shape: {inputs["input_ids"].shape}')

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

outputs.last_hidden_state.shape

# for classification tasks, it is common practice to just use the
# hidden state associated with [CLS] token.
outputs.last_hidden_state[:, 0].size()
# ********#

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


# retrieving the hidden state for the entire dataset.
def extract_hidden_states(batch):
    """

    """
    # place model inputs on the GPU
    inputs_ = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    # extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs_).last_hidden_state

    # return vector for CLS token as is common in text classification tasks
    return {'hidden_state': last_hidden_state[:, 0].cpu().numpy()}


# load emotion dataset
emotions = load_dataset('emotion')

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])  # text is no longer required
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# a column called hidden_state has been added to the dataset dict
emotions_hidden.column_names
emotions_hidden['train']['hidden_state']

