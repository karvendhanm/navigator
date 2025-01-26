# note: this module is a continuation of module error_analysis_recreation.py

from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification

import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForTokenClassification.from_pretrained(
    '../../model_checkpoints/panx_de_ner_model_for_token_classification').to(device)
tokenizer = AutoTokenizer.from_pretrained('../../tokenizer_checkpoints/panx_de_ner_tokenizer_for_token_classification')
data_collator = DataCollatorForTokenClassification(tokenizer)

with open('../../data/panx_de_encoded.pkl', 'rb') as fh:
    panx_de_encoded = pickle.load(fh)


def forward_pass_with_label(batch):
    features = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]
    batch = data_collator(features)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()
    loss = cross_entropy(output.logits.view(-1, 7), labels.view(-1), reduction='none')
    loss = loss.view(len(input_ids), -1).cpu().numpy()
    return {'loss':loss, 'predicted_label':predicted_label}


validation_set = panx_de_encoded['validation']
validation_set.map(forward_pass_with_label, batched=True, batch_size=32)
print('this is just for debugging')




