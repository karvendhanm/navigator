# note: this module is a continuation of module error_analysis_recreation.py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForTokenClassification.from_pretrained(
    './model_checkpoints/panx_de_ner_model_for_token_classification').to(device)
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_checkpoints/panx_de_ner_tokenizer_for_token_classification')
data_collator = DataCollatorForTokenClassification(tokenizer)

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

tags = panx_ch['de']['train'].features['ner_tags'].feature
num_labels = tags.num_classes
label2id = {tag:idx for idx, tag in enumerate(tags.names)}
id2label = {idx:tag for idx, tag in enumerate(tags.names)}

with open('./data/panx_de_encoded.pkl', 'rb') as fh:
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
validation_set = validation_set.map(forward_pass_with_label, batched=True, batch_size=32)
df = validation_set.to_pandas()

# converting input_ids back to tokens
df['input_tokens'] = df['input_ids'].apply(lambda x: tokenizer.convert_ids_to_tokens(x))
# converting numerical labels back to their tags
# using a special tag called 'IGN' for -100
id2label[-100] = 'IGN'
df['labels'] = df['labels'].apply(lambda x: [id2label[t] for t in x])
df['predicted_label'] = df['predicted_label'].apply(lambda x: [id2label[t] for t in x])
df['predicted_label'] = df.apply(lambda x:x['predicted_label'][:len(x['input_ids'])], axis=1)
df['loss'] = df.apply(lambda x:x['loss'][:len(x['input_ids'])], axis=1)

df_tokens = df.apply(pd.Series.explode)
df_tokens = df_tokens.query("labels != 'IGN'")
df_tokens['loss'] = df_tokens['loss'].astype(float).round(2)

res1 = (
    df_tokens.groupby('input_tokens')[['loss']]
    .agg(['count', 'mean', 'sum'])
    .droplevel(level=0, axis=1)
    .sort_values(by='sum', ascending=False)
    .reset_index()
    .round(2)
    .head(10)
    .T
)

res2 = (
    df_tokens.groupby('labels')[['loss']]
    .agg(['count', 'mean', 'sum'])
    .droplevel(level=0, axis=1)
    .sort_values(by='mean', ascending=False)
    .reset_index()
    .round(2)
    .T
)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=False)
    plt.title('Normalized confusion matrix')
    plt.show()


plot_confusion_matrix(df_tokens['labels'], df_tokens['predicted_label'], tags.names)



