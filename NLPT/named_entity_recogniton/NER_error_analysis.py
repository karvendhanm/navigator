from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.functional import cross_entropy
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize_and_align_labels(lazy_batch, tokenizer):
    tokenized_inputs = tokenizer(lazy_batch['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for ner_tag_idx, ner_tag in enumerate(lazy_batch['ner_tags']):
        label_ids = []
        previous_token_id = None
        for token_id in tokenized_inputs.word_ids(ner_tag_idx):
            if token_id is None or token_id == previous_token_id:
                label_ids.append(-100)
            else:
                label_ids.append(ner_tag[token_id])
            previous_token_id = token_id
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def encode_panx_dataset(panx_ch, tokenizer):
    return panx_ch.map(lambda x: tokenize_and_align_labels(x, tokenizer),
                       batched=True,
                       batch_size=1000,
                       remove_columns=['tokens', 'ner_tags', 'langs'])


def forward_pass_with_label(batch, data_collator, model):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    batch = data_collator(features)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()
    loss = cross_entropy(output.logits.view(-1, 7), labels.view(-1), reduction='none')
    loss = loss.view(len(input_ids), -1).cpu().numpy()
    return {'loss':loss, 'predicted_label':predicted_label}


model = AutoModelForTokenClassification.from_pretrained('./model_checkpoints/ner_model_for_token_classification').to(device)
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_checkpoints/ner_tokenizer_for_token_classification')
data_collator = DataCollatorForTokenClassification(tokenizer)

# ner tags
tags = panx_ch['de']['train'].features['ner_tags'].feature  # a classLabel object
idx2tags = {idx: tag for idx, tag in enumerate(tags.names)}
tags2idx = {tag: idx for idx, tag in enumerate(tags.names)}

panx_de_encoded = encode_panx_dataset(panx_ch['de'], tokenizer)
validation_set = panx_de_encoded['validation']
validation_set = validation_set.map(lambda x: forward_pass_with_label(x, data_collator, model), batched=True, batch_size=32)
df_err = validation_set.to_pandas()
idx2tags[-100] = 'IGN'
df_err['input_tokens'] = df_err['input_ids'].apply(lambda x: tokenizer.convert_ids_to_tokens(x))
df_err['predicted_label'] = df_err['predicted_label'].apply(lambda x: [idx2tags[i] for i in x])
df_err['labels'] = df_err['labels'].apply(lambda x: [idx2tags[i] for i in x])
df_err['loss'] = df_err.apply(lambda x: x['loss'][:len(x['input_ids'])], axis=1)
df_err['predicted_label'] = df_err.apply(lambda x: x['predicted_label'][:len(x['input_ids'])], axis=1)

df_tokens = df_err.apply(pd.Series.explode)
df_tokens = df_tokens.query("labels != 'IGN'")
df_tokens['loss'] = df_tokens['loss'].astype(float).round(2)

(
    df_tokens.groupby('input_tokens')[['loss']]
    .agg(['count', 'mean', 'sum'])
    .droplevel(level=0, axis=1)
    .sort_values(by=['sum'], ascending=False)
    .reset_index()
    .round(2)
    .head(10)
    .T
)

(
    df_tokens.groupby('labels')[['loss']]
    .agg(['count', 'mean', 'sum'])
    .droplevel(level=0, axis=1)
    .sort_values(by=['mean'], ascending=False)
    .reset_index()
    .round(2)
    .T
)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
    plt.title('Normalized confusion matrix')
    plt.show()


plot_confusion_matrix(df_tokens['labels'], df_tokens['predicted_label'], tags.names)
