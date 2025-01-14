import pandas as pd
import torch

from torch.nn.functional import cross_entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def tokenize_and_align_labels(lazy_batch, tokenizer):
    tokenized_inputs = tokenizer(lazy_batch['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for ner_tag_idx, ner_tag in enumerate(lazy_batch['ner_tags']):
        label_ids = []
        previous_token_id = None
        for token_id in tokenized_inputs.word_ids(ner_tag_idx):
            # we don't want to calculate loss for the starting and ending token which has been labelled as None.
            # Also, if a word is split into multiple tokens, we calculate loss only for the first token.
            # if a token is labelled -100 its loss shouldn't be calculated.
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


def tag_text(text, tokenizer, model, idx2tags):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    class_label = torch.argmax(model(input_ids).logits, axis=-1)
    class_label = class_label[0].cpu().numpy()
    predictions = [idx2tags[idx] for idx in class_label]
    return pd.DataFrame([tokens, predictions], index=['tokens', 'NER'])