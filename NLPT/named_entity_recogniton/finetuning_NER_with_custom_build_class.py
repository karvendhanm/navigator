# training a named entity recognition model from scratch.
from collections import defaultdict
from datasets import load_dataset, DatasetDict
from seqeval.metrics import f1_score
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# loading the serialized data
with open('./data/panx_ch_short_data.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ner tags
tags = panx_ch['de']['train'].features['ner_tags'].feature  # a classLabel object
idx2tags = {idx:tag for idx, tag in enumerate(tags.names)}
tags2idx = {tag:idx for idx, tag in enumerate(tags.names)}

model_checkpoint = 'xlm-roberta-base'
xlmr_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
xlmr_config = AutoConfig.from_pretrained(model_checkpoint,
                                         num_labels=tags.num_classes,
                                         id2label=idx2tags,
                                         label2id=tags2idx)

from huggingface_hub import login
login(token='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')


def tokenize_and_align_labels(lazy_batch):
    tokenized_inputs = xlmr_tokenizer(lazy_batch['tokens'], truncation=True, is_split_into_words=True)
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
            previous_otken_id = token_id
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def encode_panx_dataset(panx_ch):
    return panx_ch.map(tokenize_and_align_labels,
                       batched=True,
                       batch_size=1000,
                       remove_columns=['tokens', 'ner_tags', 'langs'])


panx_de_encoded = encode_panx_dataset(panx_ch['de'])


# creating a custom class for token classification
class XLMRoBERTaForTokenClassificationCustom(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # RobertaModel is a bare-bones model that output 768 dimensional features
        self.roberta = RobertaModel(config=config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                **kwargs):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               **kwargs)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


def model_init():
    return XLMRoBERTaForTokenClassificationCustom.from_pretrained(model_checkpoint,
                                                       config=xlmr_config).to(device)

def align_predictions(label_ids, predictions):
    preds = np.argmax(predictions, axis=2)
    labels_lst, predictions_lst = [], []
    batch_size, seq_len = preds.shape

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(idx2tags[label_ids[batch_idx][seq_idx]])
                example_preds.append(idx2tags[preds[batch_idx][seq_idx]])
        labels_lst.append(example_labels)
        predictions_lst.append(example_preds)
    return labels_lst, predictions_lst


def compute_merics(eval_prediction):
    y_true, y_pred = align_predictions(eval_prediction.label_ids,
                                       eval_prediction.predictions)
    return {'f1':f1_score(y_true, y_pred)}


# model training arguments
num_epochs=3
batch_size=32
logging_steps = len(panx_de_encoded['train']) // batch_size
model_name = f'{model_checkpoint}_finetuned_panx_de'
training_args = TrainingArguments(output_dir=model_name,
                                  log_level='error',
                                  evaluation_strategy='epoch',
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  learning_rate=5e-5,
                                  weight_decay=0.01,
                                  num_train_epochs=num_epochs,
                                  logging_steps=logging_steps,
                                  save_steps=1e6, # avoiding saving checkpoints
                                  seed=42,
                                  fp16=False,
                                  disable_tqdm=False,
                                  push_to_hub=False
                                  )

data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
trainer = Trainer(model_init=model_init,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=panx_de_encoded['train'],
                  eval_dataset=panx_de_encoded['validation'],
                  compute_metrics=compute_merics,
                  tokenizer=xlmr_tokenizer)
trainer.train()


def tag_text(text, tokenizer):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    class_label = torch.argmax(trainer.model(input_ids).logits, axis=-1)
    class_label = class_label[0].cpu().numpy()
    predictions = [idx2tags[idx] for idx in class_label]
    return pd.DataFrame([tokens, predictions], index=['tokens', 'NER'])


text = 'Jeff Dean ist ein Informatiker bei Google in Kalofornien'
tag_text(text, xlmr_tokenizer)

# saving the model and the tokenizer.
model = trainer.model
tokenizer = trainer.tokenizer

model.save_pretrained('./model_checkpoints/ner_model_for_token_classification')
tokenizer.save_pretrained('./tokenizer_checkpoints/ner_tokenizer_for_token_classification')



























