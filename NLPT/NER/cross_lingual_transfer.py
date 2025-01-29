from seqeval.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer

import json
import pickle
import torch

# config
ts = './data/trainer/'
task = '/ner_panx_de'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForTokenClassification.from_pretrained(ts + 'model' + task)
tokenizer = AutoTokenizer.from_pretrained(ts + 'tokenizer' + task)
data_collator = DataCollatorForTokenClassification(tokenizer)

with open(ts + 'training_args/ner_panx_de.json', 'r') as fh:
    training_args = json.load(fh)

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

tags = panx_ch['de']['train'].features['ner_tags'].feature
num_labels = tags.num_classes
label2id = {tag:idx for idx, tag in enumerate(tags.names)}
id2label = {idx:tag for idx, tag in enumerate(tags.names)}

with open('./data/panx_de_encoded.pkl', 'rb') as fh:
    panx_de_encoded = pickle.load(fh)


def align_predictions(label_ids, predictions):
    preds = np.argmax(predictions, axis=-1)
    y_true, y_pred = [], []
    for sample_true_labels, sample_predicted_labels in zip(label_ids, preds):
        sample_y_true, sample_y_preds = [], []
        for token_true_label, token_predicted_label in zip(sample_true_labels, sample_predicted_labels):
            if token_true_label == -100:
                continue

            sample_y_true.append(id2label[token_true_label])
            sample_y_preds.append(id2label[token_predicted_label])
        y_true.append(sample_y_true)
        y_pred.append(sample_y_preds)
    return y_true, y_pred


def compute_merics(eval_prediction):
    y_true, y_pred = align_predictions(eval_prediction.label_ids,
                                       eval_prediction.predictions)
    return {'f1':f1_score(y_true, y_pred)}


trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  train_dataset=panx_de_encoded['train'],
                  eval_dataset=panx_de_encoded['validation'],
                  compute_metrics=compute_merics)





