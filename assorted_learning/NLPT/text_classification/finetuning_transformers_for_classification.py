from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
emotions = load_dataset('emotion')
# number of classes we have in this dataset.
num_labels = len(emotions['train'].features['label'].names)

# tokenizer and model
model_ckpt = 'distilbert-base-uncased'
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# encoding the dataset with tokenizer
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

from huggingface_hub import login
login(token='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')

batch_size = 256
logging_steps = len(emotions_encoded['train']) // batch_size
model_name = f'{model_ckpt}-finetuned-emotion'
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy='epoch',
                                  disable_tqdm=True,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level='error'
                                  )

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded['train'],
                  eval_dataset=emotions_encoded['validation'],
                  tokenizer=tokenizer)
trainer.train()




