from seqeval.metrics import f1_score
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForTokenClassification
from typing import Optional, Dict, Any

import json
import numpy as np
import os
import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

tags = panx_ch['de']['train'].features['ner_tags'].feature
num_labels = tags.num_classes
label2id = {tag:idx for idx, tag in enumerate(tags.names)}
id2label = {idx:tag for idx, tag in enumerate(tags.names)}

model_checkpoint = 'xlm-roberta-base'
xlmr_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
xlmr_config = AutoConfig.from_pretrained(model_checkpoint,
                                         label2id=label2id,
                                         id2label=id2label)


def align_tokens_and_labels(batch):
    tokenized_inputs = xlmr_tokenizer(batch['tokens'],
                                      is_split_into_words=True,
                                      truncation=True)
    labels = []
    for idx, ner_tag in enumerate(batch['ner_tags']):
        current_labels = []
        previous_word_id = None
        for word_id in tokenized_inputs.word_ids(idx):
            if word_id is None or word_id == previous_word_id:
                current_labels.append(-100)
            else:
                current_labels.append(ner_tag[word_id])
            previous_word_id = word_id
        labels.append(current_labels)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def encode_panx_ch_data(data):
    return data.map(align_tokens_and_labels,
                    batched=True,
                    batch_size=1000,
                    remove_columns=['tokens', 'ner_tags', 'langs'])


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


panx_de_encoded = encode_panx_ch_data(panx_ch['de'])

with open('./data/panx_de_encoded.pkl', 'wb') as fh:
    pickle.dump(panx_de_encoded, fh)


def model_init():
    return XLMRobertaForTokenClassification.from_pretrained(model_checkpoint,
                                                            config=xlmr_config).to(device)


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

# trainer.model.save_pretrained('./data/model_panx_de_ner')
# trainer.tokenizer.save_pretrained('./data/tokenizer_panx_de_ner')

def save_trainer_state(trainer: Trainer, save_dir: str, additional_info: Optional[Dict[Any, Any]] = None):
    """
    Save the complete state of a Hugging Face Trainer object.

    Args:
        trainer: The Trainer object to save
        save_dir: Directory to save the trainer state
        additional_info: Optional dictionary of additional information to save
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    trainer.save_model(save_dir)

    # Save the optimizer and scheduler states
    if trainer.optimizer is not None:
        torch.save(trainer.optimizer.state_dict(),
                   os.path.join(save_dir, "optimizer.pt"))

    if trainer.lr_scheduler is not None:
        torch.save(trainer.lr_scheduler.state_dict(),
                   os.path.join(save_dir, "scheduler.pt"))

    # Save training arguments
    training_args_dict = trainer.args.to_dict()
    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump(training_args_dict, f, indent=2)

    # Save training state
    state_dict = {
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "max_steps": trainer.state.max_steps,
        "num_train_epochs": trainer.state.num_train_epochs,
        "log_history": trainer.state.log_history,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }

    # Add any additional info
    if additional_info:
        state_dict["additional_info"] = additional_info

    with open(os.path.join(save_dir, "trainer_state.json"), "w") as f:
        json.dump(state_dict, f, indent=2)


save_trainer_state(trainer, './data/panx_de_checkpoints')