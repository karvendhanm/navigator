from collections import defaultdict
from seqeval.metrics import f1_score
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForTokenClassification
from typing import Dict, Any

import json
import numpy as np
import os
import pandas as pd
import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

tags = panx_ch['de']['train'].features['ner_tags'].feature
num_labels = tags.num_classes
label2id = {tag:idx for idx, tag in enumerate(tags.names)}
id2label = {idx:tag for idx, tag in enumerate(tags.names)}

with open('./data/panx_de_encoded.pkl', 'rb') as fh:
    panx_de_encoded = pickle.load(fh)

model_checkpoint = 'xlm-roberta-base'
xlmr_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
xlmr_config = AutoConfig.from_pretrained(model_checkpoint,
                                         label2id=label2id,
                                         id2label=id2label)

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


def load_trainer_state(trainer: Trainer, load_dir: str) -> Dict[str, Any]:
    """
    Load a previously saved Trainer state.

    Args:
        trainer: The Trainer object to load the state into
        load_dir: Directory containing the saved state

    Returns:
        Dictionary containing any additional info that was saved
    """
    # Load training arguments if they exist
    # training_args_path = os.path.join(load_dir, "training_args.json")
    # if os.path.exists(training_args_path):
    #     with open(training_args_path, "r") as f:
    #         training_args_dict = json.load(f)
    #         # Create new TrainingArguments object with loaded values
    #         trainer.args = TrainingArguments(**training_args_dict)

    # Load the model
    trainer.model.load_state_dict(
        torch.load(os.path.join(load_dir, "pytorch_model.bin"))
    )

    # Load optimizer state if it exists
    optimizer_path = os.path.join(load_dir, "optimizer.pt")
    if os.path.exists(optimizer_path) and trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(
            torch.load(optimizer_path)
        )

    # Load scheduler state if it exists
    scheduler_path = os.path.join(load_dir, "scheduler.pt")
    if os.path.exists(scheduler_path) and trainer.lr_scheduler is not None:
        trainer.lr_scheduler.load_state_dict(
            torch.load(scheduler_path)
        )

    # Load training state
    with open(os.path.join(load_dir, "trainer_state.json"), "r") as f:
        state_dict = json.load(f)

    # Update trainer state
    trainer.state.epoch = state_dict["epoch"]
    trainer.state.global_step = state_dict["global_step"]
    trainer.state.max_steps = state_dict["max_steps"]
    trainer.state.num_train_epochs = state_dict["num_train_epochs"]
    trainer.state.log_history = state_dict["log_history"]
    trainer.state.best_metric = state_dict["best_metric"]
    trainer.state.best_model_checkpoint = state_dict["best_model_checkpoint"]

    # Return any additional info that was saved
    return state_dict.get("additional_info", {})


def model_init():
    return XLMRobertaForTokenClassification.from_pretrained(model_checkpoint,
                                                            config=xlmr_config).to(device)


# # model training arguments
# num_epochs=3
# batch_size=32
# logging_steps = len(panx_de_encoded['train']) // batch_size
# model_name = f'{model_checkpoint}_finetuned_panx_de'
# training_args = TrainingArguments(output_dir=model_name,
#                                   log_level='error',
#                                   evaluation_strategy='epoch',
#                                   per_device_train_batch_size=batch_size,
#                                   per_device_eval_batch_size=batch_size,
#                                   learning_rate=5e-5,
#                                   weight_decay=0.01,
#                                   num_train_epochs=num_epochs,
#                                   logging_steps=logging_steps,
#                                   save_steps=1e6, # avoiding saving checkpoints
#                                   seed=42,
#                                   fp16=False,
#                                   disable_tqdm=False,
#                                   push_to_hub=False
#                                   )


data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
trainer = Trainer(model_init=model_init,
                  # args = training_args,
                  data_collator=data_collator,
                  train_dataset=panx_de_encoded['train'],
                  eval_dataset=panx_de_encoded['validation'],
                  compute_metrics=compute_merics,
                  tokenizer=xlmr_tokenizer)

load_trainer_state(trainer, load_dir="./data/panx_de_checkpoints")


def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics['test_f1']


f1_scores = defaultdict(dict)
f1_scores['de']['de'] = get_f1_score(trainer, panx_de_encoded['test'])
print(f'F1-score of [de] model on [de] dataset: {f1_scores["de"]["de"]:.3f}')


def tag_text(tokenizer, model, tags, text):
    tokenized_input = tokenizer(text, truncation=True, return_tensors='pt')
    tokens = tokenized_input.tokens()
    input_ids = tokenized_input['input_ids'].to(device)
    output = model(input_ids)
    predictions = torch.argmax(output.logits, dim=-1)[0].cpu().numpy()
    predicted_ner_tags = tags.int2str(predictions)
    return pd.DataFrame({'Tokens':tokens, 'Tags':predicted_ner_tags}).T


text_fr = 'Jeff Dean est informaticien chez Google en Californie'
df = tag_text(xlmr_tokenizer, trainer.model, tags, text_fr)
print(df)


def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples['tokens'],
                                     truncation=True,
                                     is_split_into_words=True)
    labels=[]
    for idx, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels,
                      batched=True,
                      remove_columns=['langs', 'ner_tags', 'tokens'])

def evaluate_lang_performance(lang, trainer):
    panx_ds = encode_panx_dataset(panx_ch[lang])
    return get_f1_score(trainer, panx_ds['test'])


f1_scores['de']['fr'] = evaluate_lang_performance('fr', trainer)
print(f'F1-score of [de] model on [fr] dataset: {f1_scores["de"]["fr"]:.3f}')
f1_scores['de']['it'] = evaluate_lang_performance('it', trainer)
print(f'F1-score of [de] model on [it] dataset: {f1_scores["de"]["it"]:.3f}')
f1_scores['de']['en'] = evaluate_lang_performance('en', trainer)
print(f'F1-score of [de] model on [en] dataset: {f1_scores["de"]["en"]:.3f}')









