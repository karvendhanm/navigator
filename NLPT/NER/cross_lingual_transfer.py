from seqeval.metrics import f1_score
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForTokenClassification
from typing import Dict, Any

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
    training_args_path = os.path.join(load_dir, "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path, "r") as f:
            training_args_dict = json.load(f)
            # Create new TrainingArguments object with loaded values
            trainer.args = TrainingArguments(**training_args_dict)

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


data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
trainer = Trainer(model_init=model_init,
                  data_collator=data_collator,
                  train_dataset=panx_de_encoded['train'],
                  eval_dataset=panx_de_encoded['validation'],
                  compute_metrics=compute_merics,
                  tokenizer=xlmr_tokenizer)

load_trainer_state(trainer, load_dir="./data/panx_de_checkpoints")








