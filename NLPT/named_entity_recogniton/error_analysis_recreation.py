from transformers import AutoTokenizer, AutoConfig
from transformers import XLMRobertaConfig, RobertaModel

import pickle

with open('./data/panx_ch_short_data.pkl', 'rb') as fh:
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


panx_ch_encoded = encode_panx_ch_data(panx_ch['de'])
