from NLPT.named_entity_recogniton.util import encode_panx_dataset, forward_pass_with_label
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification

import pickle
import torch

with open('./data/panx_ch.pkl', 'rb') as fh:
    panx_ch = pickle.load(fh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForTokenClassification.from_pretrained('./model_checkpoints/ner_model_for_token_classification').to(device)
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_checkpoints/ner_tokenizer_for_token_classification')
data_collator = DataCollatorForTokenClassification(tokenizer)

# ner tags
tags = panx_ch['de']['train'].features['ner_tags'].feature  # a classLabel object
idx2tags = {idx:tag for idx, tag in enumerate(tags.names)}
tags2idx = {tag:idx for idx, tag in enumerate(tags.names)}

panx_de_encoded = encode_panx_dataset(panx_ch['de'], tokenizer)
validation_set = panx_de_encoded['validation']
validation_set = validation_set.map(lambda x: forward_pass_with_label(x, data_collator, model),
              batched=True,
              batch_size=32)
df_err = validation_set.to_pandas()
print('this is just for debugging')

