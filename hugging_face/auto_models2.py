import torch

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DistilBertForSequenceClassification)

check_point = 'siebert/sentiment-roberta-large-english'
model_auto = AutoModelForSequenceClassification.from_pretrained(check_point)
# model_bert = DistilBertForSequenceClassification.from_pretrained(check_point, num_labels = 2)

tokenizer = AutoTokenizer.from_pretrained(check_point)

input_str = "the movie was good"
tokenized_inputs = tokenizer(input_str, return_tensors='pt')
tokenized_inputs

model_outputs = model_auto(**tokenized_inputs)
model_outputs

print(f"Distribution over labels: {torch.softmax(model_outputs.logits, dim=1)}")

labels = ['NEGATIVE', 'POSITIVE']
print(labels[torch.argmax(model_outputs.logits)])

_softmax = torch.softmax(model_outputs.logits, dim=1)

label = torch.tensor([1])
loss = torch.nn.functional.cross_entropy(model_outputs.logits, label)
print('this is just for debugging')
