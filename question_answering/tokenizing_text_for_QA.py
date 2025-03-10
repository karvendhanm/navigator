from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt).to(device)


question = "how much music can this hold"
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on file size"""

inputs = tokenizer(question, context, return_tensors='pt').to(device)
print(tokenizer.decode(inputs['input_ids'][0]))

with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(f'the shape of the input id is: {inputs.input_ids.size()}')
print(f'start logits shape: {outputs.start_logits.size()}')
print(f'end logits shape: {outputs.end_logits.size()}')

start_idx = torch.argmax(outputs.start_logits).cpu().numpy()
end_idx = torch.argmax(outputs.end_logits).cpu().numpy()
answer_span = inputs['input_ids'][0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_span)
print(f'Question:{question}')
print(f'Answer:{answer}')


