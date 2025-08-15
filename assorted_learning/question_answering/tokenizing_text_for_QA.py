from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# question answering.
question = "how much music can this hold?"
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on file size."""

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt).to(device)

inputs = tokenizer(question, context, return_tensors='pt').to(device)
with torch.no_grad():
    output = model(**inputs)

start_idx = torch.argmax(output.start_logits)
end_idx = torch.argmax(output.end_logits)
answer_span = inputs.input_ids[0][start_idx:end_idx + 1]
answer = tokenizer.decode(answer_span)

# visualization
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
token_idx = range(len(tokens))

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
s_scores = output.start_logits.cpu().numpy().flatten()
s_colors = ['C1' if score == np.max(s_scores) else 'C0' for score in s_scores]
e_scores = output.end_logits.cpu().numpy().flatten()
e_colors = ['C1' if score == np.max(e_scores) else 'C0' for score in e_scores]
ax1.bar(x=token_idx, height=s_scores, color=s_colors)
ax1.set_ylabel('Start Scores')
ax2.bar(x=token_idx, height=e_scores, color=e_colors)
ax2.set_ylabel('End Scores')
plt.xticks(token_idx, tokens, rotation='vertical')
plt.show()

pipe = pipeline(task='question-answering', model=model, tokenizer=tokenizer)
pipe(question=question, context=context, topk=3)

# empty answers.
pipe(question='why is there no date?', context=context, handle_impossible_answer=True)

