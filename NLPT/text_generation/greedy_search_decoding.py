from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)