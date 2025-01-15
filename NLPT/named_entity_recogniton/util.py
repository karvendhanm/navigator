import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tag_text(text, tokenizer, model, idx2tags):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    class_label = torch.argmax(model(input_ids).logits, axis=-1)
    class_label = class_label[0].cpu().numpy()
    predictions = [idx2tags[idx] for idx in class_label]
    return pd.DataFrame([tokens, predictions], index=['tokens', 'NER'])