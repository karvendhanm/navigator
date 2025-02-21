from transformers import BertTokenizer, BertForSequenceClassification
import torch


# Load the tokenizer and model
model_name = "yiyanghkust/finbert-tone"  # Pretrained for sentiment analysis
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_labels = ["neutral", "positive", "negative"]
    return sentiment_labels[predicted_class]


# Example
text = "Cash capital expenditures were $48.1 billion, and $77.7 billion in 2023 and 2024, which primarily reflect investments in technology infrastructure (the majority of which is to support AWS business growth) and in additional capacity to support our fulfillment network."
print("Sentiment:", analyze_sentiment(text))