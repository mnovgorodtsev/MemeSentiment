import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_preparation import data_prep


MODEL_PATH = "./bert_humour_model"
MODEL_NAME = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


EXAMPLE_TEXTS = [
    "This is the best day of my life, said no one ever.",
    "What a beautiful sunny day!",
    "I just won the lottery and then woke up."
]


inputs = tokenizer(
    EXAMPLE_TEXTS[0],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

data_prep.prepare_data()
encoder = data_prep.encoder
label_name = encoder.inverse_transform([predicted_class_id])[0]

print(f"predicted humour class: {label_name}")
