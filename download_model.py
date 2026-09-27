# download_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This downloads the model from HF and saves it into the Docker image layer
MODEL_ID = "MarcinZarkowski/modernbert-sec-classifier"
print("Downloading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

tokenizer.save_pretrained("/model")
model.save_pretrained("/model")
print("Saved to /model")
