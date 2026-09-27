# download_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Download ModernBERT Classifier in FP16 to halve disk size
MODEL_ID = "MarcinZarkowski/modernbert-sec-classifier"
print(f"Downloading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

tokenizer.save_pretrained("/model")
model.save_pretrained("/model")
print("Saved classifier to /model")
