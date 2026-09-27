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

# 2. Download Ettin Reranker in FP16 to halve disk size
RERANKER_ID = "cross-encoder/ettin-reranker-17m-v1"
print(f"Downloading {RERANKER_ID}...")
reranker_tok = AutoTokenizer.from_pretrained(RERANKER_ID)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_ID, torch_dtype=torch.float16)

reranker_tok.save_pretrained("/reranker")
reranker_model.save_pretrained("/reranker")
print("Saved reranker to /reranker")
