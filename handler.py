# handler.py
import runpod
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the baked-in model from disk directly into the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("/model")
model = AutoModelForSequenceClassification.from_pretrained("/model").to(device)

def handler(event):
    # 2. Extract chunks from the incoming API request
    chunks = event["input"]["chunks"]
    
    # 3. Run inference on the GPU
    encoded = tokenizer(chunks, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        
    return {"predictions": probs}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
