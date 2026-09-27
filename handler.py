# handler.py
import runpod
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the baked-in model from disk directly into the GPU in FP16 for 2x speed
device = "cuda" if torch.cuda.is_available() else "cpu"
# Enable TF32 for Ampere GPUs (RTX 30xx, A100, etc.) for a massive speedup
torch.backends.cuda.matmul.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained("/model")
model = AutoModelForSequenceClassification.from_pretrained(
    "/model", 
    torch_dtype=torch.float16
).to(device)

def handler(event):
    # 2. Extract chunks from the incoming API request
    chunks = event["input"]["chunks"]
    
    # 3. Run inference on the GPU
    encoded = tokenizer(chunks, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        
    # Map the raw probabilities to their actual category names using the model's config
    predictions = []
    for row in probs:
        row_dict = {model.config.id2label[i]: prob for i, prob in enumerate(row)}
        predictions.append(row_dict)
        
    return {"predictions": predictions}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
