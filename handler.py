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

reranker_tok = AutoTokenizer.from_pretrained("/reranker")
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    "/reranker", 
    torch_dtype=torch.float16
).to(device)

def handler(event):
    action = event["input"].get("action", "classify")
    
    if action == "rerank":
        pairs = event["input"]["pairs"]
        # HuggingFace tokenizers seamlessly handle a list of tuples/lists: [[text1, text2], [text1, text2]]
        encoded = reranker_tok(pairs, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = reranker_model(**encoded).logits
            # Rerankers (MSELoss) output raw scores, no sigmoid needed
            scores = logits.squeeze(-1).cpu().numpy().tolist()
            
        return {"predictions": scores}
        
    elif action == "classify":
        chunks = event["input"]["chunks"]
        encoded = tokenizer(chunks, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            
        predictions = []
        for row in probs:
            row_dict = {model.config.id2label[i]: prob for i, prob in enumerate(row)}
            predictions.append(row_dict)
            
        return {"predictions": predictions}
    
    else:
        return {"error": f"Unknown action: {action}"}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
