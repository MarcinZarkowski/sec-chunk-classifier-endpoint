import modal

app = modal.App("stonky-chunk-classifier")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.14.0",
        "transformers>=5.17.0", 
        "huggingface_hub"
    )
    .run_commands(
        "hf download MarcinZarkowski/modernbert-sec-classifier --local-dir /model"
    )
)

@app.cls(gpu="A10G", image=image)
class ClassifierAPI:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained("/model")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/model", 
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    @modal.fastapi_endpoint(method="POST")
    def score(self, data: dict):
        import torch
        chunks = data["chunks"]
        
        encoded = self.tokenizer(
            chunks, 
            padding=True, 
            truncation=True, 
            max_length=2048, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            
        predictions = []
        for row in probs:
            row_dict = {self.model.config.id2label[i]: prob for i, prob in enumerate(row)}
            predictions.append(row_dict)
            
        return {"predictions": predictions}
