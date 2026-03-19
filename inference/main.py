from fastapi import FastAPI

app = FastAPI(
    title="Inferix — Inference Cluster",
    version="0.1.0",
    description="Inference cluster stub. Serves AI model requests, deducts GFT tokens.",
)

@app.get("/")
def root():
    return {"name": "Inferix Inference Cluster", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/models")
def list_models():
    return {
        "models": [
            {"id": "llama3-8b",  "flops_per_token": 100.0, "precision": "fp16"},
            {"id": "llama3-70b", "flops_per_token": 800.0, "precision": "fp16"},
            {"id": "mistral-7b", "flops_per_token": 90.0,  "precision": "fp16"},
        ]
    }
