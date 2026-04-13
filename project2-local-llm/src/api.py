import json
from fastapi import FastAPI
from pydantic import BaseModel
from ollama_client import generate

app = FastAPI(title="Local LLM Bench API")

class GenerateRequest(BaseModel):
    model: str = "llama3.2:3b"
    prompt: str
    temperature: float = 0.7
    require_json: bool = False

class GenerateResponse(BaseModel):
    model: str
    response: str
    tokens_per_second: float
    duration_ms: float
    ttft_ms: float
    valid_json: bool = False

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    result = generate(req.model, req.prompt, req.temperature)
    valid = False
    if req.require_json:
        try:
            json.loads(result.response)
            valid = True
        except Exception:
            valid = False
    return GenerateResponse(
        model=result.model,
        response=result.response,
        tokens_per_second=result.tokens_per_second,
        duration_ms=result.duration_ms,
        ttft_ms=result.ttft_ms,
        valid_json=valid
    )

@app.get("/health")
def health():
    return {"status": "ok"}
