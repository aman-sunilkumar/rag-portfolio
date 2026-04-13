import httpx, time, json
from dataclasses import dataclass

OLLAMA_BASE = "http://localhost:11434"

@dataclass
class InferenceResult:
    model: str
    prompt: str
    response: str
    tokens_generated: int
    duration_ms: float
    ttft_ms: float
    tokens_per_second: float

def generate(model: str, prompt: str, temperature: float = 0.7) -> InferenceResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature}
    }

    start = time.perf_counter()
    first_token_time = None
    full_response = ""

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                full_response += chunk.get("response", "")

                if chunk.get("done"):
                    total_ms = (time.perf_counter() - start) * 1000
                    ttft_ms = (first_token_time - start) * 1000
                    toks = chunk.get("eval_count", 0)
                    tps = toks / (total_ms / 1000) if total_ms > 0 else 0

                    return InferenceResult(
                        model=model,
                        prompt=prompt,
                        response=full_response,
                        tokens_generated=toks,
                        duration_ms=round(total_ms, 1),
                        ttft_ms=round(ttft_ms, 1),
                        tokens_per_second=round(tps, 1)
                    )
