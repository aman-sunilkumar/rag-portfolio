import json
from pydantic import BaseModel
from rich.console import Console
from ollama_client import generate

console = Console()

class ExtractedJob(BaseModel):
    company: str | None = None
    job_title: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    currency: str = "USD"

SYSTEM_PROMPT = """You are an information extraction assistant.
Always respond with ONLY valid JSON matching this exact schema:
{
  "company": "<string or null>",
  "job_title": "<string or null>",
  "salary_min": <integer or null>,
  "salary_max": <integer or null>,
  "currency": "<string, default USD>"
}
No explanation. No markdown. No backticks. Just the raw JSON object."""

def extract_with_retry(model: str, text: str) -> ExtractedJob | None:
    prompt = f"{SYSTEM_PROMPT}\n\nExtract from this text:\n{text}"

    for attempt in range(2):
        result = generate(model, prompt, temperature=0.0)
        raw = result.response.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
            validated = ExtractedJob(**data)
            console.print(f"  [green]✓ Valid on attempt {attempt + 1}[/green]")
            return validated
        except Exception as e:
            console.print(f"  [yellow]⚠ Attempt {attempt + 1} failed: {e}[/yellow]")
            if attempt == 0:
                prompt = f"{SYSTEM_PROMPT}\n\nExtract from this text:\n{text}\n\nYour last response was invalid JSON: {e}\nReturn ONLY the raw JSON object."

    console.print("  [red]✗ Both attempts failed — returning None[/red]")
    return None

if __name__ == "__main__":
    test_cases = [
        "We're hiring a Senior ML Engineer at DeepMind, salary $180,000-$220,000.",
        "Join Acme Corp as a Data Analyst. Compensation: $70k-$90k annually.",
        "The weather today is sunny with a high of 75 degrees.",
        "OpenAI is looking for a Research Scientist. Compensation not disclosed.",
    ]

    for text in test_cases:
        console.print(f"\n[bold]Input:[/bold] {text[:70]}")
        result = extract_with_retry("llama3.2:3b", text)
        if result:
            console.print(result.model_dump())
        else:
            console.print("[red]No result[/red]")
