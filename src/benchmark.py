import json, yaml, time, sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from ollama_client import generate

console = Console()

def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["prompts"]

def run_benchmark(model: str, prompts: list[dict], temperature: float = 0.7) -> list[dict]:
    results = []
    console.print(f"\n[bold]Running:[/bold] {model}  [dim]temp={temperature}[/dim]")

    for p in prompts:
        console.print(f"  {p['id']}...", end=" ")
        try:
            r = generate(model, p["text"], temperature=temperature)
            result = {
                "model": model,
                "prompt_id": p["id"],
                "category": p["category"],
                "temperature": temperature,
                "tokens_generated": r.tokens_generated,
                "duration_ms": r.duration_ms,
                "ttft_ms": r.ttft_ms,
                "tokens_per_second": r.tokens_per_second,
                "response_preview": r.response[:100]
            }
            results.append(result)
            console.print(f"[green]{r.tokens_per_second} tok/s  {r.ttft_ms}ms TTFT[/green]")
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")

    return results

def save_results(results: list[dict], model: str, temperature: float) -> Path:
    ts = int(time.time())
    tag = model.replace(":", "_").replace(".", "_")
    out = Path(f"../data/results/{tag}_temp{temperature}_{ts}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    console.print(f"[dim]Saved → {out}[/dim]")
    return out

def print_summary(results: list[dict]):
    table = Table(title="Summary")
    table.add_column("Prompt")
    table.add_column("Category")
    table.add_column("Tok/s", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("Total ms", justify="right")
    for r in results:
        table.add_row(
            r["prompt_id"], r["category"],
            str(r["tokens_per_second"]),
            str(r["ttft_ms"]),
            str(r["duration_ms"])
        )
    console.print(table)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
    prompts = load_prompts("../data/prompts/benchmark_prompts.yaml")
    results = run_benchmark(model, prompts)
    save_results(results, model, 0.7)
    print_summary(results)
