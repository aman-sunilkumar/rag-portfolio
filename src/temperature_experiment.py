import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from ollama_client import generate

console = Console()

PROMPTS = [
    "What is the capital of Australia?",
    "Name three programming languages used for data engineering.",
    "What does API stand for?",
    "List two advantages of using vector databases.",
    "What is the difference between supervised and unsupervised learning?",
]

def run_experiment(model: str = "llama3.2:3b", n_runs: int = 3):
    results = {0.0: {}, 0.7: {}}

    for temp in [0.0, 0.7]:
        console.print(f"\n[bold]Temperature {temp}[/bold]")
        for prompt in PROMPTS:
            runs = []
            for i in range(n_runs):
                r = generate(model, prompt, temperature=temp)
                runs.append(r.response.strip())
                console.print(f"  run {i+1}: {r.response.strip()[:60]}")
            results[temp][prompt] = runs

    out = Path("../data/results/temperature_experiment.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\n[dim]Saved → {out}[/dim]")

    table = Table(title="Temperature variance — same prompt, 3 runs each")
    table.add_column("Prompt", max_width=35)
    table.add_column("Temp 0.0 — consistent?")
    table.add_column("Temp 0.7 — consistent?")

    for prompt in PROMPTS:
        r0 = results[0.0][prompt]
        r7 = results[0.7][prompt]
        all_same_0 = "✓ identical" if len(set(r0)) == 1 else f"✗ {len(set(r0))} variants"
        all_same_7 = "✓ identical" if len(set(r7)) == 1 else f"✗ {len(set(r7))} variants"
        table.add_row(prompt[:35], all_same_0, all_same_7)

    console.print(table)

if __name__ == "__main__":
    run_experiment()
