import json, pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def load_all_results() -> pd.DataFrame:
    rows = []
    for f in Path("../data/results").glob("*.json"):
        if "temperature_experiment" in f.name:
            continue
        data = json.loads(f.read_text())
        rows.extend(data)
    return pd.DataFrame(rows)

def print_model_summary(df: pd.DataFrame):
    summary = df.groupby("model").agg(
        avg_tps=("tokens_per_second", "mean"),
        p50_tps=("tokens_per_second", lambda x: x.quantile(0.5)),
        p95_ttft=("ttft_ms", lambda x: x.quantile(0.95)),
        avg_ttft=("ttft_ms", "mean"),
        avg_duration=("duration_ms", "mean"),
        n_prompts=("prompt_id", "count")
    ).round(1).reset_index()

    table = Table(title="Model comparison — all prompts")
    for col in summary.columns:
        table.add_column(col, justify="right" if col != "model" else "left")
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)
    return summary

def print_category_breakdown(df: pd.DataFrame):
    cat = df.groupby(["model", "category"]).agg(
        avg_tps=("tokens_per_second", "mean"),
        avg_duration=("duration_ms", "mean")
    ).round(1).reset_index()

    table = Table(title="Performance by category")
    for col in cat.columns:
        table.add_column(col, justify="right" if col not in ("model", "category") else "left")
    for _, row in cat.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)

if __name__ == "__main__":
    df = load_all_results()
    console.print(f"\nLoaded [bold]{len(df)}[/bold] results across [bold]{df['model'].nunique()}[/bold] models\n")
    summary = print_model_summary(df)
    print_category_breakdown(df)
    Path("../reports").mkdir(exist_ok=True)
    summary.to_csv("../reports/model_comparison.csv", index=False)
    console.print("[dim]Saved → reports/model_comparison.csv[/dim]")
