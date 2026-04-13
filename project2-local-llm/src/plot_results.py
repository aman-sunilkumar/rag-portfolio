import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison():
    rows = []
    for f in Path("../data/results").glob("*.json"):
        if "temperature" in f.name:
            continue
        rows.extend(json.loads(f.read_text()))
    df = pd.DataFrame(rows)

    # shorten model names for display
    df["model_short"] = df["model"].apply(lambda x: x.split(":")[0])

    models = df["model_short"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model comparison — Apple M1 8GB", fontsize=13)

    tps   = [df[df["model_short"]==m]["tokens_per_second"].values for m in models]
    ttft  = [df[df["model_short"]==m]["ttft_ms"].values for m in models]
    dur   = [df[df["model_short"]==m]["duration_ms"].values for m in models]

    axes[0].boxplot(tps, tick_labels=models)
    axes[0].set_title("Tokens per second")
    axes[0].set_ylabel("tok/s")

    axes[1].boxplot(ttft, tick_labels=models)
    axes[1].set_title("Time to first token (ms)")
    axes[1].set_ylabel("ms")

    axes[2].boxplot(dur, tick_labels=models)
    axes[2].set_title("Total response time (ms)")
    axes[2].set_ylabel("ms")

    plt.tight_layout()
    out = Path("../reports/model_comparison.png")
    plt.savefig(out, dpi=150)
    console_print = print
    console_print(f"Saved → {out}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
