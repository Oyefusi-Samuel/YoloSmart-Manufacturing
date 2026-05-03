"""
visualize.py
------------
Generates all figures needed for the IEEE paper from benchmark_results.json.

Outputs (saved to plots/):
  1. bar_accuracy.pdf      — mAP@50 and mAP@50:95 grouped bar chart
  2. bar_speed.pdf         — GPU FPS and CPU latency side-by-side
  3. bar_efficiency.pdf    — Parameter count vs model size
  4. radar_tradeoff.pdf    — Radar chart: accuracy / speed / efficiency
  5. scatter_tradeoff.pdf  — mAP50 vs FPS scatter (bubble = model size)
  6. pr_curves.pdf         — Precision-Recall curves per model (if CSVs exist)
  7. confusion_matrices/   — Per-model confusion matrix images

Usage:
    python visualize.py                          # uses results/benchmark_results.json
    python visualize.py --results path/to/file.json
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.format":  "pdf",
})

# Colour palette — one per architecture family
PALETTE = {
    "yolov5s": "#2196F3",   # blue family
    "yolov5m": "#0D47A1",
    "yolov8s": "#4CAF50",   # green family
    "yolov8m": "#1B5E20",
    "yolov10s": "#FF9800",  # orange family
    "yolov10m": "#E65100",
}

DISPLAY = {
    "yolov5s":  "YOLOv5s",
    "yolov5m":  "YOLOv5m",
    "yolov8s":  "YOLOv8s",
    "yolov8m":  "YOLOv8m",
    "yolov10s": "YOLOv10s",
    "yolov10m": "YOLOv10m",
}

PLOTS_DIR = Path("plots")


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for model_name, data in raw["models"].items():
        agg = data["aggregated"]
        row = {"model": model_name, **agg}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def bar_accuracy(df: pd.DataFrame):
    """Grouped bar: mAP@50 and mAP@50:95 per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    models = list(df.index)
    x      = np.arange(len(models))
    width  = 0.35

    bars1 = ax.bar(x - width/2, df["mAP50"],    width,
                   color=[PALETTE[m] for m in models],
                   label="mAP@50", alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, df["mAP50_95"], width,
                   color=[PALETTE[m] for m in models],
                   label="mAP@50:95", alpha=0.55, edgecolor="white",
                   linewidth=0.5, hatch="//")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    # Std error bars if available
    if "mAP50_std" in df.columns:
        ax.errorbar(x - width/2, df["mAP50"],    yerr=df["mAP50_std"],
                    fmt="none", color="black", capsize=3, linewidth=1)
        ax.errorbar(x + width/2, df["mAP50_95"], yerr=df["mAP50_95_std"],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, df["mAP50"].max() * 1.20))
    ax.set_title("Detection Accuracy: mAP@50 and mAP@50:95")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = PLOTS_DIR / "bar_accuracy.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def bar_speed(df: pd.DataFrame):
    """Dual y-axis: GPU FPS (left) and CPU latency ms (right)."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    models = list(df.index)
    x      = np.arange(len(models))
    width  = 0.35

    ax1.bar(x - width/2, df["fps_gpu"], width,
            color=[PALETTE[m] for m in models],
            label="GPU FPS", alpha=0.9, edgecolor="white")
    ax2.bar(x + width/2, df["latency_cpu_ms"], width,
            color=[PALETTE[m] for m in models],
            label="CPU Latency (ms)", alpha=0.55, edgecolor="white", hatch="\\\\")

    ax1.set_xticks(x)
    ax1.set_xticklabels([DISPLAY[m] for m in models])
    ax1.set_ylabel("GPU Throughput (FPS)  ↑ better")
    ax2.set_ylabel("CPU Inference Latency (ms)  ↓ better")
    ax1.set_title("Inference Speed: GPU Throughput vs. CPU Latency")

    lines1 = mpatches.Patch(color="#555", alpha=0.9, label="GPU FPS")
    lines2 = mpatches.Patch(color="#555", alpha=0.55, hatch="\\\\", label="CPU Latency (ms)")
    ax1.legend(handles=[lines1, lines2], loc="upper right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    out = PLOTS_DIR / "bar_speed.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def bar_efficiency(df: pd.DataFrame):
    """Grouped bar: parameters (M) and model size (MB)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    models = list(df.index)
    x      = np.arange(len(models))
    width  = 0.35

    ax.bar(x - width/2, df["params_M"],  width,
           color=[PALETTE[m] for m in models],
           label="Parameters (M)", alpha=0.9, edgecolor="white")
    ax.bar(x + width/2, df["size_MB"],   width,
           color=[PALETTE[m] for m in models],
           label="Model Size (MB)", alpha=0.55, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("Count / Size  ↓ better for edge")
    ax.set_title("Model Efficiency: Parameters vs. File Size")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = PLOTS_DIR / "bar_efficiency.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def radar_tradeoff(df: pd.DataFrame):
    """Radar chart: normalised accuracy / speed / efficiency per model."""
    categories = ["mAP@50", "mAP@50:95", "GPU FPS", "Edge Speed\n(1/Latency)", "Efficiency\n(1/Params)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    def normalise(series):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return series * 0 + 0.5
        return (series - lo) / (hi - lo)

    norm = pd.DataFrame({
        "mAP@50":          normalise(df["mAP50"]),
        "mAP@50:95":       normalise(df["mAP50_95"]),
        "GPU FPS":         normalise(df["fps_gpu"]),
        "Edge Speed":      normalise(1 / df["latency_cpu_ms"]),
        "Efficiency":      normalise(1 / df["params_M"]),
    })

    for model_name in df.index:
        values = norm.loc[model_name].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5,
                color=PALETTE[model_name], label=DISPLAY[model_name])
        ax.fill(angles, values, alpha=0.08, color=PALETTE[model_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=7)
    ax.set_title("Multi-Dimensional Trade-off\n(normalised per metric)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    fig.tight_layout()
    out = PLOTS_DIR / "radar_tradeoff.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def scatter_tradeoff(df: pd.DataFrame):
    """Bubble scatter: mAP50 vs FPS, bubble size = model size MB."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, row in df.iterrows():
        size  = row.get("size_MB", 20) * 3   # scale bubble
        ax.scatter(row["fps_gpu"], row["mAP50"],
                   s=size, color=PALETTE[model_name],
                   alpha=0.8, edgecolors="black", linewidths=0.6,
                   zorder=3)
        ax.annotate(DISPLAY[model_name],
                    (row["fps_gpu"], row["mAP50"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5)

    ax.set_xlabel("GPU Throughput (FPS)  →  faster")
    ax.set_ylabel("mAP@50  →  more accurate")
    ax.set_title("Speed–Accuracy Trade-off\n(bubble area ∝ model file size)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Bubble legend
    for sz_mb in [10, 30, 60]:
        ax.scatter([], [], s=sz_mb*3, color="grey", alpha=0.5,
                   label=f"{sz_mb} MB", edgecolors="black", linewidths=0.5)
    ax.legend(title="Model size", loc="lower right")

    fig.tight_layout()
    out = PLOTS_DIR / "scatter_tradeoff.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def precision_recall_bar(df: pd.DataFrame):
    """Bar chart for Precision, Recall, F1 grouped by model."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models  = list(df.index)
    x       = np.arange(len(models))
    width   = 0.25

    ax.bar(x - width, df["precision"], width,
           color=[PALETTE[m] for m in models],
           label="Precision", alpha=0.9)
    ax.bar(x,          df["recall"],   width,
           color=[PALETTE[m] for m in models],
           label="Recall",    alpha=0.65, hatch="//")
    ax.bar(x + width,  df["f1"],       width,
           color=[PALETTE[m] for m in models],
           label="F1-Score",  alpha=0.45, hatch="\\\\")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision, Recall, and F1-Score per Model")

    lines = [
        mpatches.Patch(color="#555", alpha=0.9,  label="Precision"),
        mpatches.Patch(color="#555", alpha=0.65, hatch="//",  label="Recall"),
        mpatches.Patch(color="#555", alpha=0.45, hatch="\\\\", label="F1-Score"),
    ]
    ax.legend(handles=lines, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = PLOTS_DIR / "bar_prf.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def training_convergence(runs_dir: Path = Path("runs")):
    """Plot training mAP50 curves from Ultralytics results.csv files."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    found   = False

    for model_name, colour in PALETTE.items():
        csv_path = runs_dir / f"{model_name}_seed42" / "results.csv"
        if not csv_path.exists():
            continue
        df_run = pd.read_csv(csv_path)
        df_run.columns = df_run.columns.str.strip()
        # Ultralytics column name variations
        col = next((c for c in df_run.columns if "mAP50" in c and "95" not in c), None)
        if col is None:
            continue
        ax.plot(df_run.index + 1, df_run[col],
                color=colour, label=DISPLAY[model_name], linewidth=1.4)
        found = True

    if not found:
        print("  [SKIP] training_convergence — no results.csv files found in runs/")
        plt.close()
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP@50")
    ax.set_title("Training Convergence (seed=42)")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = PLOTS_DIR / "convergence.pdf"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="results/benchmark_results.json")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)

    if not Path(args.results).exists():
        print(f"[ERROR] {args.results} not found. Run benchmark.py first.")
        return

    print("[INFO] Loading results...")
    df = load_results(args.results)
    print(df[["mAP50", "mAP50_95", "fps_gpu", "latency_cpu_ms", "params_M"]].to_string())
    print()

    print("[INFO] Generating plots...")
    bar_accuracy(df)
    bar_speed(df)
    bar_efficiency(df)
    radar_tradeoff(df)
    scatter_tradeoff(df)
    precision_recall_bar(df)
    training_convergence(Path(args.runs_dir))

    print(f"\n[DONE] All plots saved to {PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()
