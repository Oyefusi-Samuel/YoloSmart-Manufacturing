"""
tables.py
---------
Generates publication-ready LaTeX tables from benchmark_results.json.

Tables produced:
  T1 — Main accuracy table (mAP@50, mAP@50:95, P, R, F1)
  T2 — Speed & efficiency table (FPS, latency, params, size, GFLOPs)
  T3 — Combined summary with deployment recommendation tier

Usage:
    python tables.py
    python tables.py --results results/benchmark_results.json --out tables/
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DISPLAY = {
    "yolov5s":  r"YOLOv5s",
    "yolov5m":  r"YOLOv5m",
    "yolov8s":  r"YOLOv8s",
    "yolov8m":  r"YOLOv8m",
    "yolov10s": r"YOLOv10s",
    "yolov10m": r"YOLOv10m",
}

TIER_RULES = {
    # Rule: assign deployment tier based on mAP50 and FPS
    # Tier A: high accuracy + fast  → Production edge
    # Tier B: balanced              → Cloud/server
    # Tier C: slow/large            → Offline analysis only
}


def load_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for model_name, data in raw["models"].items():
        agg = data["aggregated"]
        rows.append({"model": model_name, **agg})
    return pd.DataFrame(rows).set_index("model")


def bold_best(series: pd.Series, fmt: str = ".4f", higher_is_better: bool = True) -> list[str]:
    """Return list of formatted strings; bold the best value."""
    best_idx = series.idxmax() if higher_is_better else series.idxmin()
    out = []
    for idx, val in series.items():
        s = f"{val:{fmt}}"
        if idx == best_idx:
            s = r"\textbf{" + s + "}"
        out.append(s)
    return out


def std_cell(df: pd.DataFrame, col: str, fmt: str = ".4f",
             higher_is_better: bool = True) -> list[str]:
    """Format as mean ± std; bold the best mean."""
    vals = df[col]
    stds = df.get(f"{col}_std", pd.Series([0.0] * len(df), index=df.index))
    best_idx = vals.idxmax() if higher_is_better else vals.idxmin()
    out = []
    for idx in df.index:
        m = vals[idx]
        s = stds[idx] if idx in stds.index else 0.0
        cell = f"{m:{fmt}}"
        if s > 0:
            cell += rf"\,$\pm$\,{s:{fmt}}"
        if idx == best_idx:
            cell = r"\textbf{" + cell + "}"
        out.append(cell)
    return out


def table_accuracy(df: pd.DataFrame) -> str:
    """Table 1 — Detection accuracy metrics."""
    header = r"""
\begin{table}[!htbp]
\centering
\caption{Detection Accuracy on NEU Surface Defect Test Set (mean $\pm$ std, 3 seeds)}
\label{tab:accuracy}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l c c c c c}
\toprule
\textbf{Model} & \textbf{mAP@50} & \textbf{mAP@50:95} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule"""

    rows = []
    map50   = std_cell(df, "mAP50",     higher_is_better=True)
    map5095 = std_cell(df, "mAP50_95",  higher_is_better=True)
    prec    = std_cell(df, "precision", higher_is_better=True)
    rec     = std_cell(df, "recall",    higher_is_better=True)
    f1      = std_cell(df, "f1",        higher_is_better=True)

    for i, model_name in enumerate(df.index):
        rows.append(
            f"{DISPLAY[model_name]} & {map50[i]} & {map5095[i]} & "
            f"{prec[i]} & {rec[i]} & {f1[i]} \\\\"
        )

    footer = r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize Bold = best per column. Results averaged over 3 random seeds.} \\
\end{tabular}%
}
\end{table}"""

    return header + "\n" + "\n".join(rows) + footer


def table_speed(df: pd.DataFrame) -> str:
    """Table 2 — Speed and efficiency metrics."""
    header = r"""
\begin{table}[!htbp]
\centering
\caption{Inference Speed and Model Efficiency}
\label{tab:speed}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l c c c c c}
\toprule
\textbf{Model} & \textbf{GPU FPS$\uparrow$} & \textbf{CPU Latency (ms)$\downarrow$} & \textbf{Params (M)$\downarrow$} & \textbf{Size (MB)$\downarrow$} & \textbf{GFLOPs$\downarrow$} \\
\midrule"""

    fps  = std_cell(df, "fps_gpu",        fmt=".1f",  higher_is_better=True)
    lat  = std_cell(df, "latency_cpu_ms", fmt=".1f",  higher_is_better=False)
    par  = bold_best(df["params_M"],      fmt=".2f",  higher_is_better=False)
    sz   = bold_best(df["size_MB"],       fmt=".1f",  higher_is_better=False)
    gfl  = bold_best(df.get("gflops", pd.Series([-1.0]*len(df), index=df.index)),
                     fmt=".1f", higher_is_better=False)

    rows = []
    for i, model_name in enumerate(df.index):
        rows.append(
            f"{DISPLAY[model_name]} & {fps[i]} & {lat[i]} & "
            f"{par[i]} & {sz[i]} & {gfl[i]} \\\\"
        )

    footer = r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize GPU: NVIDIA [insert GPU]. CPU: Intel [insert CPU] (edge simulation, batch=1).} \\
\end{tabular}%
}
\end{table}"""

    return header + "\n" + "\n".join(rows) + footer


def table_summary(df: pd.DataFrame) -> str:
    """Table 3 — Deployment recommendation summary."""

    def tier(row) -> str:
        fps = row.get("fps_gpu", 0)
        map50 = row.get("mAP50", 0)
        if fps > 150 and map50 > 0.85:
            return r"\textbf{A} (Edge-Optimal)"
        elif fps > 80 and map50 > 0.80:
            return "B (Balanced)"
        elif map50 > 0.85:
            return "B+ (Accuracy-First)"
        else:
            return "C (Development)"

    header = r"""
\begin{table}[!htbp]
\centering
\caption{Deployment Recommendation Framework for Smart Manufacturing}
\label{tab:deployment}
\begin{tabular}{l c c l}
\toprule
\textbf{Model} & \textbf{mAP@50} & \textbf{GPU FPS} & \textbf{Deployment Tier} \\
\midrule"""

    rows = []
    for model_name, row in df.iterrows():
        rows.append(
            f"{DISPLAY[model_name]} & "
            f"{row.get('mAP50', 0):.4f} & "
            f"{row.get('fps_gpu', 0):.1f} & "
            f"{tier(row)} \\\\"
        )

    footer = r"""
\bottomrule
\multicolumn{4}{l}{\footnotesize Tier A: $\geq$150 FPS + mAP@50 $\geq$0.85 — recommended for real-time edge deployment.} \\
\multicolumn{4}{l}{\footnotesize Tier B: balanced speed-accuracy for server-side inference.} \\
\end{tabular}
\end{table}"""

    return header + "\n" + "\n".join(rows) + footer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/benchmark_results.json")
    parser.add_argument("--out",     default="tables")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)

    if not Path(args.results).exists():
        print(f"[ERROR] {args.results} not found. Run benchmark.py first.")
        return

    df = load_df(args.results)

    t1 = table_accuracy(df)
    t2 = table_speed(df)
    t3 = table_summary(df)

    combined = (
        "% ─────────────────────────────────────────────\n"
        "% Auto-generated by tables.py — paste into LaTeX\n"
        "% Requires: \\usepackage{booktabs,multirow,resizebox}\n"
        "% ─────────────────────────────────────────────\n\n"
        + t1 + "\n\n" + t2 + "\n\n" + t3
    )

    out = Path(args.out) / "all_tables.tex"
    out.write_text(combined)
    print(f"[DONE] LaTeX tables saved to {out}")

    # Also save individually
    (Path(args.out) / "table_accuracy.tex").write_text(t1)
    (Path(args.out) / "table_speed.tex").write_text(t2)
    (Path(args.out) / "table_deployment.tex").write_text(t3)
    print("       Individual tables also saved.")


if __name__ == "__main__":
    main()
