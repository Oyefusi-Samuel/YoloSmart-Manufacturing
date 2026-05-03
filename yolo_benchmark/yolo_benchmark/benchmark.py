"""
benchmark.py
------------
Trains and evaluates YOLOv5s/m, YOLOv8s/m, YOLOv10s/m on the
manufacturing dataset under identical conditions.

Each model is trained 3 times with different seeds; mean ± std is reported.
Results are saved to results/benchmark_results.json.

Usage:
    python benchmark.py                          # full run (all 6 models × 3 seeds)
    python benchmark.py --models yolov8s yolov8m # specific models only
    python benchmark.py --epochs 50 --seeds 1    # fast test run
    python benchmark.py --eval-only              # skip training, re-evaluate existing weights
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ─────────────────────────── Configuration ────────────────────────────────

MODEL_WEIGHTS = {
    "yolov5s":  "yolov5su.pt",    # 'u' = ultralytics-format v5
    "yolov5m":  "yolov5mu.pt",
    "yolov8s":  "yolov8s.pt",
    "yolov8m":  "yolov8m.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
}

TRAIN_DEFAULTS = dict(
    data      = "configs/manufacturing.yaml",
    epochs    = 100,
    imgsz     = 640,
    batch     = 16,
    lr0       = 0.01,
    lrf       = 0.01,
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    hsv_h     = 0.015,      # colour augmentation
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    degrees   = 0.0,
    translate = 0.1,
    scale     = 0.5,
    flipud    = 0.0,
    fliplr    = 0.5,
    mosaic    = 1.0,
    mixup     = 0.0,
    patience  = 20,          # early stopping
    save      = True,
    exist_ok  = True,
    verbose   = False,
)

SEEDS      = [42, 123, 2024]
RESULTS_DIR = Path("results")
RUNS_DIR    = Path("runs")

# ──────────────────────────── Helpers ─────────────────────────────────────

def gpu_info() -> dict:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"gpu": name, "vram_gb": round(vram, 1), "device": 0}
    return {"gpu": "CPU only", "vram_gb": 0, "device": "cpu"}


def count_params(model: YOLO) -> int:
    return sum(p.numel() for p in model.model.parameters())


def model_size_mb(weight_path: str) -> float:
    try:
        return round(Path(weight_path).stat().st_size / 1e6, 2)
    except FileNotFoundError:
        return -1.0


def gpu_fps(model: YOLO, imgsz: int = 640, warmup: int = 300, runs: int = 1000) -> float:
    """Measure GPU throughput (FPS) with batch=1."""
    device = next(model.model.parameters()).device
    dummy  = torch.zeros(1, 3, imgsz, imgsz, device=device)
    model.model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.model(dummy)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(runs):
            model.model(dummy)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return round(runs / elapsed, 2)


def cpu_latency_ms(model: YOLO, imgsz: int = 640, warmup: int = 100, runs: int = 300) -> float:
    """Measure CPU inference latency (ms) — simulates edge deployment."""
    cpu_model = model.model.cpu().eval()
    dummy     = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for _ in range(warmup):
            cpu_model(dummy)
        t0 = time.perf_counter()
        for _ in range(runs):
            cpu_model(dummy)
    elapsed = (time.perf_counter() - t0) / runs * 1000   # ms per image
    return round(elapsed, 2)


def extract_val_metrics(val_result) -> dict:
    """Pull all accuracy metrics from a YOLO val() result."""
    box = val_result.box
    return {
        "mAP50":     round(float(box.map50), 4),
        "mAP50_95":  round(float(box.map),   4),
        "precision": round(float(box.mp),    4),
        "recall":    round(float(box.mr),    4),
        "f1":        round(2 * float(box.mp) * float(box.mr) /
                           (float(box.mp) + float(box.mr) + 1e-9), 4),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Compute mean ± std across multiple seed runs."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in runs]
        agg[k]           = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
    return agg


# ──────────────────────────── Core Loop ───────────────────────────────────

def train_and_eval(model_name: str, weights: str, seed: int,
                   epochs: int, device: int | str) -> dict:
    """Train one model with one seed and return all metrics."""
    run_name = f"{model_name}_seed{seed}"
    print(f"\n{'─'*55}")
    print(f"  Training: {model_name}  |  seed={seed}  |  epochs={epochs}")
    print(f"{'─'*55}")

    model = YOLO(weights)

    cfg = {**TRAIN_DEFAULTS,
           "epochs": epochs,
           "seed":   seed,
           "device": device,
           "name":   run_name,
           "project": str(RUNS_DIR)}

    t_start = time.time()
    model.train(**cfg)
    train_time = round((time.time() - t_start) / 60, 1)   # minutes

    # Best weights path
    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = RUNS_DIR / run_name / "weights" / "last.pt"
    best_model = YOLO(str(best_weights))

    # Accuracy
    val_result = best_model.val(
        data   = TRAIN_DEFAULTS["data"],
        imgsz  = 640,
        batch  = 1,
        device = device,
        verbose= False,
    )
    metrics = extract_val_metrics(val_result)

    # Speed (GPU)
    fps = gpu_fps(best_model)

    # Speed (CPU edge simulation)
    lat = cpu_latency_ms(best_model)

    # Model size stats
    params_m  = round(count_params(best_model) / 1e6, 2)
    size_mb   = model_size_mb(str(best_weights))

    # GFLOPs from val result
    try:
        gflops = round(float(val_result.speed.get("inference", 0)), 2)
    except Exception:
        gflops = -1.0

    return {
        **metrics,
        "fps_gpu":     fps,
        "latency_cpu_ms": lat,
        "params_M":    params_m,
        "size_MB":     size_mb,
        "gflops":      gflops,
        "train_min":   train_time,
        "best_weights": str(best_weights),
        "seed":         seed,
    }


# ──────────────────────────── Main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=list(MODEL_WEIGHTS.keys()),
                        help="Subset of models to run")
    parser.add_argument("--epochs",     type=int,  default=100)
    parser.add_argument("--seeds",      type=int,  default=3,
                        help="Number of random seeds (1–3)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; re-evaluate existing best.pt files")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    hw      = gpu_info()
    device  = hw["device"]
    seeds   = SEEDS[:args.seeds]

    print("\n" + "═"*55)
    print("  YOLO Benchmark — Smart Manufacturing")
    print("═"*55)
    print(f"  Hardware : {hw['gpu']}  ({hw['vram_gb']} GB)")
    print(f"  Models   : {args.models}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Seeds    : {seeds}")
    print("═"*55 + "\n")

    all_results = {"hardware": hw, "models": {}}

    for model_name in args.models:
        if model_name not in MODEL_WEIGHTS:
            print(f"[WARN] Unknown model '{model_name}' — skipping")
            continue

        weights = MODEL_WEIGHTS[model_name]
        seed_runs = []

        if args.eval_only:
            # Re-evaluate existing best.pt for each seed
            for seed in seeds:
                best_pt = RUNS_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
                if not best_pt.exists():
                    print(f"[WARN] {best_pt} not found — skipping seed {seed}")
                    continue
                m = YOLO(str(best_pt))
                vr = m.val(data=TRAIN_DEFAULTS["data"], imgsz=640, batch=1,
                           device=device, verbose=False)
                run = {**extract_val_metrics(vr),
                       "fps_gpu":      gpu_fps(m),
                       "latency_cpu_ms": cpu_latency_ms(m),
                       "params_M":     round(count_params(m) / 1e6, 2),
                       "size_MB":      model_size_mb(str(best_pt)),
                       "seed":         seed}
                seed_runs.append(run)
        else:
            for seed in seeds:
                run = train_and_eval(model_name, weights, seed, args.epochs, device)
                seed_runs.append(run)
                # Save checkpoint after each run in case of crash
                _tmp = RESULTS_DIR / f"{model_name}_partial.json"
                _tmp.write_text(json.dumps(seed_runs, indent=2))

        agg = aggregate_runs(seed_runs)
        all_results["models"][model_name] = {
            "aggregated": agg,
            "runs":       seed_runs,
        }

        print(f"\n[RESULT] {model_name}")
        for k in ["mAP50", "mAP50_95", "precision", "recall", "f1",
                  "fps_gpu", "latency_cpu_ms", "params_M", "size_MB"]:
            std_k = f"{k}_std"
            std   = f" ± {agg[std_k]:.4f}" if std_k in agg else ""
            print(f"  {k:<20s}: {agg[k]:.4f}{std}")

    # Final save
    out = RESULTS_DIR / "benchmark_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n[DONE] Results saved to {out}\n")


if __name__ == "__main__":
    main()
