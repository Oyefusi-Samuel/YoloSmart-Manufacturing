"""
mock_results.py
---------------
Generates a realistic benchmark_results.json with plausible values so you
can test visualize.py and tables.py immediately — before GPU training is done.

Values are based on published Ultralytics benchmarks and NEU defect dataset
literature; adjust them once your real results are in.

Usage:
    python mock_results.py
"""

import json
import random
from pathlib import Path

random.seed(42)

# Realistic base values from literature + Ultralytics benchmarks
# (NEU Surface Defect, 640x640, RTX 3090 GPU)
BASE = {
    "yolov5s": dict(
        mAP50=0.872, mAP50_95=0.621, precision=0.884, recall=0.851,
        fps_gpu=312.4, latency_cpu_ms=18.6,
        params_M=7.05, size_MB=14.1, gflops=16.5, train_min=38.2,
    ),
    "yolov5m": dict(
        mAP50=0.903, mAP50_95=0.658, precision=0.911, recall=0.882,
        fps_gpu=198.7, latency_cpu_ms=32.4,
        params_M=21.17, size_MB=42.3, gflops=49.0, train_min=61.4,
    ),
    "yolov8s": dict(
        mAP50=0.891, mAP50_95=0.641, precision=0.896, recall=0.869,
        fps_gpu=287.3, latency_cpu_ms=21.2,
        params_M=11.17, size_MB=22.4, gflops=28.6, train_min=44.7,
    ),
    "yolov8m": dict(
        mAP50=0.921, mAP50_95=0.681, precision=0.926, recall=0.901,
        fps_gpu=176.4, latency_cpu_ms=41.8,
        params_M=25.90, size_MB=51.8, gflops=78.9, train_min=74.2,
    ),
    "yolov10s": dict(
        mAP50=0.897, mAP50_95=0.649, precision=0.903, recall=0.874,
        fps_gpu=295.8, latency_cpu_ms=19.8,
        params_M=8.06, size_MB=16.1, gflops=24.5, train_min=41.3,
    ),
    "yolov10m": dict(
        mAP50=0.918, mAP50_95=0.676, precision=0.922, recall=0.897,
        fps_gpu=183.2, latency_cpu_ms=38.6,
        params_M=16.49, size_MB=32.9, gflops=63.3, train_min=68.9,
    ),
}

SEEDS = [42, 123, 2024]
NOISE = 0.006   # ±0.6% variance across seeds


def jitter(val: float, noise: float = NOISE) -> float:
    return round(val + random.uniform(-noise, noise), 4)


def make_run(base: dict, seed: int) -> dict:
    random.seed(seed)
    return {
        "mAP50":          jitter(base["mAP50"]),
        "mAP50_95":       jitter(base["mAP50_95"]),
        "precision":      jitter(base["precision"]),
        "recall":         jitter(base["recall"]),
        "f1":             round(2 * jitter(base["precision"]) * jitter(base["recall"]) /
                                (jitter(base["precision"]) + jitter(base["recall"]) + 1e-9), 4),
        "fps_gpu":        round(base["fps_gpu"] + random.uniform(-5, 5), 2),
        "latency_cpu_ms": round(base["latency_cpu_ms"] + random.uniform(-0.5, 0.5), 2),
        "params_M":       base["params_M"],
        "size_MB":        base["size_MB"],
        "gflops":         base["gflops"],
        "train_min":      round(base["train_min"] + random.uniform(-2, 2), 1),
        "seed":           seed,
        "best_weights":   f"runs/{list(BASE.keys())[0]}_seed{seed}/weights/best.pt",
    }


def aggregate(runs: list[dict]) -> dict:
    import statistics
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and k != "seed"]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in runs]
        agg[k]          = round(statistics.mean(vals), 4)
        agg[f"{k}_std"] = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
    return agg


def main():
    out = {
        "hardware": {
            "gpu":      "NVIDIA RTX 3090 (mock)",
            "vram_gb":  24.0,
            "device":   0,
        },
        "models": {}
    }

    for model_name, base in BASE.items():
        runs = [make_run(base, s) for s in SEEDS]
        out["models"][model_name] = {
            "aggregated": aggregate(runs),
            "runs":       runs,
        }
        print(f"  {model_name:10s} → mAP50={out['models'][model_name]['aggregated']['mAP50']:.4f}  "
              f"FPS={out['models'][model_name]['aggregated']['fps_gpu']:.1f}")

    Path("results").mkdir(exist_ok=True)
    path = Path("results/benchmark_results.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[DONE] Mock results written to {path}")
    print("       Run: python visualize.py && python tables.py\n")


if __name__ == "__main__":
    main()
