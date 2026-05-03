# YOLO Benchmark — Smart Manufacturing
## IEEE Paper: *Benchmarking YOLOv5, YOLOv8, and YOLOv10 for Real-Time Object Detection in Smart Manufacturing Environments*

---

## Project Structure

```
yolo_benchmark/
├── configs/
│   └── manufacturing.yaml     ← Dataset config (auto-updated by dataset_setup.py)
├── data/
│   └── manufacturing/         ← Downloaded dataset goes here
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── results/
│   └── benchmark_results.json ← All metrics (output of benchmark.py)
├── runs/                      ← Ultralytics training outputs (auto-created)
├── plots/                     ← All PDF figures (output of visualize.py)
├── tables/                    ← LaTeX table files (output of tables.py)
│
├── dataset_setup.py           ← Download + validate NEU Surface Defect dataset
├── benchmark.py               ← Train & evaluate all 6 models (3 seeds each)
├── visualize.py               ← Generate all paper figures
├── tables.py                  ← Generate IEEE LaTeX tables
├── mock_results.py            ← Fake results for testing plots/tables
├── run_pipeline.sh            ← One-shot orchestration script
└── requirements.txt
```

---

## Quickstart

### Test plots/tables immediately (no GPU needed)
```bash
python mock_results.py     # generate results
python visualize.py        # produce all plots → plots/
python tables.py           # produce LaTeX tables → tables/
```

### Full run with real training
```bash
# 1. Get a free API key at roboflow.com → Settings → API Keys
ROBOFLOW_KEY=your_key_here bash run_pipeline.sh

# Or with local dataset:
python dataset_setup.py --local-dir /path/to/your/dataset
python benchmark.py --epochs 100 --seeds 3
python visualize.py
python tables.py
```

### Individual commands
```bash
# Train just YOLOv8 variants, 1 seed, 50 epochs
python benchmark.py --models yolov8s yolov8m --epochs 50 --seeds 1

# Re-evaluate without re-training
python benchmark.py --eval-only

# Fast test run (mock results + full plots)
bash run_pipeline.sh --fast
```

---

## Dataset: NEU Surface Defect Database

| Class | Description |
|---|---|
| crazing | Network of fine cracks |
| inclusion | Metallic/oxide inclusions |
| patches | Irregular surface patches |
| pitted_surface | Pits and voids |
| rolled_in_scale | Rolled-in oxide scales |
| scratches | Linear scratches |

- **Source**: Northeastern University (NEU), freely available via Roboflow Universe
- **Split**: 70% train / 15% val / 15% test
- **Augmentations**: horizontal flip, mosaic, brightness jitter, random crop

---

## Models Compared

| Model | Variant | Params | Key Feature |
|---|---|---|---|
| YOLOv5 | s, m | 7M, 21M | Anchor-based, mature ecosystem |
| YOLOv8 | s, m | 11M, 25M | Anchor-free, C2f blocks |
| YOLOv10 | s, m | 8M, 16M | NMS-free, dual-head |

All trained with **identical** hyperparameters for fair comparison.

---

## Figures Generated

| File | Description |
|---|---|
| `bar_accuracy.pdf` | mAP@50 and mAP@50:95 grouped bar chart |
| `bar_speed.pdf` | GPU FPS vs CPU latency |
| `bar_efficiency.pdf` | Parameters and model size |
| `radar_tradeoff.pdf` | Multi-dimensional radar chart |
| `scatter_tradeoff.pdf` | mAP50 vs FPS bubble plot |
| `bar_prf.pdf` | Precision, Recall, F1 comparison |
| `convergence.pdf` | Training convergence curves |

---

## LaTeX Usage

```latex
% In your IEEE paper preamble:
\usepackage{booktabs}
\usepackage{graphicx}  % for \resizebox

% Include tables:
\input{tables/table_accuracy.tex}
\input{tables/table_speed.tex}
\input{tables/table_deployment.tex}

% Include figures:
\includegraphics[width=\columnwidth]{plots/bar_accuracy.pdf}
```

---

## Expected Runtime

| Setup | Time |
|---|---|
| RTX 3090, 6 models × 3 seeds × 100 epochs | ~6–8 hours |
| RTX 3090, fast test (50 epochs, 1 seed) | ~1.5 hours |
| Mock results + all plots/tables | < 30 seconds |
