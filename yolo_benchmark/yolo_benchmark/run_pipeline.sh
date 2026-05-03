#!/usr/bin/env bash
# run_pipeline.sh
# ──────────────────────────────────────────────────────
# Full benchmark pipeline: dataset → train → plot → tables
#
# Usage:
#   bash run_pipeline.sh                    # full run, 3 seeds × 6 models × 100 epochs
#   bash run_pipeline.sh --fast             # 1 seed × 50 epochs (quick test)
#   bash run_pipeline.sh --eval-only        # skip training, re-run eval + plots
#   ROBOFLOW_KEY=xxx bash run_pipeline.sh   # auto-download dataset
# ──────────────────────────────────────────────────────
set -euo pipefail

EPOCHS=100
SEEDS=3
EVAL_ONLY=0
FAST=0

for arg in "$@"; do
  case $arg in
    --fast)       EPOCHS=50; SEEDS=1; FAST=1 ;;
    --eval-only)  EVAL_ONLY=1 ;;
  esac
done

echo "========================================================"
echo "  YOLO Benchmark Pipeline — Smart Manufacturing"
echo "  Epochs: $EPOCHS | Seeds: $SEEDS"
echo "========================================================"

# ── Step 0: Install dependencies ──────────────────────
echo ""
echo "[STEP 0] Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages -q

# ── Step 1: Dataset setup ─────────────────────────────
if [ "$EVAL_ONLY" -eq 0 ] && [ "$FAST" -eq 0 ]; then
  echo ""
  echo "[STEP 1] Setting up dataset..."
  if [ -n "${ROBOFLOW_KEY:-}" ]; then
    python dataset_setup.py --api-key "$ROBOFLOW_KEY"
  elif [ -d "data/manufacturing/images/train" ]; then
    echo "  Dataset already present at data/manufacturing/"
    python dataset_setup.py --local-dir data/manufacturing
  else
    echo ""
    echo "  ⚠️  No dataset found and no ROBOFLOW_KEY set."
    echo "  Options:"
    echo "    1. ROBOFLOW_KEY=your_key bash run_pipeline.sh"
    echo "    2. python dataset_setup.py --local-dir /path/to/your/dataset"
    echo "    3. bash run_pipeline.sh --fast  (runs with mock data, no training)"
    echo ""
    exit 1
  fi
fi

# ── Step 2: Train & benchmark ─────────────────────────
if [ "$EVAL_ONLY" -eq 0 ] && [ "$FAST" -eq 0 ]; then
  echo ""
  echo "[STEP 2] Running benchmark (this will take several hours on GPU)..."
  python benchmark.py --epochs "$EPOCHS" --seeds "$SEEDS"
elif [ "$EVAL_ONLY" -eq 1 ]; then
  echo ""
  echo "[STEP 2] Re-evaluating existing weights..."
  python benchmark.py --eval-only
elif [ "$FAST" -eq 1 ]; then
  echo ""
  echo "[STEP 2] Fast mode: generating mock results (no training)..."
  python mock_results.py
fi

# ── Step 3: Visualisations ───────────────────────────
echo ""
echo "[STEP 3] Generating plots..."
python visualize.py

# ── Step 4: LaTeX tables ─────────────────────────────
echo ""
echo "[STEP 4] Generating LaTeX tables..."
python tables.py

# ── Done ─────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Pipeline complete!"
echo "  Plots  → plots/"
echo "  Tables → tables/all_tables.tex"
echo "  Data   → results/benchmark_results.json"
echo "========================================================"
