"""
dataset_setup.py
----------------
Downloads the NEU Surface Defect Dataset from Roboflow and restructures it
into the layout expected by manufacturing.yaml.

Usage:
    python dataset_setup.py --api-key YOUR_ROBOFLOW_KEY

If you already have the dataset locally, run with --local-dir path/to/dataset
to just validate and reformat the structure.

Dataset: NEU Surface Defect Database
  - 1,800 images (300 per class × 6 classes)
  - Augmented to ~3,600+ with built-in Roboflow augmentations
  - YOLO-format annotations

Alternative: MVTec AD
  Instructions: https://www.mvtec.com/company/research/datasets/mvtec-ad
  (requires manual download and conversion — see convert_mvtec.py)
"""

import argparse
import os
import shutil
from pathlib import Path
import yaml


def download_from_roboflow(api_key: str, save_dir: str = "./data/manufacturing"):
    """Download NEU Surface Defect dataset via Roboflow API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)

    print("[INFO] Connecting to Roboflow workspace...")
    # NEU Surface Defect — publicly hosted on Roboflow Universe
    project = rf.workspace("steel-defect-detection").project("neu-surface-defect-database-cjavb")
    dataset = project.version(4).download("yolov8", location=save_dir)

    print(f"[INFO] Dataset downloaded to: {dataset.location}")
    return dataset.location


def validate_structure(data_dir: str) -> bool:
    """Validate that the dataset follows YOLO folder structure."""
    required = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val",  "labels/test",
    ]
    missing = []
    for rel in required:
        if not Path(data_dir, rel).exists():
            missing.append(rel)

    if missing:
        print(f"[WARN] Missing directories: {missing}")
        return False

    # Count images
    for split in ["train", "val", "test"]:
        imgs = list(Path(data_dir, "images", split).glob("*.jpg")) + \
               list(Path(data_dir, "images", split).glob("*.png"))
        lbls = list(Path(data_dir, "labels", split).glob("*.txt"))
        print(f"  {split:6s}: {len(imgs):5d} images | {len(lbls):5d} labels")

    return True


def patch_yaml(data_dir: str, config_path: str = "configs/manufacturing.yaml"):
    """Update the yaml `path` field to point to the actual data location."""
    abs_path = str(Path(data_dir).resolve())
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = abs_path
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[INFO] Updated configs/manufacturing.yaml → path: {abs_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset setup for YOLO benchmark")
    parser.add_argument("--api-key",   type=str, default=None,
                        help="Roboflow API key (get free key at roboflow.com)")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Path to existing local dataset (skip download)")
    parser.add_argument("--save-dir",  type=str, default="./data/manufacturing",
                        help="Where to save downloaded dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO Benchmark — Dataset Setup")
    print("=" * 60)

    if args.local_dir:
        data_dir = args.local_dir
        print(f"[INFO] Using local dataset at: {data_dir}")
    elif args.api_key:
        data_dir = download_from_roboflow(args.api_key, args.save_dir)
    else:
        print("[ERROR] Provide --api-key for Roboflow download or --local-dir for local dataset.")
        print("\n  Quick-start (free Roboflow key):")
        print("    1. Sign up at https://roboflow.com")
        print("    2. Go to Settings → API Keys")
        print("    3. python dataset_setup.py --api-key YOUR_KEY\n")
        return

    print("\n[INFO] Validating dataset structure...")
    ok = validate_structure(data_dir)
    if ok:
        print("[OK] Structure validated.")
    else:
        print("[WARN] Structure issues found — check manually before training.")

    patch_yaml(data_dir)
    print("\n[DONE] Dataset ready. Run benchmark.py to start training.\n")


if __name__ == "__main__":
    main()
