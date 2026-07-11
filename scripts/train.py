"""
Train a single configuration (baseline / +P2 / control) under one seed.

All configurations share the same training pipeline and hyperparameters; only
the model architecture and the seed change. This corresponds to the fixed,
reproducible training settings described in Section 4.4.

Hyperparameters (fixed across all configurations):
    epochs = 100, image size = 1280, batch = 16, patience = 30 (early stopping),
    deterministic = True

Configurations:
    baseline : stock YOLOv8s (P3/P4/P5), loaded directly from 'yolov8s.pt'
    p2       : configs/yolov8s-P2P3P4P5.yaml  (candidate modification)
    control  : configs/yolov8s-P2P3P4.yaml    (supplementary)

For p2/control, the custom architecture is built from YAML and initialized with
the COCO-pretrained yolov8s weights (via .load), matching the paper's setup.

Usage:
    # baseline, seed 42
    python train.py --config baseline --seed 42 --data ./ldpolyp/data.yaml

    # +P2, seed 43
    python train.py --config p2 --seed 43 --data ./ldpolyp/data.yaml

The best checkpoint is written to runs/<config>_seed<seed>/weights/best.pt.
"""

import argparse
import os

from ultralytics import YOLO

CONFIG_YAML = {
    "p2": "configs/yolov8s-P2P3P4P5.yaml",
    "control": "configs/yolov8s-P2P3P4.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["baseline", "p2", "control"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default="./ldpolyp/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    name = f"{args.config}_seed{args.seed}"

    if args.config == "baseline":
        # stock YOLOv8s architecture (P3/P4/P5) from pretrained weights
        model = YOLO("yolov8s.pt")
    else:
        # custom architecture from YAML, initialized with COCO-pretrained weights
        cfg = CONFIG_YAML[args.config]
        if not os.path.exists(cfg):
            raise FileNotFoundError(
                f"Config not found: {cfg} (run from the repository root)."
            )
        model = YOLO(cfg)
        model.load("yolov8s.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        name=name,
        project=args.project,
        val=True,
        exist_ok=True,
    )
    print(f"Done. Best checkpoint: {args.project}/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
