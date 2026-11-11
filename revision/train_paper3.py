#!/usr/bin/env python3
"""Train one Paper 3 baseline/P2/control run with recoverable settings."""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


CONFIG_FILES = {
    "baseline": None,
    "p2": "configs/yolov8s-P2P3P4P5.yaml",
    "control": "configs/yolov8s-P2P3P4.yaml",
    "yolov8m": None,
}

PRETRAINED = {
    "baseline": "yolov8s.pt",
    "p2": "yolov8s.pt",
    "control": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--config", choices=sorted(CONFIG_FILES), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_value(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    project = Path(args.project).resolve()
    data_path = Path(args.data).resolve()
    os.chdir(repo)
    name = args.name or f"{args.config}_seed{args.seed}"
    run_dir = project / name
    best_path = run_dir / "weights" / "best.pt"
    if best_path.exists() and not args.overwrite:
        print(f"Skip existing checkpoint: {best_path}")
        return

    config_file = CONFIG_FILES[args.config]
    if config_file:
        config_path = repo / config_file
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        model = YOLO(str(config_path))
        model.load(PRETRAINED[args.config])
    else:
        model = YOLO(PRETRAINED[args.config])

    project.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "cudnn": torch.backends.cudnn.version()
        if torch.cuda.is_available()
        else None,
        "code_commit": git_value(repo, "rev-parse", "HEAD"),
        "code_describe": git_value(repo, "describe", "--tags", "--always"),
        "code_dirty": bool(git_value(repo, "status", "--porcelain")),
        "argv": sys.argv,
        "settings": {
            "config": args.config,
            "seed": args.seed,
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "workers": args.workers,
            "device": args.device,
            "deterministic": True,
            "cache": "disk",
            "save_period": 10,
        },
    }

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        cache="disk",
        val=True,
        plots=True,
        save_period=10,
        name=name,
        project=str(project),
        exist_ok=True,
    )

    if not best_path.exists():
        raise FileNotFoundError(f"Training ended without {best_path}")
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["checkpoint"] = str(best_path)
    metadata["checkpoint_bytes"] = best_path.stat().st_size
    with open(run_dir / "recovery_metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Checkpoint ready: {best_path}")


if __name__ == "__main__":
    main()
