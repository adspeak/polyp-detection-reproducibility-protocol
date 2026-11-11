#!/usr/bin/env bash
set -euo pipefail

# Safe setup and execution helper for AutoDL.
# Usage:
#   bash autodl_reproduce_v1.0.1.sh setup
#   bash autodl_reproduce_v1.0.1.sh prepare
#   bash autodl_reproduce_v1.0.1.sh kvasir
#   bash autodl_reproduce_v1.0.1.sh train-kvasir
#   bash autodl_reproduce_v1.0.1.sh eval-kvasir
#   bash autodl_reproduce_v1.0.1.sh eval-coco
#   bash autodl_reproduce_v1.0.1.sh train-positive
#   bash autodl_reproduce_v1.0.1.sh eval-positive
#   bash autodl_reproduce_v1.0.1.sh ensemble
#   bash autodl_reproduce_v1.0.1.sh smoke
#   bash autodl_reproduce_v1.0.1.sh benchmark
#   bash autodl_reproduce_v1.0.1.sh train
#   bash autodl_reproduce_v1.0.1.sh collect
#   bash autodl_reproduce_v1.0.1.sh evaluate
#   bash autodl_reproduce_v1.0.1.sh backup
#   bash autodl_reproduce_v1.0.1.sh check-results

ROOT="${POLYP_ROOT:-/root/autodl-tmp/polyp-repro}"
REPO="$ROOT/polyp-detection-reproducibility-protocol"
DATA="${POLYP_DATA_ROOT:-/root/autodl-tmp/polyp-data}"
BACKUP_DIR="${POLYP_BACKUP_DIR:-$ROOT/backups}"
TAG="v1.0.1"
MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.1.0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

setup() {
  mkdir -p "$ROOT" "$DATA"
  if [[ ! -d "$REPO/.git" ]]; then
    git clone --branch "$TAG" --depth 1 \
      https://github.com/adspeak/polyp-detection-reproducibility-protocol \
      "$REPO"
  fi
  git -C "$REPO" fetch --tags --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" || true
  git -C "$REPO" checkout --detach "$TAG"

  cd "$REPO"
  if [[ "${SETUP_SKIP_PIP:-0}" == "1" ]]; then
    echo "Skipping package reinstall; reusing the existing environment."
  else
    python -m pip install \
      "ultralytics==${ULTRALYTICS_VERSION}" \
      "numpy==1.26.4" \
      "scipy==1.11.4" \
      "Pillow==10.2.0" \
      "PyYAML==6.0.1" \
      "matplotlib==3.8.3" \
      "opencv-python==4.9.0.80" \
      "pycocotools==2.0.7"
  fi
  python -c "import pycocotools" 2>/dev/null ||
    python -m pip install "pycocotools==2.0.7"

  python - <<'PY'
import cv2
import numpy
import scipy
import torch
import ultralytics

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("OpenCV:", cv2.__version__)
PY

  python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

base = Path.cwd()
checks = {
    "p2": base / "configs/yolov8s-P2P3P4P5.yaml",
    "control": base / "configs/yolov8s-P2P3P4.yaml",
}
for name, cfg in checks.items():
    model = YOLO(str(cfg))
    stride = [float(x) for x in model.model.model[-1].stride]
    params = sum(p.numel() for p in model.model.parameters())
    print(f"{name}: params={params}, stride={stride}")
PY

  python "$SCRIPT_DIR/verify_recorded_results.py" \
    --multiseed results/multiseed_8seeds.json \
    --crossover results/crossover_fp_8seeds.json
}

prepare() {
  cd "$REPO"
  if [[ -f "$DATA/ldpolyp/data.yaml" ]]; then
    echo "Using existing processed dataset: $DATA/ldpolyp"
  else
    [[ -d "$DATA/LDPolypVideo/TrainValid/Images" ]] ||
      die "Place LDPolypVideo at $DATA/LDPolypVideo (TrainValid/Images and Test/Images)."
    python data_preparation/prepare_dataset.py \
      --ldpolyp-root "$DATA/LDPolypVideo" \
      --out "$DATA/ldpolyp"
  fi
  python data_preparation/compute_size_thresholds.py \
    --dataset "$DATA/ldpolyp" \
    --out "$DATA/size_thresholds.npy"
  echo "Expected image counts: train=2118, val=402, test=1570."
  echo "Expected box terciles: 626 / 611 / 640."
}

kvasir() {
  cd "$REPO"
  if [[ -f "$DATA/kvasir_yolo/data.yaml" ]]; then
    echo "Using existing Kvasir YOLO dataset: $DATA/kvasir_yolo"
  else
    mkdir -p "$DATA/kvasir"
    if [[ ! -f "$DATA/kvasir/kvasir-seg.zip" ]]; then
      curl -L --retry 5 --retry-delay 3 \
        "https://datasets.simula.no/downloads/kvasir-seg.zip" \
        -o "$DATA/kvasir/kvasir-seg.zip"
    fi
    unzip -q -o "$DATA/kvasir/kvasir-seg.zip" -d "$DATA/kvasir"
    python "$SCRIPT_DIR/prepare_kvasir_seg.py" \
      --kvasir-root "$DATA/kvasir" \
      --out "$DATA/kvasir_yolo" \
      --train-frac 0.80 --val-frac 0.20 --seed 42
  fi
  python data_preparation/compute_size_thresholds.py \
    --dataset "$DATA/kvasir_yolo" \
    --out "$DATA/kvasir_thresholds.npy"
  echo "Expected: train=800, val=200, p33 about 11.10%, p67 about 27.50%."
}

train_kvasir() {
  cd "$REPO"
  [[ -f "$DATA/kvasir_yolo/data.yaml" ]] || die "Run kvasir first."
  for config in baseline p2; do
    python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
      --config "$config" --seed 42 --data "$DATA/kvasir_yolo/data.yaml" \
      --epochs 100 --imgsz 640 --batch 16 --project "$ROOT/kvasir_runs"
  done
}

eval_kvasir() {
  cd "$REPO"
  [[ -f "$DATA/kvasir_thresholds.npy" ]] || die "Run kvasir first."
  python "$SCRIPT_DIR/evaluate_kvasir.py" \
    --dataset "$DATA/kvasir_yolo" \
    --thresholds "$DATA/kvasir_thresholds.npy" \
    --baseline "$ROOT/kvasir_runs/baseline_seed42/weights/best.pt" \
    --candidate "$ROOT/kvasir_runs/p2_seed42/weights/best.pt" \
    --split val --imgsz 640 --conf 0.25 --iou-match 0.5 \
    --out "$ROOT/evaluation/kvasir_eval.json"
}

eval_coco() {
  cd "$REPO"
  [[ -d "$DATA/ldpolyp/images/test" ]] || die "Run prepare first."
  mkdir -p "$ROOT/evaluation"
  python "$SCRIPT_DIR/evaluate_coco_ap.py" \
    --dataset "$DATA/ldpolyp" --split test --imgsz 1280 --conf 0.001 \
    --baseline "$ROOT/weights/baseline_seed42.pt" \
    --candidate "$ROOT/weights/p2_seed42.pt" \
    --out "$ROOT/evaluation/coco_ap.json"
}

train_positive() {
  cd "$REPO"
  [[ -f "$DATA/ldpolyp/data.yaml" ]] || die "Run prepare first."
  for seed in 42 43 44 45 46 47 48 49; do
    python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
      --config yolov8m --seed "$seed" --data "$DATA/ldpolyp/data.yaml" \
      --epochs 100 --imgsz 1280 --batch 16 --project "$ROOT/positive_runs"
  done
}

eval_positive() {
  cd "$REPO"
  mkdir -p "$ROOT/positive_weights" "$ROOT/evaluation"
  for seed in 42 43 44 45 46 47 48 49; do
    cp "$ROOT/runs/baseline_seed${seed}/weights/best.pt" \
      "$ROOT/positive_weights/baseline_seed${seed}.pt"
    cp "$ROOT/positive_runs/yolov8m_seed${seed}/weights/best.pt" \
      "$ROOT/positive_weights/yolov8m_seed${seed}.pt"
  done
  python evaluation/multiseed_eval.py \
    --dataset "$DATA/ldpolyp" \
    --thresholds "$DATA/size_thresholds.npy" \
    --weights-dir "$ROOT/positive_weights" \
    --baseline baseline_seed42.pt baseline_seed43.pt baseline_seed44.pt \
      baseline_seed45.pt baseline_seed46.pt baseline_seed47.pt \
      baseline_seed48.pt baseline_seed49.pt \
    --candidate yolov8m_seed42.pt yolov8m_seed43.pt yolov8m_seed44.pt \
      yolov8m_seed45.pt yolov8m_seed46.pt yolov8m_seed47.pt \
      yolov8m_seed48.pt yolov8m_seed49.pt \
    --seeds 42 43 44 45 46 47 48 49 \
    --imgsz 1280 --conf 0.25 --iou-match 0.5 \
    --out "$ROOT/evaluation/positive_multiseed.json"
}

ensemble() {
  cd "$REPO"
  mkdir -p "$ROOT/evaluation"
  python "$SCRIPT_DIR/ensemble_eval.py" \
    --dataset "$DATA/ldpolyp" \
    --thresholds "$DATA/size_thresholds.npy" \
    --baseline "$ROOT/weights/baseline_seed42.pt" \
      "$ROOT/weights/baseline_seed43.pt" \
      "$ROOT/weights/baseline_seed44.pt" \
      "$ROOT/weights/baseline_seed45.pt" \
      "$ROOT/weights/baseline_seed46.pt" \
      "$ROOT/weights/baseline_seed47.pt" \
      "$ROOT/weights/baseline_seed48.pt" \
      "$ROOT/weights/baseline_seed49.pt" \
    --candidate "$ROOT/weights/p2_seed42.pt" \
      "$ROOT/weights/p2_seed43.pt" \
      "$ROOT/weights/p2_seed44.pt" \
      "$ROOT/weights/p2_seed45.pt" \
      "$ROOT/weights/p2_seed46.pt" \
      "$ROOT/weights/p2_seed47.pt" \
      "$ROOT/weights/p2_seed48.pt" \
      "$ROOT/weights/p2_seed49.pt" \
    --groups 42,43,44 45,46,47 48,49 \
    --split test --imgsz 1280 --conf 0.25 --iou-match 0.5 \
    --out "$ROOT/evaluation/ensemble_eval.json"
}

smoke() {
  cd "$REPO"
  [[ -f "$DATA/ldpolyp/data.yaml" ]] || die "Run prepare first."
  python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
    --config baseline --seed 42 --data "$DATA/ldpolyp/data.yaml" \
    --epochs 1 --imgsz 640 --batch 4 --project "$ROOT/smoke_runs"
  python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
    --config p2 --seed 42 --data "$DATA/ldpolyp/data.yaml" \
    --epochs 1 --imgsz 640 --batch 4 --project "$ROOT/smoke_runs"
  echo "Smoke run passed. Full-run reproduction uses imgsz=1280 and batch=16."
}

train() {
  cd "$REPO"
  [[ -f "$DATA/ldpolyp/data.yaml" ]] || die "Run prepare first."
  for seed in 42 43 44 45 46 47 48 49; do
    for config in baseline p2; do
      run_dir="$ROOT/runs/${config}_seed${seed}"
      if [[ -f "$run_dir/weights/best.pt" ]]; then
        echo "Skip existing $run_dir/weights/best.pt"
        continue
      fi
      python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
        --config "$config" --seed "$seed" --data "$DATA/ldpolyp/data.yaml" \
        --epochs 100 --imgsz 1280 --batch 16 --project "$ROOT/runs"
    done
  done
  python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
    --config control --seed 42 --data "$DATA/ldpolyp/data.yaml" \
    --epochs 100 --imgsz 1280 --batch 16 --project "$ROOT/runs"
}

benchmark() {
  cd "$REPO"
  [[ -f "$DATA/ldpolyp/data.yaml" ]] || die "Run prepare first."
  python "$SCRIPT_DIR/train_paper3.py" --repo "$REPO" \
    --config baseline --seed 42 --data "$DATA/ldpolyp/data.yaml" \
    --epochs 100 --imgsz 1280 --batch 16 --project "$ROOT/runs"
  echo "Benchmark run complete. Inspect time, GPU utilization, and peak memory."
  echo "If memory is insufficient, rerun with batch=8 and document the deviation."
}

collect() {
  cd "$REPO"
  mkdir -p "$ROOT/weights"
  for seed in 42 43 44 45 46 47 48 49; do
    cp "$ROOT/runs/baseline_seed${seed}/weights/best.pt" \
      "$ROOT/weights/baseline_seed${seed}.pt"
    cp "$ROOT/runs/p2_seed${seed}/weights/best.pt" \
      "$ROOT/weights/p2_seed${seed}.pt"
  done
  ls -lh "$ROOT/weights"
}

evaluate() {
  cd "$REPO"
  mkdir -p "$ROOT/evaluation"
  python evaluation/multiseed_eval.py \
    --dataset "$DATA/ldpolyp" \
    --thresholds "$DATA/size_thresholds.npy" \
    --weights-dir "$ROOT/weights" \
    --baseline baseline_seed42.pt baseline_seed43.pt baseline_seed44.pt \
      baseline_seed45.pt baseline_seed46.pt baseline_seed47.pt \
      baseline_seed48.pt baseline_seed49.pt \
    --candidate p2_seed42.pt p2_seed43.pt p2_seed44.pt p2_seed45.pt \
      p2_seed46.pt p2_seed47.pt p2_seed48.pt p2_seed49.pt \
    --seeds 42 43 44 45 46 47 48 49 \
    --imgsz 1280 --conf 0.25 --iou-match 0.5 \
    --out "$ROOT/evaluation/multiseed_from_scratch.json"

  # The author script has no --thresholds option here, despite its old README.
  python evaluation/crossover_fp_analysis.py \
    --dataset "$DATA/ldpolyp" \
    --weights-dir "$ROOT/weights" \
    --baseline baseline_seed42.pt baseline_seed43.pt baseline_seed44.pt \
      baseline_seed45.pt baseline_seed46.pt baseline_seed47.pt \
      baseline_seed48.pt baseline_seed49.pt \
    --candidate p2_seed42.pt p2_seed43.pt p2_seed44.pt p2_seed45.pt \
      p2_seed46.pt p2_seed47.pt p2_seed48.pt p2_seed49.pt \
    --seeds 42 43 44 45 46 47 48 49 \
    --imgsz 1280 --conf 0.25 --iou-match 0.5 \
    --out "$ROOT/evaluation/crossover_from_scratch.json"

  echo "Compare these JSON files with results/*_8seeds.json."
  echo "Note: multiseed_eval.py prints population SD (ddof=0), while the paper uses sample SD."
}

backup() {
  mkdir -p "$BACKUP_DIR"
  cd "$ROOT"
  backup_items=()
  for item in weights evaluation kvasir_runs; do
    [[ -e "$item" ]] && backup_items+=("$item")
  done
  if [[ ${#backup_items[@]} -gt 0 ]]; then
    tar -czf "$BACKUP_DIR/paper3_weights_and_results.tar.gz" \
      "${backup_items[@]}"
  fi
  if [[ -d "$DATA/ldpolyp" ]]; then
    tar -czf "$BACKUP_DIR/ldpolyp_dataset.tar.gz" -C "$DATA" ldpolyp
  fi
  if [[ -d "$DATA/kvasir_yolo" ]]; then
    tar -czf "$BACKUP_DIR/kvasir_yolo.tar.gz" -C "$DATA" kvasir_yolo
  fi
  find "$BACKUP_DIR" -maxdepth 1 -type f -print0 |
    sort -z | xargs -0 sha256sum > "$BACKUP_DIR/SHA256SUMS.txt"
  echo "Backup written to $BACKUP_DIR"
  echo "Copy this directory to a second physical/cloud location before closing the instance."
}

check_results() {
  cd "$REPO"
  python "$SCRIPT_DIR/verify_recorded_results.py" \
    --multiseed results/multiseed_8seeds.json \
    --crossover results/crossover_fp_8seeds.json
}

case "$MODE" in
  setup|prepare|kvasir|train-kvasir|eval-kvasir|eval-coco|train-positive|eval-positive|ensemble|smoke|benchmark|train|collect|evaluate|backup|check-results) "$MODE" ;;
  *)
    echo "Usage: bash $0 {setup|prepare|kvasir|train-kvasir|eval-kvasir|eval-coco|train-positive|eval-positive|ensemble|smoke|benchmark|train|collect|evaluate|backup|check-results}"
    exit 2
    ;;
esac
