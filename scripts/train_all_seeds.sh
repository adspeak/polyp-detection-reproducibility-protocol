#!/usr/bin/env bash
# Train baseline and +P2 under all three seeds (42, 43, 44).
# Run from the repository root, after preparing the dataset:
#   python data_preparation/prepare_dataset.py --ldpolyp-root <LDPolypVideo> --out ./ldpolyp
#
# Each run writes runs/<config>_seed<seed>/weights/best.pt
set -e

DATA=./ldpolyp/data.yaml

for seed in 42 43 44; do
  echo "=== baseline, seed ${seed} ==="
  python scripts/train.py --config baseline --seed "${seed}" --data "${DATA}"

  echo "=== +P2, seed ${seed} ==="
  python scripts/train.py --config p2 --seed "${seed}" --data "${DATA}"
done

# Optional: the supplementary control configuration (single seed is sufficient,
# it is reported only as a supplementary observation).
echo "=== control (P2P3P4), seed 42 ==="
python scripts/train.py --config control --seed 42 --data "${DATA}"

echo "All training runs complete. Checkpoints are under runs/."
