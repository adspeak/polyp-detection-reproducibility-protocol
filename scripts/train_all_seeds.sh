#!/usr/bin/env bash
# Train baseline and +P2 under all eight seeds (42-49).
# Each run writes runs/<config>_seed<seed>/weights/best.pt
#
# Usage:
#   bash scripts/train_all_seeds.sh [path/to/data.yaml]

set -euo pipefail

DATA="${1:-./ldpolyp/data.yaml}"

for seed in 42 43 44 45 46 47 48 49; do
  echo "=== baseline, seed ${seed} ==="
  python scripts/train.py --config baseline --seed "${seed}" --data "${DATA}"
  echo "=== +P2, seed ${seed} ==="
  python scripts/train.py --config p2 --seed "${seed}" --data "${DATA}"
done

# Optional: the supplementary control configuration. It is reported as a
# supplementary observation only (Section 4.4), so a single seed is sufficient.
echo "=== control (P2P3P4), seed 42 ==="
python scripts/train.py --config control --seed 42 --data "${DATA}"

echo "All runs complete. Collect the 16 best.pt checkpoints into ./weights"
echo "before running evaluation/multiseed_eval.py."
