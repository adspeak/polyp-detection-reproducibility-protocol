#!/usr/bin/env python3
"""Recompute the paper's recorded multi-seed and crossover statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def recompute_multiseed(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["data"]
    baseline = np.asarray(rows["baseline"]["small"], dtype=float) * 100
    candidate = np.asarray(rows["p2"]["small"], dtype=float) * 100

    delta = candidate.mean() - baseline.mean()
    pooled_sd = np.sqrt(
        (baseline.std(ddof=1) ** 2 + candidate.std(ddof=1) ** 2) / 2
    )
    t_value, p_value = stats.ttest_rel(candidate, baseline)
    cohens_d = delta / pooled_sd

    print("Recorded small-stratum result (sample SD, ddof=1)")
    print(f"  baseline:  {baseline.mean():.6f} +/- {baseline.std(ddof=1):.6f} pp")
    print(f"  +P2:       {candidate.mean():.6f} +/- {candidate.std(ddof=1):.6f} pp")
    print(f"  mean diff: {delta:+.6f} pp")
    print(f"  pooled SD: {pooled_sd:.6f} pp")
    print(f"  paired t:  {t_value:.6f}, p={p_value:.6f}")
    print(f"  Cohen d:   {cohens_d:.6f}")
    print(
        "  decision:  "
        + (
            "reliable"
            if delta > 0 and p_value < 0.05 and delta > pooled_sd
            else "not reliable"
        )
    )

    for stratum in ("medium", "large"):
        b = np.asarray(rows["baseline"][stratum], dtype=float) * 100
        c = np.asarray(rows["p2"][stratum], dtype=float) * 100
        print(
            f"  {stratum:6s}: B={b.mean():.3f}+/-{b.std(ddof=1):.3f}, "
            f"C={c.mean():.3f}+/-{c.std(ddof=1):.3f}, "
            f"diff={c.mean() - b.mean():+.3f}"
        )

    print("Population SD used by the current multiseed_eval.py print block")
    print(f"  baseline: {baseline.std(ddof=0):.6f} pp")
    print(f"  +P2:      {candidate.std(ddof=0):.6f} pp")


def recompute_crossover(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crossovers = np.asarray(data["crossovers"], dtype=float)
    fp_changes = np.asarray(data["fp_changes"], dtype=float)
    print("\nRecorded scale-response result")
    print(
        f"  crossover: {crossovers.mean():.6f} +/- "
        f"{crossovers.std(ddof=1):.6f}% relative area"
    )
    print(
        f"  small FP:  {fp_changes.mean():+.6f} +/- "
        f"{fp_changes.std(ddof=1):.6f}%"
    )


def kvasir_mask_stats(mask_dir):
    from PIL import Image

    paths = sorted(Path(mask_dir).glob("*.jpg"))
    areas = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("L"))
        y, x = np.where(image > 127)
        if not len(x):
            continue
        height, width = image.shape
        area = ((x.max() - x.min() + 1) * (y.max() - y.min() + 1)) / (
            width * height
        )
        areas.append(area)

    areas = np.asarray(areas)
    print("\nKvasir-SEG mask-derived tight-box distribution")
    print(f"  images:       {len(areas)}")
    print(f"  median:       {np.median(areas) * 100:.6f}%")
    print(f"  below 2.04%:  {(areas < 0.0204).sum()}/{len(areas)}")
    print(
        "  all terciles: "
        f"{np.percentile(areas, 100 / 3) * 100:.6f}% / "
        f"{np.percentile(areas, 200 / 3) * 100:.6f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiseed",
        default="results/multiseed_8seeds.json",
        help="Path to the author's multiseed_8seeds.json.",
    )
    parser.add_argument(
        "--crossover",
        default="results/crossover_fp_8seeds.json",
        help="Path to the author's crossover_fp_8seeds.json.",
    )
    parser.add_argument(
        "--kvasir-mask-dir",
        default=None,
        help="Optional Kvasir-SEG masks directory.",
    )
    args = parser.parse_args()

    recompute_multiseed(args.multiseed)
    recompute_crossover(args.crossover)
    if args.kvasir_mask_dir:
        kvasir_mask_stats(args.kvasir_mask_dir)


if __name__ == "__main__":
    main()
