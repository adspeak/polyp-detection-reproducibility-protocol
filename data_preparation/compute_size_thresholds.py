"""
Compute clinically anchored size-stratification thresholds (Section 4.3).

Two criteria are computed on the TRAINING SET ONLY (leakage control) and their
agreement is examined:

  1. Data-driven tercile thresholds (primary protocol): the training-set boxes
     are partitioned at the terciles of relative bounding-box area, producing
     balanced small / medium / large strata.

  2. Clinical anchoring (auxiliary): under a simplified field-of-view assumption
     of F ~= 35 mm, a lesion of physical diameter d maps to relative area
     (d / F)^2. The 5 mm diminutive boundary then corresponds to ~2.04% relative
     area. This is an order-of-magnitude reference used only to check that the
     data-driven small-lesion boundary falls in a clinically plausible range.

On the paper's dataset build, this reproduces the published values:
    tercile boundaries 1.87% and 6.31%
    balanced strata 626 / 611 / 640 (small / medium / large)

The thresholds are saved to size_thresholds.npy as DECIMAL FRACTIONS
(e.g. 0.0187, 0.0631), NOT percentages, because the evaluation code compares
against un-scaled relative areas (w * h in [0, 1]).

Usage:
    python compute_size_thresholds.py --dataset ./ldpolyp --out size_thresholds.npy
"""

import argparse
import glob
import os

import numpy as np

FOV_MM = 35.0  # reference field diameter (modelling parameter, not a measurement)


def diam_to_relarea(d_mm, fov_mm=FOV_MM):
    return (d_mm / fov_mm) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./ldpolyp",
                    help="YOLO-format dataset dir produced by prepare_dataset.py.")
    ap.add_argument("--out", default="size_thresholds.npy",
                    help="Output .npy path for the computed thresholds.")
    args = ap.parse_args()

    lbl_train = os.path.join(args.dataset, "labels/train")

    # collect relative areas from the TRAINING SET ONLY (leakage control)
    areas = []
    for lp in glob.glob(os.path.join(lbl_train, "*.txt")):
        for ln in open(lp):
            p = ln.split()
            if len(p) == 5:
                w, h = float(p[3]), float(p[4])
                areas.append(w * h)
    areas = np.array(areas)
    print(f"Training-set box count = {len(areas)}")

    # --- primary: data-driven terciles ---
    p33 = np.percentile(areas, 100 / 3)
    p67 = np.percentile(areas, 200 / 3)
    small = int((areas < p33).sum())
    medium = int(((areas >= p33) & (areas < p67)).sum())
    large = int((areas >= p67).sum())

    print("\n[Data-driven terciles (primary protocol)] thresholds on relative area:")
    print(f"  small  : rel_area <  {p33:.6f}  ({p33 * 100:.3f}%)")
    print(f"  medium : {p33:.6f} - {p67:.6f}  ({p33 * 100:.3f}% - {p67 * 100:.3f}%)")
    print(f"  large  : rel_area >= {p67:.6f}  ({p67 * 100:.3f}%)")
    print(f"  strata counts: small={small}  medium={medium}  large={large}")

    # --- auxiliary: clinical anchoring ---
    c_dim = diam_to_relarea(5)    # 5 mm diminutive upper bound
    c_large = diam_to_relarea(10)  # 10 mm large lower bound
    frac_below_5mm = float((areas < c_dim).mean())
    print("\n[Clinical anchoring (auxiliary, FOV ~= 35 mm)]:")
    print(f"  diminutive <=5 mm  ~= rel_area < {c_dim * 100:.3f}%")
    print(f"  large      >=10 mm ~= rel_area >= {c_large * 100:.3f}%")
    print(f"  fraction of training boxes below the 5 mm line = {frac_below_5mm * 100:.1f}%")

    # save thresholds as DECIMAL FRACTIONS (not percentages)
    thr = {
        "p33": float(p33),
        "p67": float(p67),
        "clinical_diminutive": float(c_dim),
        "clinical_large": float(c_large),
        "n_boxes_train": int(len(areas)),
        "bin_counts": {"small": small, "medium": medium, "large": large},
        "frac_below_5mm": frac_below_5mm,
        "fov_mm": FOV_MM,
    }
    np.save(args.out, thr)
    print(f"\nThresholds saved to {args.out} (decimal fractions, e.g. {p33:.4f}).")


if __name__ == "__main__":
    main()
