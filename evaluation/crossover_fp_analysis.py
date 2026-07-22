"""
Scale-response stability analysis: recall-vs-scale crossover and small-scale
false-positive change across seeds (Sections 3.3 and 5.2).

For each seed this script

  1. estimates the recall-versus-scale curve for the baseline and the candidate
     with a sliding window over relative lesion area on a logarithmic grid,
  2. locates the baseline-candidate crossover by linear interpolation, and
  3. computes the change in small-scale false positives relative to the
     baseline count under the same fixed confidence threshold,

then reports the across-seed mean and standard deviation of both quantities.

This is the implementation used to produce the values reported in the paper and
recorded in `results/crossover_fp_8seeds.json`; the defaults below are the
settings used there. Two details are worth stating explicitly because they
affect the result:

  * The evaluation grid is fixed at 40 logarithmically spaced points spanning
    0.3% to 30% relative area, independent of the seed being evaluated, so that
    crossover locations are directly comparable across seeds.
  * The crossover is taken to be the first point at which the candidate curve
    falls from above the baseline curve to at or below it, interpolated
    linearly between the two bracketing grid points.

A false positive is a prediction (at confidence >= conf) whose IoU with every
ground-truth box is below `--iou-match`. It is assigned to the small scale when
its own relative area falls below the frozen small-lesion threshold (0.0187).
The reported change is (FP_candidate - FP_baseline) / FP_baseline as a
percentage.

Usage:
    python evaluation/crossover_fp_analysis.py \
        --dataset ./ldpolyp \
        --weights-dir ./weights \
        --baseline  baseline_seed42.pt ... baseline_seed49.pt \
        --candidate p2_seed42.pt       ... p2_seed49.pt \
        --seeds 42 43 44 45 46 47 48 49 \
        --imgsz 1280 --conf 0.25 --iou-match 0.5 \
        --out crossover_fp_results.json
"""

import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO


def box_iou(b, bs):
    """IoU of a single box b=[x1,y1,x2,y2] against an array bs of boxes."""
    if len(bs) == 0:
        return np.array([])
    x1 = np.maximum(b[0], bs[:, 0])
    y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2])
    y2 = np.minimum(b[3], bs[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def eval_full(weight, dataset, imgsz=1280, conf=0.25, iou_m=0.5,
              p33=0.0187, p67=0.0631):
    """
    Evaluate one checkpoint on the test set.

    Returns
        recs : array of (relative_area, hit) for every ground-truth box
        (fp_small, fp_med, fp_large) : false-positive counts by predicted scale
    """
    m = YOLO(weight)
    recs = []
    fp_small = fp_med = fp_large = 0

    for ip in sorted(glob.glob(f"{dataset}/images/test/*.jpg")):
        lp = f"{dataset}/labels/test/{os.path.basename(ip)[:-4]}.txt"
        gts = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w2, h = map(float, p)
                    gts.append([cx, cy, w2, h])

        r = m.predict(ip, conf=conf, imgsz=imgsz, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))

        gt_xyxy = []
        for cx, cy, w2, h in gts:
            rel = w2 * h
            g = [(cx - w2 / 2) * W, (cy - h / 2) * H,
                 (cx + w2 / 2) * W, (cy + h / 2) * H]
            gt_xyxy.append(g)
            io = box_iou(g, preds)
            hit = 1 if (len(io) > 0 and io.max() >= iou_m) else 0
            recs.append((rel, hit))

        # false positives: a prediction matching no ground-truth box, binned by
        # its own relative area against the frozen training-set thresholds
        for pb in preds:
            pa = (pb[2] - pb[0]) * (pb[3] - pb[1]) / (W * H)
            matched = False
            for g in gt_xyxy:
                if box_iou(g, np.array([pb]))[0] >= iou_m:
                    matched = True
                    break
            if not matched:
                if pa < p33:
                    fp_small += 1
                elif pa < p67:
                    fp_med += 1
                else:
                    fp_large += 1

    return np.array(recs), (fp_small, fp_med, fp_large)


def sliding_recall(recs, xs, win=0.5, min_lesions=10):
    """
    Sliding-window recall over relative area (in percent).

    For each evaluation point x, recall is computed over the lesions whose
    relative area lies in [x / (1 + win), x * (1 + win)], provided the window
    holds at least `min_lesions` lesions; sparser points are left undefined.
    """
    rel = recs[:, 0] * 100
    hit = recs[:, 1]
    ys = []
    for x in xs:
        lo, hi = x / (1 + win), x * (1 + win)
        mask = (rel >= lo) & (rel < hi)
        ys.append(hit[mask].mean() * 100 if mask.sum() >= min_lesions else np.nan)
    return np.array(ys)


def find_crossover(yb, yp, xs):
    """
    First scale at which the candidate curve falls to or below the baseline.

    Returns the interpolated crossing point, or None if the candidate never
    crosses from above to below on the defined part of the grid.
    """
    diff = yp - yb
    valid = ~np.isnan(diff)
    xv = xs[valid]
    dv = diff[valid]
    for i in range(len(dv) - 1):
        if dv[i] > 0 and dv[i + 1] <= 0:
            x0, x1 = xv[i], xv[i + 1]
            d0, d1 = dv[i], dv[i + 1]
            return x0 + (x1 - x0) * d0 / (d0 - d1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./ldpolyp")
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="Baseline checkpoint filenames, one per seed (in seed order).")
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="+P2 checkpoint filenames, one per seed (in seed order).")
    ap.add_argument("--seeds", nargs="+",
                    default=["42", "43", "44", "45", "46", "47", "48", "49"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=0.5,
                    help="Multiplicative window parameter w (Section 3.3).")
    ap.add_argument("--min-lesions", type=int, default=10,
                    help="Minimum lesions per window for a defined point.")
    ap.add_argument("--grid-lo", type=float, default=0.3,
                    help="Lower end of the evaluation grid, in percent.")
    ap.add_argument("--grid-hi", type=float, default=30.0,
                    help="Upper end of the evaluation grid, in percent.")
    ap.add_argument("--grid-points", type=int, default=40)
    ap.add_argument("--p33", type=float, default=0.0187,
                    help="Frozen small-lesion threshold (relative area).")
    ap.add_argument("--p67", type=float, default=0.0631,
                    help="Frozen large-lesion threshold (relative area).")
    ap.add_argument("--out", default="crossover_fp_results.json")
    args = ap.parse_args()

    assert len(args.baseline) == len(args.candidate) == len(args.seeds), \
        "Provide one baseline and one candidate checkpoint per seed."

    xs = np.logspace(np.log10(args.grid_lo), np.log10(args.grid_hi),
                     args.grid_points)

    crossovers, fp_changes, per_seed = [], [], {}

    print("Per-seed scale-response and false-positive results:")
    for s, bw, pw in zip(args.seeds, args.baseline, args.candidate):
        rb, fpb = eval_full(os.path.join(args.weights_dir, bw), args.dataset,
                            args.imgsz, args.conf, args.iou_match,
                            args.p33, args.p67)
        rp, fpp = eval_full(os.path.join(args.weights_dir, pw), args.dataset,
                            args.imgsz, args.conf, args.iou_match,
                            args.p33, args.p67)

        yb = sliding_recall(rb, xs, args.window, args.min_lesions)
        yp = sliding_recall(rp, xs, args.window, args.min_lesions)
        cr = find_crossover(yb, yp, xs)

        fp_chg = (fpp[0] - fpb[0]) / fpb[0] * 100 if fpb[0] > 0 else 0.0

        crossovers.append(cr)
        fp_changes.append(fp_chg)
        per_seed[s] = {
            "crossover_relative_area_pct": (None if cr is None else float(cr)),
            "fp_small_baseline": int(fpb[0]),
            "fp_small_candidate": int(fpp[0]),
            "fp_small_change_pct": float(fp_chg),
        }

        cr_str = f"{cr:.2f}%" if cr is not None else "none"
        print(f"  seed {s}: crossover={cr_str:>8s}  small-scale FP change={fp_chg:+.0f}%"
              f"  (baseline FP={fpb[0]}, +P2 FP={fpp[0]})")

    cr_valid = [c for c in crossovers if c is not None]

    print("\n" + "=" * 60)
    print(f"Crossover ({len(args.seeds)} seeds)")
    print("=" * 60)
    print(f"  valid crossovers: {len(cr_valid)}/{len(args.seeds)}")
    if cr_valid:
        print(f"  locations: {[f'{c:.2f}' for c in cr_valid]}")
        print(f"  mean = {np.mean(cr_valid):.2f}%   "
              f"SD = {np.std(cr_valid, ddof=1):.2f}%")
        if len(cr_valid) > 1 and np.std(cr_valid, ddof=1) > np.mean(cr_valid):
            print("  note: the SD exceeds the mean, so the crossover location is "
                  "not stable across seeds.")

    print(f"\nSmall-scale false-positive change ({len(args.seeds)} seeds)")
    print("=" * 60)
    print(f"  {[f'{f:+.0f}' for f in fp_changes]}")
    print(f"  mean = {np.mean(fp_changes):+.0f}%   "
          f"SD = {np.std(fp_changes, ddof=1):.0f}%")
    n_pos = int(np.sum(np.asarray(fp_changes) > 0))
    if 0 < n_pos < len(fp_changes):
        print(f"  {n_pos} of {len(fp_changes)} seeds positive - the sign changes "
              f"between seeds.")

    out = {
        "seeds": args.seeds,
        "per_seed": per_seed,
        "crossovers": [float(c) if c is not None else None for c in crossovers],
        "crossover_mean": float(np.mean(cr_valid)) if cr_valid else None,
        "crossover_std": float(np.std(cr_valid, ddof=1)) if len(cr_valid) > 1 else None,
        "fp_changes": [float(f) for f in fp_changes],
        "fp_mean": float(np.mean(fp_changes)),
        "fp_std": float(np.std(fp_changes, ddof=1)),
        "settings": {"window_w": args.window, "min_lesions": args.min_lesions,
                     "grid_lo_pct": args.grid_lo, "grid_hi_pct": args.grid_hi,
                     "grid_points": args.grid_points, "conf": args.conf,
                     "iou_match": args.iou_match, "imgsz": args.imgsz,
                     "p33": args.p33, "p67": args.p67},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
