# -*- coding: utf-8 -*-
"""
Figure 3 - Recall as a function of relative lesion area under a single seed
(Section 5.1).

Sliding-window recall curves for the baseline and the +P2 candidate under the
representative seed (42), with the baseline-candidate crossover marked. This is
the single-seed picture the protocol is designed to test: the crossover falls
near the image-space equivalent of the 5 mm diminutive reference, which the
multi-seed analysis in Section 5.2 shows is not a reproducible property.

Curve estimation and crossover location use the same procedure as
`evaluation/crossover_fp_analysis.py`: a fixed logarithmic grid of 40 points
spanning 0.3% to 30% relative area, a multiplicative window with w = 0.5, a
minimum of ten lesions per window, and linear interpolation at the first
descending crossing.

Style matches the other figures: greyscale, serif, vector PDF with embedded
fonts (pdf.fonttype 42).

Usage:
    python figures/make_fig3_singleseed_crossover.py \
        --dataset ./ldpolyp \
        --baseline ./weights/baseline_seed42.pt \
        --candidate ./weights/p2_seed42.pt \
        --outdir figures
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


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


def per_gt_hits(weights, dataset, imgsz=1280, conf=0.25, iou_m=0.5):
    """Return an array of (relative_area, hit) for every ground-truth box."""
    m = YOLO(weights)
    recs = []
    for ip in sorted(glob.glob(f"{dataset}/images/test/*.jpg")):
        lp = f"{dataset}/labels/test/{os.path.basename(ip)[:-4]}.txt"
        gts = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w, h = map(float, p)
                    gts.append([cx, cy, w, h])
        if not gts:
            continue
        r = m.predict(ip, conf=conf, imgsz=imgsz, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        for cx, cy, w, h in gts:
            rel = w * h
            g = [(cx - w / 2) * W, (cy - h / 2) * H,
                 (cx + w / 2) * W, (cy + h / 2) * H]
            io = box_iou(g, preds)
            hit = 1 if (len(io) > 0 and io.max() >= iou_m) else 0
            recs.append((rel, hit))
    return np.array(recs)


def sliding_recall(recs, xs, win=0.5, min_lesions=10):
    """Sliding-window recall over relative area (in percent)."""
    rel = recs[:, 0] * 100
    hit = recs[:, 1]
    ys = []
    for x in xs:
        lo, hi = x / (1 + win), x * (1 + win)
        mask = (rel >= lo) & (rel < hi)
        ys.append(hit[mask].mean() * 100 if mask.sum() >= min_lesions else np.nan)
    return np.array(ys)


def find_crossover(yb, yp, xs):
    """First descending crossing of the candidate below the baseline."""
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


def build_figure(xs, yb, yp, cross):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

    ax.plot(xs, yb, "o-", color="#999999", lw=1.8, ms=5,
            markerfacecolor="#d9d9d9", markeredgecolor="#333333",
            markeredgewidth=0.7, label="Baseline")
    ax.plot(xs, yp, "s--", color="#333333", lw=1.8, ms=5,
            markerfacecolor="#4d4d4d", markeredgecolor="#000000",
            markeredgewidth=0.7, label="Modification (+P2)")

    if cross:
        ax.axvline(cross, ls=":", color="#666666", lw=1.2, alpha=0.9)
        ax.text(cross, 6, f"crossover\n{cross:.2f}%", color="#333333",
                ha="center", fontsize=9.5)

    ax.set_xscale("log")
    ax.set_xlabel("Polyp relative area (%, log scale)", fontsize=12)
    ax.set_ylabel("Local recall (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./ldpolyp")
    ap.add_argument("--baseline", required=True,
                    help="Baseline checkpoint for the representative seed (42).")
    ap.add_argument("--candidate", required=True,
                    help="+P2 checkpoint for the representative seed (42).")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=0.5)
    ap.add_argument("--min-lesions", type=int, default=10)
    ap.add_argument("--grid-lo", type=float, default=0.3)
    ap.add_argument("--grid-hi", type=float, default=30.0)
    ap.add_argument("--grid-points", type=int, default=40)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    base = per_gt_hits(args.baseline, args.dataset, args.imgsz,
                       args.conf, args.iou_match)
    p2 = per_gt_hits(args.candidate, args.dataset, args.imgsz,
                     args.conf, args.iou_match)
    print(f"ground-truth boxes: baseline={len(base)}  +P2={len(p2)}")

    xs = np.logspace(np.log10(args.grid_lo), np.log10(args.grid_hi),
                     args.grid_points)
    yb = sliding_recall(base, xs, args.window, args.min_lesions)
    yp = sliding_recall(p2, xs, args.window, args.min_lesions)
    cross = find_crossover(yb, yp, xs)
    print(f"crossover: {cross:.2f}%" if cross else "crossover: none found")

    os.makedirs(args.outdir, exist_ok=True)
    fig = build_figure(xs, yb, yp, cross)

    stem = os.path.join(args.outdir, "fig3_singleseed_crossover")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    print(f"saved: {stem}.pdf")
    print(f"saved: {stem}.png")


if __name__ == "__main__":
    main()
