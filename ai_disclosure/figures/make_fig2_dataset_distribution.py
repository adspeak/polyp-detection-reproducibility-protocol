# -*- coding: utf-8 -*-
"""
Figure 2 - Distribution of relative bounding-box area for LDPolypVideo and
Kvasir-SEG (Section 4.1).

Overlaid density histograms on a logarithmic area axis, with the clinical
diminutive-polyp reference (5 mm, 2.04% relative area) marked. The figure shows
that LDPolypVideo holds a substantially larger proportion of lesions near the
diminutive reference than Kvasir-SEG, which motivates its selection as the
primary dataset.

Areas are read from YOLO-format label files as w * h (relative box area),
expressed in percent. LDPolypVideo uses its training split; Kvasir-SEG pools
its train and val splits.

Style matches the other figures: greyscale, serif, vector PDF with embedded
fonts (pdf.fonttype 42).

Usage:
    python figures/make_fig2_dataset_distribution.py \
        --ldpolyp-labels ./ldpolyp/labels/train \
        --kvasir-labels  ./kvasir/labels/train ./kvasir/labels/val \
        --outdir figures
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def get_areas(label_dir):
    """Relative bounding-box areas (in percent) from a YOLO label directory."""
    areas = []
    for lp in glob.glob(f"{label_dir}/*.txt"):
        for ln in open(lp):
            p = ln.split()
            if len(p) == 5:
                _, cx, cy, w, h = map(float, p)
                areas.append(w * h * 100)
    return np.array(areas)


def build_figure(ld_areas, kv_areas, clinical_ref=2.04):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

    bins = np.logspace(np.log10(0.1), np.log10(100), 35)

    ax.hist(ld_areas, bins=bins, alpha=0.75, color="#4d4d4d", edgecolor="black",
            linewidth=0.5, label="LDPolypVideo", density=True)
    ax.hist(kv_areas, bins=bins, alpha=0.55, color="#d9d9d9", edgecolor="black",
            linewidth=0.5, label="Kvasir-SEG", density=True)

    ax.axvline(clinical_ref, ls=":", color="#333333", lw=1.3)

    # widen the y-axis first so the annotation clears the tallest bar
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)
    ax.text(clinical_ref, ax.get_ylim()[1] * 0.80, "5 mm reference\n(2.04%)",
            color="#333333", fontsize=8.5, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Relative bounding-box area (%, log scale)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldpolyp-labels", nargs="+",
                    default=["./ldpolyp/labels/train"],
                    help="LDPolypVideo label directories (training split).")
    ap.add_argument("--kvasir-labels", nargs="+",
                    default=["./kvasir/labels/train", "./kvasir/labels/val"],
                    help="Kvasir-SEG label directories (train and val pooled).")
    ap.add_argument("--clinical-ref", type=float, default=2.04,
                    help="Diminutive reference in relative area, percent.")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    ld_areas = np.concatenate([get_areas(d) for d in args.ldpolyp_labels])
    kv_areas = np.concatenate([get_areas(d) for d in args.kvasir_labels])

    print(f"LDPolypVideo: {len(ld_areas)} boxes, median = {np.median(ld_areas):.2f}%")
    print(f"Kvasir-SEG:   {len(kv_areas)} boxes, median = {np.median(kv_areas):.2f}%")

    os.makedirs(args.outdir, exist_ok=True)
    fig = build_figure(ld_areas, kv_areas, args.clinical_ref)

    stem = os.path.join(args.outdir, "fig2_dataset_distribution")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    print(f"saved: {stem}.pdf")
    print(f"saved: {stem}.png")


if __name__ == "__main__":
    main()
