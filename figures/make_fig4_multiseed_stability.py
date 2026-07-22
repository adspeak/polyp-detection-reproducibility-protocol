# -*- coding: utf-8 -*-
"""
Figure 4 - Recall by lesion-size stratum, mean +/- SD across eight seeds
(Section 5.2, Table 4).

Grouped bar chart with seed-to-seed standard deviation as error bars, showing
that the apparent small-lesion gain is accompanied by comparable or larger
variability across all strata.

Values default to the across-seed statistics reported in the paper; pass
`--results results/multiseed_8seeds.json` to plot them from the recorded
per-seed file instead.

Style matches the other figures: greyscale, serif, vector PDF with embedded
fonts (pdf.fonttype 42).

Usage:
    python figures/make_fig4_multiseed_stability.py --outdir figures
    python figures/make_fig4_multiseed_stability.py --results results/multiseed_8seeds.json
"""

import argparse
import json
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

STRATA = ["Small", "Medium", "Large"]

# Across-seed statistics reported in the paper (Table 4, eight seeds).
PAPER_VALUES = {
    "baseline_mean": [28.3, 43.2, 47.8],
    "baseline_std": [3.1, 5.7, 8.6],
    "p2_mean": [29.2, 42.4, 50.3],
    "p2_std": [3.5, 4.0, 7.9],
}


def load_from_results(path):
    """Recompute mean +/- SD per stratum from the recorded per-seed file."""
    with open(path) as f:
        data = json.load(f)

    # results/multiseed_8seeds.json nests the per-seed recalls under "data",
    # keyed "baseline" and "p2"; accept a flat layout as well.
    per_seed = data.get("data", data)
    out = {}
    for src, key in (("baseline", "baseline"), ("p2", "p2")):
        if src not in per_seed:
            raise KeyError(f"'{src}' not found in {path}")
        means, stds = [], []
        for k in ("small", "medium", "large"):
            v = np.asarray(per_seed[src][k], dtype=float) * 100.0
            means.append(v.mean())
            stds.append(v.std(ddof=1))
        out[f"{key}_mean"] = means
        out[f"{key}_std"] = stds
    return out


def build_figure(vals):
    x = np.arange(len(STRATA))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

    ax.bar(x - width / 2, vals["baseline_mean"], width, yerr=vals["baseline_std"],
           label="Baseline", color="#d9d9d9", edgecolor="black",
           linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.0, "capthick": 1.0})
    ax.bar(x + width / 2, vals["p2_mean"], width, yerr=vals["p2_std"],
           label="Modification (+P2)", color="#4d4d4d", edgecolor="black",
           linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.0, "capthick": 1.0})

    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_xlabel("Lesion-size stratum", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATA, fontsize=11)
    ax.set_ylim(0, 68)
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="Optional results/multiseed_8seeds.json to plot from; "
                         "defaults to the values reported in the paper.")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    vals = load_from_results(args.results) if args.results else PAPER_VALUES

    os.makedirs(args.outdir, exist_ok=True)
    fig = build_figure(vals)

    stem = os.path.join(args.outdir, "fig4_multiseed_stability_8seeds")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    print(f"saved: {stem}.pdf")
    print(f"saved: {stem}.png")


if __name__ == "__main__":
    main()
