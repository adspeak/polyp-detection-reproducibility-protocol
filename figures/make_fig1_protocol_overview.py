# -*- coding: utf-8 -*-
"""
Figure 1 - Overview of the proposed evaluation protocol (Section 3).

Renders the protocol schematic: the two configurations, multi-seed training,
the three protocol components, the operational assessment rule, and the two
possible reliability outcomes.

This figure is a schematic of the protocol itself; it plots no experimental
data, so it depends on no results file and can be regenerated standalone.

Style follows the other figures in the paper: greyscale, serif, vector PDF
with embedded fonts (pdf.fonttype 42) so the text remains selectable.

Usage:
    python figures/make_fig1_protocol_overview.py --outdir figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # LaTeX-style math (Delta, subscripts)
    "pdf.fonttype": 42,         # embed fonts -> text stays selectable/editable
    "ps.fonttype": 42,
})

C_WHITE, C_LIGHT, C_MID, C_DARK, EDGE = "white", "#ededed", "#d9d9d9", "#4d4d4d", "black"


def box(ax, cx, cy, w, h, text, fc, ec=EDGE, tc="black", fs=8.2,
        bold=False, lw=1.0, pad=0.02):
    """Rounded box centred at (cx, cy) with centred text."""
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad={pad},rounding_size=2.2",
                       facecolor=fc, edgecolor=ec, lw=lw, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight=("bold" if bold else "normal"), linespacing=1.35, zorder=5)


def arrow(ax, x0, y0, x1, y1, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=11, lw=lw, color="black"))


def build_figure():
    fig, ax = plt.subplots(figsize=(6.2, 8.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 200)
    ax.axis("off")

    cx = 50

    # inputs: baseline detector and candidate modification
    box(ax, cx, 190, 62, 9,
        r"$\bf{Baseline\ detector}$ $B$  $\bf{and}$  $\bf{Candidate\ modification}$ $C$",
        C_WHITE, fs=8.0)
    arrow(ax, cx, 185.5, cx, 180.5)

    # multi-seed training
    box(ax, cx, 175, 52, 11,
        r"Train $B$, $C$ under $\geq 3$ independent seeds"
        "\n(fixed, reproducible settings)",
        C_LIGHT, fs=8.0)
    arrow(ax, cx, 169.5, cx, 164.5)

    # the three protocol components
    cont_top, cont_bot = 162, 116
    ax.add_patch(Rectangle((cx - 33, cont_bot), 66, cont_top - cont_bot,
                 fill=False, edgecolor="black", lw=1.0, linestyle=(0, (4, 3))))
    ax.text(cx, cont_top - 4.5, "Evaluation protocol", ha="center", va="center",
            fontstyle="italic", fontweight="bold", fontsize=8.6)

    box(ax, cx, 149, 60, 12,
        r"1. Multi-seed stability assessment" "\n"
        r"mean $\pm$ SD, mean difference $\Delta_k$, paired test", C_LIGHT, fs=7.8)
    box(ax, cx, 135, 60, 12,
        "2. Clinically anchored size stratification\n"
        "data-driven thresholds with clinical reference (5 mm)", C_LIGHT, fs=7.8)
    box(ax, cx, 122, 60, 12,
        "3. Scale-response stability analysis\n"
        "recall-vs-scale curve; crossover stability across seeds", C_LIGHT, fs=7.8)
    arrow(ax, cx, cont_bot - 0.5, cx, cont_bot - 5.5)

    # operational assessment rule
    box(ax, cx, 106, 62, 12,
        r"$\bf{Operational\ assessment\ rule}$" "\n"
        r"$\Delta_k > 0$    supportive test    $\Delta_k > \mathrm{SD}_{\mathrm{seed}}$",
        C_DARK, tc="white", fs=8.2)
    arrow(ax, cx, 99.5, cx, 93)

    # decision
    ax.text(cx, 90, "All conditions met?", ha="center", va="center",
            fontstyle="italic", fontsize=8.4)
    lx, rx = 27, 73
    arrow(ax, cx - 4, 86.5, lx, 79.5)
    arrow(ax, cx + 4, 86.5, rx, 79.5)
    ax.text(35, 84, "Yes", ha="center", va="center", fontsize=7.8)
    ax.text(65, 84, "No", ha="center", va="center", fontsize=7.8)

    box(ax, lx, 73, 34, 12, "Reliable\nimprovement", C_WHITE, fs=8.0)
    box(ax, rx, 73, 34, 14, "Not reliably\ndistinguishable\nfrom baseline",
        C_WHITE, fs=8.0)
    arrow(ax, lx, 67, cx - 6, 60.5)
    arrow(ax, rx, 66, cx + 6, 60.5)

    box(ax, cx, 55, 56, 12,
        r"$\bf{Protocol\ report}$" "\n" r"$\bf{(Table:\ reporting\ checklist)}$",
        C_MID, fs=8.0)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures",
                    help="Directory to write fig1_protocol_overview.{pdf,png}")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fig = build_figure()

    pdf_path = os.path.join(args.outdir, "fig1_protocol_overview.pdf")
    png_path = os.path.join(args.outdir, "fig1_protocol_overview.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
