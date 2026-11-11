#!/usr/bin/env python3
"""A2 normality Q-Q plot (#31), retrained small stratum.

Reads the G1 cached test predictions and recomputes, for the retrained small
stratum, the per-seed coverage recall of the baseline and +P2 and their
per-seed differences, then draws a three-panel normal Q-Q plot and reports
Shapiro-Wilk W/p per panel. Style matches the other paper figures (serif,
cm math, greyscale, vector PDF with embedded fonts).
"""
import argparse, glob, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
CONF = 0.25
IOU_MATCH = 0.5
EXP_MS = "3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cache_index(cache, dataset):
    recs = {}
    for js in sorted(glob.glob(os.path.join(cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        if not sha256_file(r["_npz"]) == r["npz_sha256"]:
            sys.exit("ERROR: %s changed after it was written" % r["_npz"])
        if os.path.realpath(r["dataset"]) == os.path.realpath(dataset):
            recs[r["checkpoint_sha256"]] = r
    return recs


def qq_panel(ax, x, title, ylabel):
    n = len(x)
    xs = np.sort(x)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))  # Blom
    ax.scatter(q, xs, s=18, facecolor="white", edgecolor="black", linewidths=0.8, zorder=3)
    q1, q3 = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    x1, x3 = np.percentile(xs, 25), np.percentile(xs, 75)
    slope = (x3 - x1) / (q3 - q1)
    inter = x1 - slope * q1
    pad = 0.35
    lo, hi = q.min() - pad, q.max() + pad
    ax.plot([lo, hi], [slope * lo + inter, slope * hi + inter], color="black", lw=1.0, zorder=2)
    W, p = stats.shapiro(x)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Theoretical normal quantiles", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.text(0.03, 0.97, "W = %.3f,  p = %.3f" % (W, p), transform=ax.transAxes,
            va="top", ha="left", fontsize=8)
    ax.tick_params(labelsize=8)
    return float(W), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)

    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    recs = load_cache_index(args.cache, args.dataset)
    cpc.CachedYOLO.index = recs
    cpc.CachedYOLO.split = "test"
    cpc.CachedYOLO.dataset = os.path.realpath(args.dataset)

    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])
    dataset_real = os.path.realpath(args.dataset)

    ms_path = os.path.join(args.repo, "evaluation", "multiseed_eval.py")
    if not sha256_file(ms_path) == EXP_MS:
        sys.exit("ERROR: multiseed_eval.py differs from the published version")
    ms = load_module(ms_path, "ms")
    ms.YOLO = cpc.CachedYOLO

    per_seed = {}
    for side, pattern in (("baseline", "runs/baseline_seed{seed}"),
                          ("p2", "runs/p2_seed{seed}")):
        for seed in SEEDS:
            rel = os.path.join(pattern.format(seed=seed), "weights", "best.pt")
            abspath = os.path.join(args.root, rel)
            sha = sha256_file(abspath)
            rec = recs.get(sha)
            if rec is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            imgsz = int(rec["settings"]["imgsz"])
            r = ms.eval_size(abspath, dataset_real, p33, p67, imgsz, CONF, IOU_MATCH)
            per_seed[(side, seed)] = 100.0 * float(r["small"])

    b = np.array([per_seed[("baseline", s)] for s in SEEDS])
    p = np.array([per_seed[("p2", s)] for s in SEEDS])
    d = p - b

    panels = [
        ("Baseline small-stratum recall", b, "Recall (%)"),
        ("+P2 small-stratum recall", p, "Recall (%)"),
        ("Paired difference (+P2 $-$ baseline)", d, "Difference (pp)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    shapiro = {}
    for ax, (title, x, ylabel) in zip(axes, panels):
        W, pval = qq_panel(ax, x, title, ylabel)
        shapiro[title] = {"W": W, "p": pval}
    fig.tight_layout()

    os.makedirs(args.out)
    pdf = os.path.join(args.out, "figS1_qq_small_stratum.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    record = {
        "analysis": "A2 normality Q-Q, retrained small stratum (#31)",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"conf": CONF, "iou_match": IOU_MATCH, "seeds": SEEDS,
                     "p33": p33, "p67": p67},
        "per_seed_recall_pct": {
            "baseline": {s: float(per_seed[("baseline", s)]) for s in SEEDS},
            "p2": {s: float(per_seed[("p2", s)]) for s in SEEDS},
            "difference": {s: float(per_seed[("p2", s)] - per_seed[("baseline", s)])
                           for s in SEEDS},
        },
        "summary_pct": {
            "baseline_mean": float(b.mean()), "baseline_sd": float(b.std(ddof=1)),
            "p2_mean": float(p.mean()), "p2_sd": float(p.std(ddof=1)),
            "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)),
        },
        "shapiro_wilk": shapiro,
        "inputs_sha256": {
            "multiseed_eval.py": sha256_file(ms_path),
            "thresholds": sha256_file(args.thresholds),
            "this_script": sha256_file(__file__),
        },
    }
    with open(os.path.join(args.out, "a2_qq.json"), "w") as f:
        json.dump(record, f, indent=2)

    txt = ["A2 normality Q-Q, retrained small stratum; recall in pp", ""]
    txt.append("baseline: " + ", ".join("%.2f" % v for v in b))
    txt.append("p2      : " + ", ".join("%.2f" % v for v in p))
    txt.append("diff    : " + ", ".join("%+.2f" % v for v in d))
    txt.append("")
    txt.append("baseline mean %.2f SD %.2f" % (b.mean(), b.std(ddof=1)))
    txt.append("p2       mean %.2f SD %.2f" % (p.mean(), p.std(ddof=1)))
    for title, v in shapiro.items():
        txt.append("%s: W=%.4f p=%.4f" % (title, v["W"], v["p"]))
    with open(os.path.join(args.out, "a2_qq.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")

    print("\n".join(txt))
    print("figure:", pdf)


if __name__ == "__main__":
    main()
