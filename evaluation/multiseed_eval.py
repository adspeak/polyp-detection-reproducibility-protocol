"""
Multi-seed, size-stratified recall evaluation (Sections 3.1, 5.2).

This is the core evaluation of the protocol's case study. For each of the three
training seeds (42, 43, 44), it evaluates a baseline and a +P2 checkpoint on the
fixed test set, computes recall in the small / medium / large strata, then
aggregates across seeds (mean +/- std) and applies the operational assessment
rule to the small stratum via a paired t-test on the per-seed differences.

Reproduces (on the paper's checkpoints):
    Table 3  - small-lesion recall per seed and mean +/- SD
    Table 4  - recall by lesion size across seeds
    the reported small-stratum result: +2.4 +/- 2.8 pp, paired t-test p = 0.30

Terminology note (kept precise, per the paper): the across-seed difference
C_bar_k - B_bar_k is the "mean difference" (Delta_k), the PRIMARY quantity.
Cohen's d (the standardized effect size) is reported only descriptively and is
highly uncertain with three seeds; it is not the basis for the reliability call.

Recall matching: a ground-truth box counts as detected if some prediction
(at confidence >= conf) has IoU >= iou_match with it. Each GT box is assigned to
a stratum by its relative area (w * h) against the frozen training-set
thresholds p33 / p67 loaded from size_thresholds.npy (decimal fractions).

Usage:
    python multiseed_eval.py \
        --dataset ./ldpolyp \
        --thresholds size_thresholds.npy \
        --weights-dir ./weights \
        --baseline best_baseline_1280_v2.pt best_baseline_1280_s43.pt best_baseline_1280_s44.pt \
        --candidate best_P2_1280_v2.pt best_P2_1280_s43.pt best_P2_1280_s44.pt \
        --seeds 42 43 44 \
        --imgsz 1280 --conf 0.25 --iou-match 0.5 \
        --out multiseed_results.json
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
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


def eval_size(weight, dataset, p33, p67, imgsz, conf=0.25, iou_match=0.5):
    """Return {small,medium,large: recall} for one checkpoint on the test set."""
    m = YOLO(weight)
    hit = {"small": 0, "medium": 0, "large": 0}
    tot = {"small": 0, "medium": 0, "large": 0}

    for ip in sorted(glob.glob(f"{dataset}/images/test/*.jpg")):
        lp = f"{dataset}/labels/test/{os.path.basename(ip)[:-4]}.txt"
        gts = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w2, h = map(float, p)
                    gts.append([cx, cy, w2, h])
        if not gts:
            continue

        r = m.predict(ip, conf=conf, imgsz=imgsz, verbose=False)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        pr = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))

        for cx, cy, w2, h in gts:
            ra = w2 * h
            bk = "small" if ra < p33 else ("medium" if ra < p67 else "large")
            tot[bk] += 1
            g = [(cx - w2 / 2) * W, (cy - h / 2) * H,
                 (cx + w2 / 2) * W, (cy + h / 2) * H]
            io = box_iou(g, pr)
            if len(io) > 0 and io.max() >= iou_match:
                hit[bk] += 1

    return {k: (hit[k] / tot[k] if tot[k] > 0 else 0.0) for k in tot}


def cohens_d_paired(diffs):
    """Descriptive standardized effect size for paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./ldpolyp")
    ap.add_argument("--thresholds", default="size_thresholds.npy")
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="Baseline checkpoint filenames, one per seed (in seed order).")
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="+P2 checkpoint filenames, one per seed (in seed order).")
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--out", default="multiseed_results.json")
    args = ap.parse_args()

    assert len(args.baseline) == len(args.candidate) == len(args.seeds), \
        "Provide one baseline and one candidate checkpoint per seed."

    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = thr["p33"], thr["p67"]

    data = {"baseline": {"small": [], "medium": [], "large": []},
            "candidate": {"small": [], "medium": [], "large": []}}

    print("Per-seed small-lesion recall:")
    for s, bw, pw in zip(args.seeds, args.baseline, args.candidate):
        rb = eval_size(os.path.join(args.weights_dir, bw), args.dataset,
                       p33, p67, args.imgsz, args.conf, args.iou_match)
        rp = eval_size(os.path.join(args.weights_dir, pw), args.dataset,
                       p33, p67, args.imgsz, args.conf, args.iou_match)
        for k in ["small", "medium", "large"]:
            data["baseline"][k].append(rb[k])
            data["candidate"][k].append(rp[k])
        print(f"  seed {s}: baseline_small={rb['small'] * 100:.1f}%  "
              f"+P2_small={rp['small'] * 100:.1f}%  "
              f"diff={(rp['small'] - rb['small']) * 100:+.1f}pp")

    print("\n" + "=" * 64)
    print("Across-seed statistics (mean +/- SD, n = %d):" % len(args.seeds))
    print(f"{'stratum':10s}{'baseline':>16s}{'+P2':>16s}{'mean diff':>14s}")
    for k in ["small", "medium", "large"]:
        b = np.array(data["baseline"][k]) * 100
        p = np.array(data["candidate"][k]) * 100
        print(f"{k:10s}{b.mean():>8.1f}+/-{b.std():.1f}%"
              f"{p.mean():>8.1f}+/-{p.std():.1f}%{(p.mean() - b.mean()):>+11.1f}pp")

    # operational assessment rule on the small stratum
    bs = np.array(data["baseline"]["small"]) * 100
    ps = np.array(data["candidate"]["small"]) * 100
    diffs = ps - bs
    delta = float(ps.mean() - bs.mean())         # mean difference (primary)
    t, pval = stats.ttest_rel(ps, bs)            # paired t-test (supplementary)
    d = cohens_d_paired(diffs)                   # standardized effect (descriptive)
    sd_seed = float(np.std(np.concatenate([bs, ps]), ddof=1))  # pooled seed-to-seed SD

    print("\n" + "=" * 64)
    print("Operational assessment rule (small stratum):")
    print(f"  mean difference Delta_small = {delta:+.1f} pp  (primary quantity)")
    print(f"  paired t-test (supplementary): t = {t:.2f}, p = {pval:.3f}")
    print(f"  Cohen's d (descriptive only): {d:.2f}")
    print(f"  pooled seed-to-seed SD = {sd_seed:.1f} pp")
    cond1 = delta > 0
    cond2 = pval < 0.05
    cond3 = delta > sd_seed
    print(f"  condition 1 (Delta > 0):        {cond1}")
    print(f"  condition 2 (paired test p<.05):{cond2}")
    print(f"  condition 3 (Delta > SD_seed):  {cond3}")
    reliable = cond1 and cond2 and cond3
    print(f"  => reliable improvement: {reliable}")
    print("=" * 64)

    json.dump(
        {"per_seed": data, "seeds": args.seeds,
         "small": {"mean_difference_pp": delta, "t": float(t), "p": float(pval),
                   "cohens_d": d, "sd_seed_pp": sd_seed, "reliable": bool(reliable)}},
        open(args.out, "w"), indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
