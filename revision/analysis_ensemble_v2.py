"""Ensemble analysis (Paper3 revision analysis addendum v1.0, Section 9).

Models: the 16 replication checkpoints (baseline and +P2, seeds 42-49), cached predictions only.
Fusion: weighted_boxes_fusion from ensemble_boxes 1.0.9, loaded by file path from the vendored
wheel (the package itself is not installed); coordinates normalised by image width and height,
equal weights, iou_thr 0.55, skip_box_thr 0.001, conf_type "avg". A set of size 1 is the single
model's own cached predictions (no fusion).

Operating points
  matched FP (primary): target = mean number of false positives per validation image of the 16
      single models at confidence 0.25; for every set, the threshold is the smallest value on the
      grid 0.001 ... 0.999 whose validation FP rate does not exceed the target; applied to test.
  fixed 0.25 (secondary).
  A prediction is kept when its confidence is > the threshold (as Ultralytics NMS keeps
  conf > conf_thres; deviation log, entry 1). Recall and false positives follow the public code:
  a ground-truth box is hit when a kept prediction overlaps it at IoU >= 0.5; a kept prediction
  is a false positive when it overlaps no ground-truth box at IoU >= 0.5. Strata use the
  thresholds in size_thresholds.npy; small false positives use the public crossover script's
  0.0187 on the prediction's own relative area.

Statistic (k = 3, primary): T = mean small-stratum recall of the 56 +P2 seed triplets minus the
mean of the 56 baseline triplets (same seeds). Exact model-label permutation test: the fused
result of all C(16, 3) = 560 three-model sets is precomputed at both operating points; for each
of the C(16, 8) = 12,870 labellings, T* is recomputed; two-sided p = share with |T*| >= |T_obs|.
The same is done for k = 1 (which at the fixed operating point is the exact two-sample
permutation test of Section 5) and for the other strata and the false-positive counts.
Descriptive curves: for k = 1..8, all C(8, k) seed subsets per configuration, at both operating
points; SD with ddof = 1 (NA at k = 8).
"""

import argparse
import glob
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np

STRATA = ("small", "medium", "large")
METRICS = ("small", "medium", "large", "fp_total", "fp_small")
GRID = [round(i / 1000.0, 3) for i in range(1, 1000)]
FIXED = 0.25
SMALL_FP_AREA = 0.0187
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
TOL = 1e-9

G = {}  # worker globals


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_iou(b, bs):
    """Copied from the public evaluation scripts (same operations, same dtypes)."""
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


def load_wbf(path):
    spec = importlib.util.spec_from_file_location("ensemble_boxes_wbf_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.weighted_boxes_fusion


def load_split(cache_root, split, tags):
    preds, names, hw = [], None, None
    for t in tags:
        js = os.path.join(cache_root, split, t + ".json")
        r = json.load(open(js))
        npz = js[:-5] + ".npz"
        if not sha256_file(npz) == r["npz_sha256"]:
            raise SystemExit("ERROR: %s changed after it was written" % npz)
        z = np.load(npz)
        n = [str(x) for x in z["names"]]
        if names is None:
            names, hw = n, z["orig_hw"]
        elif not (n == names and np.array_equal(z["orig_hw"], hw)):
            raise SystemExit("ERROR: %s covers different images" % t)
        off = z["offsets"]
        preds.append([(z["xyxy"][off[i]:off[i + 1]], z["conf"][off[i]:off[i + 1]]) for i in range(len(n))])
    return names, hw, preds


def load_gt(dataset, split, names, hw, p33, p67):
    gt = []
    for n, (h, w) in zip(names, hw):
        W, H = int(w), int(h)
        lp = os.path.join(dataset, "labels", split, n[:-4] + ".txt")
        boxes, strata = [], []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w2, hh = map(float, p)
                    ra = w2 * hh
                    strata.append("small" if ra < p33 else ("medium" if ra < p67 else "large"))
                    boxes.append([(cx - w2 / 2) * W, (cy - hh / 2) * H, (cx + w2 / 2) * W, (cy + hh / 2) * H])
        gt.append((boxes, strata, W, H))
    return gt


def init_worker(state):
    G.update(state)
    G["wbf"] = load_wbf(state["wbf_path"])
    warnings.simplefilter("ignore")


def fuse(members, img, split):
    """Predictions of a set on one image: (xyxy pixels, conf)."""
    P = G["preds"][split]
    if len(members) == 1:
        b, c = P[members[0]][img]
        return b, c
    W, H = G["gt"][split][img][2], G["gt"][split][img][3]
    bl, sl, ll = [], [], []
    for m in members:
        b, c = P[m][img]
        bn = b.astype(np.float64) / np.array([W, H, W, H], dtype=np.float64)
        bl.append(bn.tolist())
        sl.append(c.astype(np.float64).tolist())
        ll.append([0] * len(c))
    if not any(len(s) for s in sl):
        return np.zeros((0, 4)), np.zeros((0,))
    fb, fs, _ = G["wbf"](bl, sl, ll, weights=None, iou_thr=0.55, skip_box_thr=0.001, conf_type="avg")
    fb = np.asarray(fb, dtype=np.float64).reshape(-1, 4) * np.array([W, H, W, H], dtype=np.float64)
    return fb, np.asarray(fs, dtype=np.float64)


def scan(members, split):
    """Per ground-truth box: highest confidence of a matching prediction (-1 if none);
    per prediction that matches no box: its confidence and relative area.
    The IoU of every (box, prediction) pair is computed once with the public box_iou."""
    gtmax, gstr, fpc, fpa = [], [], [], []
    for img, (boxes, strata, W, H) in enumerate(G["gt"][split]):
        pb, pc = fuse(members, img, split)
        matched = np.zeros(len(pc), dtype=bool)
        for g, s in zip(boxes, strata):
            io = box_iou(g, pb)
            ok = io >= 0.5 if len(io) else np.zeros(0, dtype=bool)
            gtmax.append(float(pc[ok].max()) if ok.any() else -1.0)
            gstr.append(s)
            matched |= ok
        for k in np.nonzero(~matched)[0]:
            fpc.append(float(pc[k]))
            fpa.append(float((pb[k, 2] - pb[k, 0]) * (pb[k, 3] - pb[k, 1]) / (W * H)))
    return np.array(gtmax), np.array(gstr), np.array(fpc), np.array(fpa)


def metrics_at(t, gtmax, gstr, fpc, fpa):
    out = {}
    for s in STRATA:
        sel = gstr == s
        out[s] = float((gtmax[sel] > t).sum()) / float(sel.sum()) * 100 if sel.sum() else None
    keep = fpc > t
    out["fp_total"] = int(keep.sum())
    out["fp_small"] = int((keep & (fpa < SMALL_FP_AREA)).sum())
    return out


def work(members):
    t0 = time.time()
    vg, vs, vfc, vfa = scan(members, "val")
    n_val = len(G["gt"]["val"])
    target = G["target"]
    thr = None
    if target is not None:
        srt = np.sort(vfc)
        for t in GRID:
            if (len(srt) - np.searchsorted(srt, t, side="right")) / n_val <= target + TOL:
                thr = t
                break
    tg, ts, tfc, tfa = scan(members, "test")
    res = {"members": list(members), "val_fp_rate_at_fixed": float((vfc > FIXED).sum()) / n_val,
           "fixed": metrics_at(FIXED, tg, ts, tfc, tfa), "seconds": time.time() - t0}
    if target is not None:
        res["matched_threshold"] = thr
        res["val_fp_rate_at_matched"] = None if thr is None else float((vfc > thr).sum()) / n_val
        res["matched"] = None if thr is None else metrics_at(thr, tg, ts, tfc, tfa)
    return res


def run_sets(sets, state, workers):
    if workers <= 1:
        init_worker(state)
        return [work(s) for s in sets]
    ctx = mp.get_context("fork")
    out, t0 = [], time.time()
    with ctx.Pool(workers, initializer=init_worker, initargs=(state,)) as pool:
        for r in pool.imap_unordered(work, sets, chunksize=1):
            out.append(r)
            if len(out) % 50 == 0 or len(out) == len(sets):
                el = time.time() - t0
                print("  %d/%d sets, %.0f min elapsed, about %.0f min left" % (
                    len(out), len(sets), el / 60, el / len(out) * (len(sets) - len(out)) / 60))
                sys.stdout.flush()
    done = {tuple(r["members"]): r for r in out}
    return [done[tuple(s)] for s in sets]  # back in input order (imap_unordered returns completion order)


def perm_index(n_models, n_lab, k):
    """For every labelling (n_lab of n_models labelled +P2): indices of its k-subsets and of the
    complement's k-subsets in the list of all k-subsets of the n_models models."""
    combos = list(itertools.combinations(range(n_models), k))
    pos = {c: i for i, c in enumerate(combos)}
    labs = list(itertools.combinations(range(n_models), n_lab))
    A, B = [], []
    for lab in labs:
        rest = tuple(m for m in range(n_models) if m not in lab)
        A.append([pos[c] for c in itertools.combinations(lab, k)])
        B.append([pos[c] for c in itertools.combinations(rest, k)])
    return combos, labs, np.array(A), np.array(B)


def perm_test(vals, labs, A, B, labelled):
    """vals: metric per k-subset (array aligned with combos). Exact two-sided permutation p."""
    t_all = vals[A].mean(axis=1) - vals[B].mean(axis=1)
    obs = float(t_all[labs.index(tuple(labelled))])
    p = float(np.mean(np.abs(t_all) >= abs(obs) - TOL))
    return obs, p, len(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--wbf", required=True, help="Vendored ensemble_boxes_wbf.py")
    ap.add_argument("--wbf-wheel", required=True, help="The wheel it was extracted from (hash recorded)")
    ap.add_argument("--check-multiseed", required=True, help="replay/multiseed_p2.json (k = 1 check)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    tags = ["%s__baseline_seed%d" % (args.project, s) for s in SEEDS] + ["%s__p2_seed%d" % (args.project, s) for s in SEEDS]
    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = thr["p33"], thr["p67"]
    state = {"wbf_path": args.wbf, "preds": {}, "gt": {}, "target": None}
    for split in ("val", "test"):
        names, hw, preds = load_split(args.cache, split, tags)
        state["preds"][split] = preds
        state["gt"][split] = load_gt(os.path.abspath(args.dataset), split, names, hw, p33, p67)
    nv = len(state["gt"]["val"])
    nt = sum(len(g[0]) for g in state["gt"]["test"])
    print("models %d; val %d images (%d boxes); test %d images (%d boxes)" % (
        len(tags), nv, sum(len(g[0]) for g in state["gt"]["val"]), len(state["gt"]["test"]), nt))

    # singles first: they define the matched-FP target
    t0 = time.time()
    singles = run_sets([(i,) for i in range(16)], state, args.workers)
    target = float(np.mean([r["val_fp_rate_at_fixed"] for r in singles]))
    state["target"] = target
    print("matched-FP target: %.4f false positives per validation image (mean of the 16 single models at 0.25)" % target)
    singles = run_sets([(i,) for i in range(16)], state, args.workers)

    # k = 1 check against the public scripts (replay of the cache)
    ms = json.load(open(args.check_multiseed))
    bad = []
    for i, r in enumerate(singles):
        grp, j = ("baseline", i) if i < 8 else ("candidate", i - 8)
        for s in STRATA:
            ref = ms["per_seed"][grp][s][j] * 100
            if not abs(ref - r["fixed"][s]) < 1e-9:
                bad.append((tags[i], s, ref, r["fixed"][s]))
    print("k = 1 check against the public multiseed script at 0.25: %s" % ("identical" if not bad else "MISMATCH %s" % bad[:4]))
    if bad:
        sys.exit("ERROR: k = 1 recall differs from the public script; stopping")

    # all sets needed
    base, cand = list(range(8)), list(range(8, 16))
    need = set(frozenset(c) for c in itertools.combinations(range(16), 3))
    for k in range(2, 9):
        for c in itertools.combinations(range(8), k):
            need.add(frozenset(base[i] for i in c))
            need.add(frozenset(cand[i] for i in c))
    need -= set(frozenset(s["members"]) for s in singles)
    order = sorted(need, key=lambda s: (-len(s), sorted(s)))  # large sets first for load balance
    print("fusing %d sets with %d workers ..." % (len(order), args.workers))
    sys.stdout.flush()
    fused = run_sets([tuple(sorted(s)) for s in order], state, args.workers)
    allres = {frozenset(r["members"]): r for r in singles + fused}
    print("fusion done in %.0f s" % (time.time() - t0))

    no_thr = [sorted(k) for k, r in allres.items() if r.get("matched") is None]
    if no_thr:
        print("WARNING: no grid threshold reaches the target for %d sets, e.g. %s" % (len(no_thr), no_thr[:3]))

    out = {"settings": {"wbf": {"iou_thr": 0.55, "skip_box_thr": 0.001, "conf_type": "avg", "weights": "equal"},
                        "grid": [GRID[0], GRID[-1], 0.001], "fixed": FIXED, "keep_rule": "conf > threshold",
                        "iou_match": 0.5, "strata_thresholds": [p33, p67], "small_fp_area": SMALL_FP_AREA},
           "matched_fp_target_per_val_image": target,
           "single_val_fp_rate_at_fixed": {tags[i]: singles[i]["val_fp_rate_at_fixed"] for i in range(16)}}

    # permutation tests, k = 1 and k = 3
    tests = {}
    for k in (1, 3):
        combos, labs, A, B = perm_index(16, 8, k)
        for op in ("matched", "fixed"):
            for mname in METRICS:
                vals = np.array([np.nan if allres[frozenset(c)].get(op) is None else allres[frozenset(c)][op][mname]
                                 for c in combos], dtype=float)
                obs, p, nlab = perm_test(vals, labs, A, B, cand)
                tests["k%d_%s_%s" % (k, op, mname)] = {"T": obs, "p_two_sided": p, "labellings": nlab}
    out["permutation_tests"] = tests

    # descriptive curves
    curves = {}
    for op in ("matched", "fixed"):
        curves[op] = {}
        for k in range(1, 9):
            row = {}
            for mname in METRICS:
                b, c, d = [], [], []
                for sub in itertools.combinations(range(8), k):
                    rb, rc = allres[frozenset(base[i] for i in sub)], allres[frozenset(cand[i] for i in sub)]
                    if rb.get(op) is None or rc.get(op) is None:
                        continue
                    b.append(rb[op][mname]); c.append(rc[op][mname]); d.append(rc[op][mname] - rb[op][mname])
                sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else None
                row[mname] = {"n_subsets": len(d), "baseline_mean": float(np.mean(b)), "baseline_sd": sd(b),
                              "p2_mean": float(np.mean(c)), "p2_sd": sd(c), "diff_mean": float(np.mean(d)), "diff_sd": sd(d)}
            curves[op][str(k)] = row
    out["curves"] = curves
    out["sets"] = {",".join(tags[m].split("__")[1] for m in sorted(k)): {kk: v for kk, v in r.items() if not kk == "members"}
                   for k, r in sorted(allres.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))}
    out["inputs"] = {"wbf_module_sha256": sha256_file(args.wbf), "wbf_wheel": os.path.basename(args.wbf_wheel),
                     "wbf_wheel_sha256": sha256_file(args.wbf_wheel),
                     "caches": {t: json.load(open(os.path.join(args.cache, sp, t + ".json")))["npz_sha256"]
                                for t in tags for sp in ("test",)},
                     "caches_val": {t: json.load(open(os.path.join(args.cache, "val", t + ".json")))["npz_sha256"] for t in tags},
                     "thresholds_sha256": sha256_file(args.thresholds), "check_multiseed_sha256": sha256_file(args.check_multiseed)}
    out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    out["this_script_sha256"] = sha256_file(__file__)
    out["run_utc"] = datetime.now(timezone.utc).isoformat()
    out["fusion_seconds_total"] = float(sum(r["seconds"] for r in allres.values()))
    os.makedirs(args.out)
    with open(os.path.join(args.out, "ensemble.json"), "w") as f:
        json.dump(out, f, indent=2)

    L = ["== Ensembles (addendum Section 9); recall in pp, FP = test false-positive count",
         "matched-FP target %.4f per validation image; sets fused %d" % (target, len(allres))]
    for k in (3, 1):
        L.append("-- k = %d, +P2 minus baseline (exact model-label permutation p over %d labellings)" % (k, tests["k%d_matched_small" % k]["labellings"]))
        for op in ("matched", "fixed"):
            L.append("   %-8s " % op + "  ".join("%s T %+.2f p %.3f" % (m, tests["k%d_%s_%s" % (k, op, m)]["T"],
                                                                     tests["k%d_%s_%s" % (k, op, m)]["p_two_sided"]) for m in METRICS))
    for op in ("matched", "fixed"):
        L.append("-- curves (%s): k, baseline small mean (SD), +P2 small mean (SD), difference mean (SD)" % op)
        for k in range(1, 9):
            r = curves[op][str(k)]["small"]
            f = lambda x: "NA" if x is None else "%.2f" % x
            L.append("   k=%d n=%-3d B %.2f (%s)  P2 %.2f (%s)  diff %+.2f (%s)" % (
                k, r["n_subsets"], r["baseline_mean"], f(r["baseline_sd"]), r["p2_mean"], f(r["p2_sd"]), r["diff_mean"], f(r["diff_sd"])))
    thr3 = [allres[frozenset(c)]["matched_threshold"] for c in itertools.combinations(range(16), 3)]
    thr1 = [allres[frozenset([i])]["matched_threshold"] for i in range(16)]
    L.append("matched thresholds: singles %s; three-model sets %.3f-%.3f (median %.3f)" % (
        thr1, min(x for x in thr3 if x is not None), max(x for x in thr3 if x is not None), float(np.median([x for x in thr3 if x is not None]))))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "ensemble.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "ensemble.json"))


if __name__ == "__main__":
    main()
