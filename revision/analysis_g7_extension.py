#!/usr/bin/env python3
"""G7 exploratory extension analysis (Paper3 S1 v1.0, Section 6).

New batch: baseline640_seed50-57 (reference B) vs baseline1280_seed50-57
(candidate C), compared as in the positive-control analysis. Original batch
(seeds 42-49) is not re-analysed; its frozen result is read from
positive_control.json.
"""

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
from itertools import product
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy import stats

STRATA = ("small", "medium", "large")
NEW_SEEDS = ["50", "51", "52", "53", "54", "55", "56", "57"]
ORIG_SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
AP_METRICS = ("AP", "AP50", "AP_S", "AP_M", "AP_L", "AR_S")
EXP_MS = "3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559"
EXP_CF = "c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2"


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
        if not os.path.realpath(r["dataset"]) == os.path.realpath(dataset):
            continue
        recs[r["checkpoint_sha256"]] = r
    return recs


def welch_of_means(B, C):
    B, C = np.asarray(B, float), np.asarray(C, float)
    n, m = len(B), len(C)
    vb, vc = B.var(ddof=1), C.var(ddof=1)
    se = math.sqrt(vb / n + vc / m)
    nu = (vb / n + vc / m) ** 2 / ((vb / n) ** 2 / (n - 1) + (vc / m) ** 2 / (m - 1))
    d = float(C.mean() - B.mean())
    tstat = d / se if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(tstat), nu))
    tcrit = stats.t.ppf(0.975, nu)
    return {"delta": d, "se": se, "df": nu, "t": tstat, "p": p,
            "ci95": [d - tcrit * se, d + tcrit * se]}


def paired_mean_diff(B, C):
    d = np.asarray(C, float) - np.asarray(B, float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = stats.t.ppf(0.975, n - 1)
    return {
        "differences": [float(x) for x in d],
        "mean": md,
        "sd": sd,
        "se": sd / math.sqrt(n),
        "df": n - 1,
        "ci95": [md - t * sd / math.sqrt(n), md + t * sd / math.sqrt(n)],
    }


def run_public_with_cache(args, cpc, tools):
    recs = load_cache_index(args.cache, args.dataset)
    cpc.CachedYOLO.index = recs
    cpc.CachedYOLO.split = "test"
    cpc.CachedYOLO.dataset = os.path.realpath(args.dataset)

    plan = {"reference": [], "candidate": []}
    for side, pattern, size in (
        ("reference", "runs_g7/baseline640_seed{seed}", 640),
        ("candidate", "runs_g7/baseline1280_seed{seed}", 1280),
    ):
        for seed in NEW_SEEDS:
            rel = os.path.join(pattern.format(seed=seed), "weights", "best.pt")
            sha = sha256_file(os.path.join(args.root, rel))
            r = recs.get(sha)
            if r is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            if not (int(r["train_args"]["seed"]) == int(seed) and r["settings"]["imgsz"] == size):
                sys.exit("ERROR: cache of %s has imgsz/seed mismatch" % rel)
            plan[side].append({"seed": seed, "checkpoint": rel, "checkpoint_sha256": sha, "tag": r["tag"]})

    class NativeSizeYOLO(cpc.CachedYOLO):
        requested = 1280
        served = {}

        def predict(self, source, conf=0.25, imgsz=None, verbose=False, **kw):
            if not imgsz == NativeSizeYOLO.requested:
                raise RuntimeError("unexpected imgsz %s" % imgsz)
            native = self.rec["settings"]["imgsz"]
            NativeSizeYOLO.served.setdefault(self.rec["tag"], set()).add(native)
            return cpc.CachedYOLO.predict(self, source, conf=conf, imgsz=native, verbose=verbose, **kw)

    ms = os.path.join(args.repo, "evaluation", "multiseed_eval.py")
    cf = os.path.join(args.repo, "evaluation", "crossover_fp_analysis.py")
    if not (sha256_file(ms) == EXP_MS and sha256_file(cf) == EXP_CF):
        sys.exit("ERROR: public evaluation scripts differ from the published versions")
    common = [
        "--dataset", os.path.realpath(args.dataset),
        "--weights-dir", os.path.abspath(args.root),
        "--imgsz", "1280", "--conf", "0.25", "--iou-match", "0.5",
    ]
    B = [p["checkpoint"] for p in plan["reference"]]
    C = [p["checkpoint"] for p in plan["candidate"]]
    jobs = [
        ("multiseed", ms, common + ["--thresholds", os.path.abspath(args.thresholds),
                                    "--baseline"] + B + ["--candidate"] + C + ["--seeds"] + NEW_SEEDS),
        ("crossover", cf, common + ["--baseline"] + B + ["--candidate"] + C + ["--seeds"] + NEW_SEEDS),
    ]
    work = args.out + ".partial_%d" % os.getpid()
    os.makedirs(work)
    outputs = {}
    for name, path, argv in jobs:
        out = os.path.join(work, name + ".json")
        mod = load_module(path, "public_" + name)
        mod.YOLO = NativeSizeYOLO
        buf, old = io.StringIO(), sys.argv
        sys.argv = [path] + argv + ["--out", out]
        try:
            with contextlib.redirect_stdout(buf):
                mod.main()
        finally:
            sys.argv = old
        outputs[name] = json.load(open(out))
    return plan, outputs, work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--g3-result", required=True)
    ap.add_argument("--ap-cost", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    plan, public, work = run_public_with_cache(args, cpc, args.tools)
    g3 = json.load(open(args.g3_result))
    apj = json.load(open(os.path.join(args.ap_cost, "ap_cost.json")))

    pm = public["multiseed"]
    pc = public["crossover"]
    per_seed = {}
    for k in STRATA:
        per_seed[k] = {
            "B_640": [100 * x for x in pm["per_seed"]["baseline"][k]],
            "C_1280": [100 * x for x in pm["per_seed"]["candidate"][k]],
        }

    orig = g3["per_seed_pp"]
    comp_new, comp_orig = {}, {}
    for k in STRATA:
        B, C = per_seed[k]["B_640"], per_seed[k]["C_1280"]
        comp_new[k] = {
            "paired": paired_mean_diff(B, C),
            "welch": welch_of_means(B, C),
            "mean_B": float(np.mean(B)),
            "sd_B": float(np.std(B, ddof=1)),
            "mean_C": float(np.mean(C)),
            "sd_C": float(np.std(C, ddof=1)),
        }
        Bo, Co = orig[k]["reference_640"], orig[k]["candidate_1280"]
        comp_orig[k] = {
            "paired": paired_mean_diff(Bo, Co),
            "welch": welch_of_means(Bo, Co),
            "mean_B": float(np.mean(Bo)),
            "sd_B": float(np.std(Bo, ddof=1)),
            "mean_C": float(np.mean(Co)),
            "sd_C": float(np.std(Co, ddof=1)),
        }

    batch_delta = {}
    for k in STRATA:
        # S1 Section 6: D is the difference between two groups of eight paired
        # differences, compared with a Welch interval, not a four-group contrast.
        d_new = np.asarray(per_seed[k]["C_1280"], float) - np.asarray(per_seed[k]["B_640"], float)
        d_orig = np.asarray(orig[k]["candidate_1280"], float) - np.asarray(orig[k]["reference_640"], float)
        w = welch_of_means(d_orig, d_new)
        batch_delta[k] = {
            "D": w["delta"],
            "se": w["se"],
            "df": w["df"],
            "t": w["t"],
            "p": w["p"],
            "ci95": w["ci95"],
        }

    pooled = {}
    for k in STRATA:
        d16 = (
            list(np.asarray(orig[k]["candidate_1280"]) - np.asarray(orig[k]["reference_640"]))
            + list(np.asarray(per_seed[k]["C_1280"]) - np.asarray(per_seed[k]["B_640"]))
        )
        pooled[k] = paired_mean_diff(np.zeros(len(d16)), d16)

    # Method sensitivity for the new batch, as specified for the primary analysis.
    method_sensitivity = {}
    for k in STRATA:
        d = np.asarray(per_seed[k]["C_1280"], float) - np.asarray(per_seed[k]["B_640"], float)
        tstat, p_paired = stats.ttest_rel(d, np.zeros(len(d)))
        S = d.sum()
        n_flip = 0
        for signs in product([1, -1], repeat=len(d)):
            if abs(sum(s * x for s, x in zip(signs, d))) >= abs(S):
                n_flip += 1
        w_stat, p_wilcoxon = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        method_sensitivity[k] = {
            "paired_t": float(tstat),
            "paired_p": float(p_paired),
            "sign_flip_p": n_flip / (2 ** len(d)),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(p_wilcoxon),
        }

    ap_out = {}
    for metric in AP_METRICS:
        new_tags = [p["tag"] for p in plan["reference"]]
        old_tags = [p["tag"] for p in plan["candidate"]]
        B = [apj["per_model"][t][metric] for t in new_tags]
        C = [apj["per_model"][t][metric] for t in old_tags]
        ap_out[metric] = {
            "paired": paired_mean_diff(B, C),
            "welch": welch_of_means(B, C),
            "mean_B": float(np.mean(B)),
            "mean_C": float(np.mean(C)),
        }

    record = {
        "analysis": "G7 exploratory extension, S1 v1.0 Section 6",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"conf": 0.25, "iou_match": 0.5, "units": "percentage points",
                     "new_seeds": NEW_SEEDS, "original_seeds": ORIG_SEEDS},
        "inputs_sha256": {
            "multiseed_eval.py": EXP_MS,
            "crossover_fp_analysis.py": EXP_CF,
            "thresholds": sha256_file(args.thresholds),
            "g3_result": sha256_file(args.g3_result),
            "ap_cost.json": sha256_file(os.path.join(args.ap_cost, "ap_cost.json")),
            "this_script": sha256_file(__file__),
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "per_seed_pp": {
            "new": {k: per_seed[k] for k in STRATA},
            "original": {k: {"B_640": orig[k]["reference_640"], "C_1280": orig[k]["candidate_1280"]} for k in STRATA},
        },
        "comparisons": {
            "new": {k: comp_new[k] for k in STRATA},
            "original": {k: comp_orig[k] for k in STRATA},
        },
        "batch_difference_D": batch_delta,
        "pooled_16_pairs": pooled,
        "new_batch_method_sensitivity": method_sensitivity,
        "coco_ap_new": ap_out,
        "crossover_and_fp_new": {
            "crossover": pc.get("crossovers"),
            "fp_changes": pc.get("fp_changes"),
        },
    }
    os.makedirs(args.out)
    with open(os.path.join(args.out, "g7_extension.json"), "w") as f:
        json.dump(record, f, indent=2)

    f2 = lambda x: "%+.2f" % x
    lines = ["G7 exploratory extension (S1 v1.0 Section 6); recall in pp", ""]
    for k in STRATA:
        c = comp_new[k]
        lines.append("new %-6s: 640 %.2f (SD %.2f)  1280 %.2f (SD %.2f)  paired delta %s CI [%s, %s]  Welch CI [%s, %s]" % (
            k, c["mean_B"], c["sd_B"], c["mean_C"], c["sd_C"], f2(c["paired"]["mean"]),
            f2(c["paired"]["ci95"][0]), f2(c["paired"]["ci95"][1]),
            f2(c["welch"]["ci95"][0]), f2(c["welch"]["ci95"][1])))
    for k in STRATA:
        d = batch_delta[k]
        lines.append("batch D %-6s: %s  CI [%s, %s] (df %.1f)" % (k, f2(d["D"]), f2(d["ci95"][0]), f2(d["ci95"][1]), d["df"]))
    for k in STRATA:
        p = pooled[k]
        lines.append("pooled16 %-6s: %s  CI [%s, %s]" % (k, f2(p["mean"]), f2(p["ci95"][0]), f2(p["ci95"][1])))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "g7_extension.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "g7_extension.json"))


if __name__ == "__main__":
    main()
