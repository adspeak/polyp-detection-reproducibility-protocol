"""Kvasir-SEG multi-seed analysis (Paper3 revision analysis addendum v1.0, Section 11).

Baseline (runs_g6/baseline_seed42-49) against +P2 (runs_g6/p2_seed42-49), trained and evaluated
at 640 on data/kvasir_v3_yolo (data_v3 roles, 700 / 150 / 150). Recall per stratum comes from
the public multiseed_eval.py, imported unchanged (hash checked), with its YOLO replaced by the
confidence-0.001 cache reader of check_public_vs_cache.py (a stand-in "ultralytics" module, as
in analysis_a11_v2.py); strata are the terciles of the 700 training boxes
(kvasir_v3_yolo/size_thresholds.npy). This is the formal analysis from the cache; the outputs
of the 640 end-to-end check (public_check_640) are not used.

Reported: the statistics of Section 5 per stratum (compare() of analysis_replication.py, Welch
primary), the small-stratum verdict as the Kvasir-SEG result, the number of test boxes N per
stratum with the step size 100/N pp, and the number of test boxes below the 5 mm reference at
F = 35 mm, a < (5/35)^2. The original single-seed result (-3.6 pp, 800/200 split) is historical
and is not recomputed here.

Built-in checks (the script stops on any failure): every checkpoint has a test cache made at
640 on this dataset with its own seed; the box counts per stratum N are computed from the
labels with the public script's rule (a = w*h; small if a < p33, medium if a < p67, else
large), and every recall the public script reports must be a whole number of boxes out of N.
CPU only.
"""

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import os
import platform
import resource
import sys
import types
from datetime import datetime, timezone

import numpy as np
import scipy

EXP_MS = "3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559"
SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
STRATA = ("small", "medium", "large")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder with check_public_vs_cache.py and analysis_replication.py")
    ap.add_argument("--cache", required=True, help="evaluation/cache/kvasir_v3")
    ap.add_argument("--dataset", required=True, help="data/kvasir_v3_yolo")
    ap.add_argument("--thresholds", required=True, help="data/kvasir_v3_yolo/size_thresholds.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    ms = os.path.join(args.repo, "evaluation", "multiseed_eval.py")
    if not sha256_file(ms) == EXP_MS:
        sys.exit("ERROR: multiseed_eval.py differs from the published version")
    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    stand_in = types.ModuleType("ultralytics")
    stand_in.YOLO = cpc.CachedYOLO
    sys.modules["ultralytics"] = stand_in
    dataset = os.path.realpath(args.dataset)
    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])

    recs = {}
    for js in sorted(glob.glob(os.path.join(args.cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        recs[r["checkpoint_sha256"]] = r
    plan = {}
    for side, pat in (("baseline", "runs_g6/baseline_seed{s}"), ("p2", "runs_g6/p2_seed{s}")):
        plan[side] = []
        for s in SEEDS:
            rel = os.path.join(pat.format(s=s), "weights", "best.pt")
            r = recs.get(sha256_file(os.path.join(args.root, rel)))
            if r is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            if not (sha256_file(r["_npz"]) == r["npz_sha256"] and r["settings"]["imgsz"] == 640
                    and int(r["train_args"]["seed"]) == int(s) and os.path.realpath(r["dataset"]) == dataset):
                sys.exit("ERROR: cache of %s does not match (hash, imgsz, seed or dataset)" % rel)
            plan[side].append({"seed": s, "checkpoint": rel, "tag": r["tag"], "npz_sha256": r["npz_sha256"]})
    cpc.CachedYOLO.index, cpc.CachedYOLO.split, cpc.CachedYOLO.dataset = recs, "test", dataset

    work = args.out + ".partial_%d" % os.getpid()
    os.makedirs(work)
    out = os.path.join(work, "multiseed_kvasir.json")
    argv = ["--dataset", dataset, "--weights-dir", os.path.abspath(args.root), "--imgsz", "640", "--conf", "0.25", "--iou-match", "0.5",
            "--thresholds", os.path.abspath(args.thresholds), "--baseline"] + [p["checkpoint"] for p in plan["baseline"]] + \
           ["--candidate"] + [p["checkpoint"] for p in plan["p2"]] + ["--seeds"] + SEEDS + ["--out", out]
    mod = load_module(ms, "public_multiseed")
    if "torch" in sys.modules:
        sys.exit("ERROR: torch was imported")
    buf, old = io.StringIO(), sys.argv
    sys.argv = [ms] + argv
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = old
    with open(out[:-5] + ".log", "w") as f:
        f.write(buf.getvalue())
    pm = json.load(open(out))

    areas = []
    for ip in sorted(glob.glob(os.path.join(dataset, "images", "test", "*.jpg"))):
        lp = os.path.join(dataset, "labels", "test", os.path.basename(ip)[:-4] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    areas.append(float(p[3]) * float(p[4]))
    areas = np.array(areas)
    N = {"small": int((areas < p33).sum()), "medium": int(((areas >= p33) & (areas < p67)).sum()), "large": int((areas >= p67).sum())}
    problems = []
    for k in STRATA:
        for side, key in (("baseline", "baseline"), ("p2", "candidate")):
            for x in pm["per_seed"][key][k]:
                hits = x * N[k]
                if not abs(hits - round(hits)) < 1e-9:
                    problems.append("%s %s recall %r is not a multiple of 1/%d" % (side, k, x, N[k]))
    if not sum(N.values()) == len(areas):
        problems.append("strata do not cover every box")
    if problems:
        sys.exit("ERROR: built-in checks failed (outputs kept in %s):\n  %s" % (work, "\n  ".join(problems)))
    below5 = int((areas < (5.0 / 35.0) ** 2).sum())
    comp = {}
    for k in STRATA:
        Bk, Ck = [100 * x for x in pm["per_seed"]["baseline"][k]], [100 * x for x in pm["per_seed"]["candidate"][k]]
        if np.std(Bk, ddof=1) == 0 and np.std(Ck, ddof=1) == 0:
            comp[k] = {"undefined": "zero variance in both configurations", "baseline": Bk, "p2": Ck}
        else:
            comp[k] = rep.compare(Bk, Ck)

    rec = {"analysis": "Kvasir-SEG multi-seed analysis, addendum v1.0 Section 11",
           "run_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
           "settings": {"imgsz": 640, "conf": 0.25, "iou_match": 0.5, "units": "pp of recall", "primary": "Welch",
                        "strata": "terciles of the 700 training boxes", "p33": p33, "p67": p67},
           "test_boxes": {"total": int(len(areas)), "per_stratum": N, "step_pp": {k: (100.0 / N[k] if N[k] else None) for k in STRATA},
                          "below_5mm_at_F35": below5},
           "historical": "original single-seed result on the 800/200 split: -3.6 pp (addendum Section 11); not recomputed",
           "inputs_sha256": {"multiseed_eval.py": EXP_MS, "thresholds": sha256_file(args.thresholds),
                             "check_public_vs_cache.py": sha256_file(os.path.join(args.tools, "check_public_vs_cache.py")),
                             "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py"))},
           "builtin_checks": "caches match checkpoints (640, seed, dataset); every recall is a whole number of boxes of its stratum",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__),
           "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
           "per_seed_pp": {k: {"baseline": [100 * x for x in pm["per_seed"]["baseline"][k]], "p2": [100 * x for x in pm["per_seed"]["candidate"][k]]} for k in STRATA},
           "comparisons": comp, "kvasir_result_small_stratum_verdict": comp["small"].get("verdict")}
    with open(os.path.join(work, "kvasir.json"), "w") as f:
        json.dump(rec, f, indent=2)
    f2 = lambda x: "%+.2f" % x
    stp = lambda k: ("%.2f pp" % (100.0 / N[k])) if N[k] else "n/a"
    L = ["Kvasir-SEG (addendum Section 11): +P2 minus baseline, 8 seeds, 640, pp of recall; Welch primary",
         "test boxes %d: small %d (step %s), medium %d (step %s), large %d (step %s); below 5 mm at F = 35 mm: %d" % (
             len(areas), N["small"], stp("small"), N["medium"], stp("medium"), N["large"], stp("large"), below5),
         "thresholds p33 %.6f, p67 %.6f" % (p33, p67), ""]
    L.append("per seed, small stratum: " + ", ".join("%s %.1f->%.1f (%s)" % (s, b, c, f2(c - b)) for s, b, c in zip(
        SEEDS, rec["per_seed_pp"]["small"]["baseline"], rec["per_seed_pp"]["small"]["p2"])))
    for k in STRATA:
        c = comp[k]
        if "undefined" in c:
            L.append("  %-6s statistics undefined: %s" % (k, c["undefined"]))
            continue
        w, p = c["welch"], c["paired"]
        L.append("  %-6s baseline %.2f (SD %.2f)  +P2 %.2f (SD %.2f)  delta %s  Welch CI [%s, %s] df %.1f -> %s" % (
            k, c["mean_reference"], c["sd_reference"], c["mean_candidate"], c["sd_candidate"], f2(w["delta"]),
            f2(w["ci95"][0]), f2(w["ci95"][1]), w["df"], c["verdict"]))
        L.append("         SD_pooled %.2f, delta/SD_pooled %.2f, P(one +P2 run > one baseline run) %.2f; paired CI [%s, %s], t p %.4f, r %.2f" % (
            c["sd_pooled"], w["delta"] / c["sd_pooled"] if c["sd_pooled"] > 0 else float("nan"),
            c["prob_one_candidate_run_beats_one_reference_run_normal_model"], f2(p["ci95"][0]), f2(p["ci95"][1]), p["t_test_p"],
            p["r_between_configurations"]))
        L.append("         sign-flip p %.4f, Wilcoxon p %s, permutation p %.4f, bootstrap CI [%s, %s], Shapiro p %.2f / %.2f; prediction interval [%s, %s]; original rule %s" % (
            p["exact_sign_flip_p"], "%.4f" % p["exact_wilcoxon"]["p"] if p["exact_wilcoxon"]["p"] is not None else "n/a",
            c["exact_two_sample_permutation_p"], f2(c["bootstrap_percentile_ci95"][0]), f2(c["bootstrap_percentile_ci95"][1]),
            c["shapiro_p"]["reference"], c["shapiro_p"]["candidate"], f2(c["prediction_interval_welch"][0]), f2(c["prediction_interval_welch"][1]),
            "met" if c["original_three_condition_rule"]["all_three"] else "not met"))
    L.append("")
    if "undefined" in comp["small"]:
        L.append("== KVASIR-SEG RESULT (small stratum): statistics undefined (%s)" % comp["small"]["undefined"])
    else:
        L.append("== KVASIR-SEG RESULT (small stratum, Welch): %s; delta %s, CI [%s, %s]" % (
            comp["small"]["verdict"], f2(comp["small"]["welch"]["delta"]), f2(comp["small"]["welch"]["ci95"][0]), f2(comp["small"]["welch"]["ci95"][1])))
    L.append("   historical: original single-seed result -3.6 pp on the 800/200 split (not recomputed)")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(work, "kvasir.txt"), "w") as f:
        f.write(txt)
    os.rename(work, args.out)
    print(txt)
    print("written:", os.path.join(args.out, "kvasir.json"))


if __name__ == "__main__":
    main()
