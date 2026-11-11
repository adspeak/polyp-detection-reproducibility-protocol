"""Positive control (Paper3 revision analysis addendum v1.0, Section 7): input size 1280 vs 640.

Candidate C: the eight replication baseline checkpoints (runs/baseline_seed42-49), trained
and evaluated at 1280. Reference B: the eight G3 baseline checkpoints (runs_g3/
baseline640_seed42-49), trained and evaluated at 640. Same thresholds (relative area).

Recall per stratum, crossover and small-scale false positives are computed by the public
scripts evaluation/multiseed_eval.py and evaluation/crossover_fp_analysis.py, imported
unchanged from the repository (hashes checked), with their YOLO replaced by the
confidence-0.001 cache reader of check_public_vs_cache.py (boxes kept if conf > 0.25),
exactly as in G1. One difference: the scripts take one --imgsz for both checkpoints, so
they are given 1280 and every checkpoint is served from its own cache, made at its
training input size (640 for B, 1280 for C). The scripts pass imgsz only to predict()
(and print it in the crossover settings), so this equals running them at 640 for B and at
1280 for C. The size served for every model is recorded.

Built-in check: the candidate checkpoints are the replication baselines, already replayed in
G1 at 1280. Their recall per stratum and small-scale FP counts here must equal the G1
replay exactly, or the script stops before any statistics.

Statistics: as in Section 5 (compare() of analysis_replication.py). Primary analysis paired
(Section 7, check13: all 355 tensors identical at 640 and 1280 for seed 42): the verdict uses
the paired t-based 95% CI; Welch and everything else are reported alongside. COCO AP and
model cost come from analysis_ap_cost.py run on the same 16 caches (its ap_cost.json is
read and its inputs are checked against the caches); AP is summarised paired and Welch.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy import stats

EXP_MS = "3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559"
EXP_CF = "c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2"
STRATA = ("small", "medium", "large")
SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
AP_METRICS = ("AP", "AP50", "AP_S", "AP_M", "AP_L", "AR_S")

WORDING = {
    "improvement": ("Evidence of a mean improvement in the small stratum ({label}): the protocol recognises an "
                    "effect of the expected kind. The manuscript may say that the protocol recognised one "
                    "positive control on this dataset; it does not claim from this that the protocol as a "
                    "whole is validated."),
    "none": ("No evidence of a mean difference: reported as \"no evidence of a mean improvement was "
             "established on this dataset\", with its CI and the operating-characteristic simulation "
             "(Section 10) for the observed effect size. The manuscript then states that eight seeds did "
             "not detect this effect, and the claim about the protocol is narrowed accordingly. The "
             "decision rule is not changed."),
    "deterioration": "Evidence of a mean deterioration: reported as observed.",
}


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


def section7_wording(verdict_text):
    if verdict_text.startswith("improvement"):
        return WORDING["improvement"].format(label=verdict_text.split(", ", 1)[1])
    if verdict_text == "deterioration":
        return WORDING["deterioration"]
    return WORDING["none"]


def paired_summary(B, C):
    B, C = np.asarray(B, float), np.asarray(C, float)
    d = C - B
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = stats.t.ppf(0.975, n - 1)
    out = {"mean": md, "sd": sd, "ci95": [md - t * sd / math.sqrt(n), md + t * sd / math.sqrt(n)] if sd > 0 else None,
           "t_test_p": float(stats.ttest_rel(C, B).pvalue) if sd > 0 else None,
           "per_seed_difference": [float(x) for x in d]}
    vb, vc = B.var(ddof=1), C.var(ddof=1)
    se = math.sqrt(vb / n + vc / n)
    if se > 0:
        nu = (vb / n + vc / n) ** 2 / ((vb / n) ** 2 / (n - 1) + (vc / n) ** 2 / (n - 1))
        tw = stats.t.ppf(0.975, nu)
        out["welch"] = {"delta": md, "df": nu, "ci95": [md - tw * se, md + tw * se]}
    else:
        out["welch"] = {"delta": md, "df": None, "ci95": None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Paper3 root ($P3); the checkpoint paths below are relative to it")
    ap.add_argument("--repo", required=True, help="Public repository (evaluation/*.py are imported from it)")
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder with check_public_vs_cache.py and analysis_replication.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--g1-replay", required=True, help="public_check/<stamp>/replay of G1")
    ap.add_argument("--ap-cost", required=True, help="Output folder of analysis_ap_cost.py for these 16 caches")
    ap.add_argument("--reference", default="runs_g3/baseline640_seed{seed}")
    ap.add_argument("--candidate", default="runs/baseline_seed{seed}")
    ap.add_argument("--reference-imgsz", type=int, default=640)
    ap.add_argument("--candidate-imgsz", type=int, default=1280)
    ap.add_argument("--seeds", nargs="+", default=SEEDS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    if not list(args.seeds) == SEEDS:
        sys.exit("ERROR: the frozen analysis uses seeds 42-49")
    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    ms, cf = os.path.join(args.repo, "evaluation", "multiseed_eval.py"), os.path.join(args.repo, "evaluation", "crossover_fp_analysis.py")
    if not (sha256_file(ms) == EXP_MS and sha256_file(cf) == EXP_CF):
        sys.exit("ERROR: the public evaluation scripts differ from the published versions")
    dataset = os.path.realpath(args.dataset)

    # caches: index by checkpoint hash; the 16 checkpoints must each have a test cache at the expected size and seed
    recs = {}
    import glob
    for js in sorted(glob.glob(os.path.join(args.cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        recs[r["checkpoint_sha256"]] = r
    plan = {"reference": [], "candidate": []}
    for side, pat, size in (("reference", args.reference, args.reference_imgsz), ("candidate", args.candidate, args.candidate_imgsz)):
        for s in args.seeds:
            rel = os.path.join(pat.format(seed=s), "weights", "best.pt")
            sha = sha256_file(os.path.join(args.root, rel))
            r = recs.get(sha)
            if r is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            if not sha256_file(r["_npz"]) == r["npz_sha256"]:
                sys.exit("ERROR: %s changed after it was written" % r["_npz"])
            if not (r["settings"]["imgsz"] == size and int(r["train_args"]["seed"]) == int(s)):
                sys.exit("ERROR: %s cache is at imgsz %s, seed %s; expected %s, %s" % (
                    r["tag"], r["settings"]["imgsz"], r["train_args"]["seed"], size, s))
            if not os.path.realpath(r["dataset"]) == dataset:
                sys.exit("ERROR: %s was cached on %s" % (r["tag"], r["dataset"]))
            plan[side].append({"seed": s, "checkpoint": rel, "checkpoint_sha256": sha, "tag": r["tag"],
                               "npz_sha256": r["npz_sha256"], "imgsz": size})
    z0 = np.load(recs[plan["candidate"][0]["checkpoint_sha256"]]["_npz"])
    for p in plan["reference"] + plan["candidate"]:
        z = np.load(recs[p["checkpoint_sha256"]]["_npz"])
        if not (list(z["names"]) == list(z0["names"]) and np.array_equal(z["orig_hw"], z0["orig_hw"])):
            sys.exit("ERROR: %s covers different images" % p["tag"])

    class NativeSizeYOLO(cpc.CachedYOLO):
        requested = args.candidate_imgsz
        served = {}

        def predict(self, source, conf=0.25, imgsz=None, verbose=False, **kw):
            if not imgsz == NativeSizeYOLO.requested:
                raise RuntimeError("imgsz %s passed by the script, expected %s" % (imgsz, NativeSizeYOLO.requested))
            native = self.rec["settings"]["imgsz"]
            NativeSizeYOLO.served.setdefault(self.rec["tag"], set()).add(native)
            return cpc.CachedYOLO.predict(self, source, conf=conf, imgsz=native, verbose=verbose, **kw)

    cpc.CachedYOLO.index = recs
    cpc.CachedYOLO.split = "test"
    cpc.CachedYOLO.dataset = dataset
    work = args.out + ".partial_%d" % os.getpid()
    os.makedirs(work)
    B = [p["checkpoint"] for p in plan["reference"]]
    C = [p["checkpoint"] for p in plan["candidate"]]
    common = ["--dataset", dataset, "--weights-dir", os.path.abspath(args.root), "--imgsz", str(args.candidate_imgsz),
              "--conf", "0.25", "--iou-match", "0.5"]
    jobs = [("multiseed", ms, common + ["--thresholds", os.path.abspath(args.thresholds), "--baseline"] + B + ["--candidate"] + C + ["--seeds"] + args.seeds),
            ("crossover", cf, common + ["--baseline"] + B + ["--candidate"] + C + ["--seeds"] + args.seeds)]
    pub = {}
    for name, path, argv in jobs:
        out = os.path.join(work, "%s_positive_control.json" % name)
        mod = load_module(path, "public_" + name)
        mod.YOLO = NativeSizeYOLO
        buf, old = io.StringIO(), sys.argv
        sys.argv = [path] + argv + ["--out", out]
        try:
            with contextlib.redirect_stdout(buf):
                mod.main()
        finally:
            sys.argv = old
        with open(out[:-5] + ".log", "w") as f:
            f.write(buf.getvalue())
        pub[name] = {"json": json.load(open(out)), "argv": argv + ["--out", out], "file": out}
    served = {t: sorted(v) for t, v in NativeSizeYOLO.served.items()}
    for side in plan:
        for p in plan[side]:
            if not served.get(p["tag"]) == [p["imgsz"]]:
                sys.exit("ERROR: %s was served at %s, expected [%s]" % (p["tag"], served.get(p["tag"]), p["imgsz"]))
    pm, pc = pub["multiseed"]["json"], pub["crossover"]["json"]

    # built-in check against the G1 replay of the same candidate checkpoints at 1280
    g1rec = json.load(open(os.path.join(args.g1_replay, "check_record.json")))
    g1m = json.load(open(os.path.join(args.g1_replay, "multiseed_p2.json")))
    g1c = json.load(open(os.path.join(args.g1_replay, "crossover_p2.json")))
    problems = []
    if not g1rec.get("all_identical"):
        problems.append("G1 check record is not all_identical")
    for p in plan["candidate"]:
        g = g1rec["caches"].get(p["tag"], {})
        if not (g.get("checkpoint_sha256") == p["checkpoint_sha256"] and g.get("npz_sha256") == p["npz_sha256"]):
            problems.append("%s: checkpoint or cache differs from G1" % p["tag"])
    if not list(g1m["seeds"]) == SEEDS:
        problems.append("G1 seeds differ")
    for k in STRATA:
        if not g1m["per_seed"]["baseline"][k] == pm["per_seed"]["candidate"][k]:
            problems.append("recall %s differs from G1: %s vs %s" % (k, g1m["per_seed"]["baseline"][k], pm["per_seed"]["candidate"][k]))
    for s in SEEDS:
        if not g1c["per_seed"][s]["fp_small_baseline"] == pc["per_seed"][s]["fp_small_candidate"]:
            problems.append("seed %s small FP differs from G1" % s)
    if problems:
        sys.exit("ERROR: built-in check against G1 failed (outputs kept in %s):\n  %s" % (work, "\n  ".join(problems)))

    # Section 5 statistics, paired primary (Section 7)
    comp = {}
    for k in STRATA:
        Bk = [100 * x for x in pm["per_seed"]["baseline"][k]]
        Ck = [100 * x for x in pm["per_seed"]["candidate"][k]]
        c = rep.compare(Bk, Ck)
        lo, hi = c["paired"]["ci95"]
        if not abs(c["paired"]["mean"] - c["welch"]["delta"]) < 1e-9:
            sys.exit("ERROR: paired mean and difference of means disagree")
        c["verdict_welch_alongside"] = c.pop("verdict")
        c["verdict_primary_paired"] = rep.verdict(lo, hi, c["welch"]["delta"], c["sd_pooled"])
        comp[k] = c
    small = comp["small"]
    cross = rep.crossover_summary(pc, True)

    # COCO AP and cost from analysis_ap_cost.py on the same caches
    apj = json.load(open(os.path.join(args.ap_cost, "ap_cost.json")))
    for side in plan:
        for p in plan[side]:
            g = apj["inputs"].get(p["tag"])
            if not (g and g["npz_sha256"] == p["npz_sha256"] and g["checkpoint_sha256"] == p["checkpoint_sha256"]):
                sys.exit("ERROR: ap_cost.json does not use the cache of %s" % p["tag"])
    ap_res = {}
    for k in AP_METRICS:
        Bk = [apj["per_model"][p["tag"]][k] for p in plan["reference"]]
        Ck = [apj["per_model"][p["tag"]][k] for p in plan["candidate"]]
        ap_res[k] = dict(paired_summary(Bk, Ck), reference_mean=float(np.mean(Bk)), candidate_mean=float(np.mean(Ck)),
                         reference_sd=float(np.std(Bk, ddof=1)), candidate_sd=float(np.std(Ck, ddof=1)))
    cfg_ref = plan["reference"][0]["tag"].split("__", 1)[1].split("_seed")[0]
    cfg_cand = plan["candidate"][0]["tag"].split("__", 1)[1].split("_seed")[0]
    cost = {"model_cost": {c: (apj.get("model_cost") or {}).get(c) for c in (cfg_ref, cfg_cand)},
            "timing_ms": {c: apj["timing_ms"].get(c) for c in (cfg_ref, cfg_cand)}}

    rec = {
        "analysis": "positive control 1280 vs 640, addendum v1.0 Section 7",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "design": {"candidate": "replication baselines at %d" % args.candidate_imgsz,
                   "reference": "G3 baselines at %d" % args.reference_imgsz,
                   "primary": "paired (mean of same-seed differences, t-based 95% CI); Welch alongside",
                   "pairing_basis": "check13 (5 Nov 2026): seed 42 all 355 tensors identical at 640 and 1280 before training"},
        "settings": {"conf": 0.25, "iou_match": 0.5, "imgsz_passed_to_public_scripts": args.candidate_imgsz,
                     "imgsz_served_per_tag": served, "units": "percentage points of recall",
                     "note": "the crossover JSON of the public script records imgsz %d for both sides; the reference was served at %d" % (
                         args.candidate_imgsz, args.reference_imgsz)},
        "plan": plan,
        "inputs_sha256": {"multiseed_eval.py": EXP_MS, "crossover_fp_analysis.py": EXP_CF,
                          "check_public_vs_cache.py": sha256_file(os.path.join(args.tools, "check_public_vs_cache.py")),
                          "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py")),
                          "thresholds": sha256_file(args.thresholds),
                          "g1_check_record": sha256_file(os.path.join(args.g1_replay, "check_record.json")),
                          "ap_cost.json": sha256_file(os.path.join(args.ap_cost, "ap_cost.json"))},
        "builtin_check_vs_g1": "candidate recall per stratum and small FP counts identical to the G1 replay",
        "public_outputs": {n: {"file": os.path.basename(v["file"]), "argv": v["argv"]} for n, v in pub.items()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "this_script_sha256": sha256_file(__file__),
        "per_seed_pp": {k: {"reference_640": [100 * x for x in pm["per_seed"]["baseline"][k]],
                            "candidate_1280": [100 * x for x in pm["per_seed"]["candidate"][k]]} for k in STRATA},
        "comparisons": comp,
        "crossover_and_fp": cross,
        "coco_ap": ap_res,
        "cost": cost,
        "section7_outcome": {"small_paired_ci95": small["paired"]["ci95"], "verdict": small["verdict_primary_paired"],
                             "welch_verdict_alongside": small["verdict_welch_alongside"],
                             "wording": section7_wording(small["verdict_primary_paired"])},
    }
    with open(os.path.join(work, "positive_control.json"), "w") as f:
        json.dump(rec, f, indent=2)

    f2 = lambda x: "%+.2f" % x
    L = ["Positive control (addendum v1.0, Section 7): 1280 (candidate) minus 640 (reference), 8 seeds, pp of recall",
         "built-in check: candidate values identical to the G1 replay; served sizes %s" % sorted(set(
             "%s:%s" % (t.split("__")[0], v[0]) for t, v in served.items())), ""]
    L.append("per seed, small stratum: " + ", ".join("%s %.1f->%.1f (%s)" % (s, b, c, f2(c - b)) for s, b, c in zip(
        SEEDS, rec["per_seed_pp"]["small"]["reference_640"], rec["per_seed_pp"]["small"]["candidate_1280"])))
    for k in STRATA:
        c = comp[k]
        p, w = c["paired"], c["welch"]
        L.append("  %-6s 640 %.2f (SD %.2f)  1280 %.2f (SD %.2f)  delta %s" % (
            k, c["mean_reference"], c["sd_reference"], c["mean_candidate"], c["sd_candidate"], f2(w["delta"])))
        L.append("         PRIMARY paired CI [%s, %s] t-test p %.4f -> %s" % (
            f2(p["ci95"][0]), f2(p["ci95"][1]), p["t_test_p"], c["verdict_primary_paired"]))
        L.append("         Welch CI [%s, %s] df %.1f -> %s; SD_pooled %.2f; delta/SD_pooled %.2f; P(one 1280 run > one 640 run) %.2f" % (
            f2(w["ci95"][0]), f2(w["ci95"][1]), w["df"], c["verdict_welch_alongside"], c["sd_pooled"],
            w["delta"] / c["sd_pooled"] if c["sd_pooled"] > 0 else float("nan"),
            c["prob_one_candidate_run_beats_one_reference_run_normal_model"]))
        L.append("         r %.2f, sign-flip p %.4f, Wilcoxon p %s, permutation p %.4f, d %.2f, d_z %.2f; bootstrap CI [%s, %s]" % (
            p["r_between_configurations"], p["exact_sign_flip_p"],
            "%.4f" % p["exact_wilcoxon"]["p"] if p["exact_wilcoxon"]["p"] is not None else "n/a",
            c["exact_two_sample_permutation_p"], c["cohens_d"], p["d_z"],
            f2(c["bootstrap_percentile_ci95"][0]), f2(c["bootstrap_percentile_ci95"][1])))
        L.append("         prediction intervals: paired [%s, %s], Welch [%s, %s]; Shapiro p: 640 %.2f, 1280 %.2f, diff %.2f; original 3-condition rule: %s" % (
            f2(p["prediction_interval"][0]), f2(p["prediction_interval"][1]), f2(c["prediction_interval_welch"][0]),
            f2(c["prediction_interval_welch"][1]), c["shapiro_p"]["reference"], c["shapiro_p"]["candidate"],
            c["shapiro_p"]["paired_differences"], "met" if c["original_three_condition_rule"]["all_three"] else "not met"))
    L.append("")
    L.append("== crossover (1280 falls to or below 640): %d/%d seeds, locations %s, mean %s, SD %s" % (
        cross["n_with_crossover"], cross["n_seeds"], ["%.2f" % x if x is not None else "none" for x in cross["locations_pct"]],
        "%.2f%%" % cross["mean_pct"] if cross["mean_pct"] is not None else "n/a",
        "%.2f%%" % cross["sd_pct"] if cross["sd_pct"] is not None else "n/a"))
    L.append("== small-scale FP (640 -> 1280): %s; change %s; positive %d, negative %d, undefined %d" % (
        {s: v for s, v in cross["fp_small_counts"].items()}, ["%+.0f%%" % c if c is not None else "undef" for c in cross["fp_small_change_pct"]],
        cross["fp_n_positive"], cross["fp_n_negative"], cross["fp_n_undefined"]))
    L.append("== COCO AP (x100): 640 mean, 1280 mean, paired diff [t CI], Welch CI")
    for k in AP_METRICS:
        a = ap_res[k]
        L.append("  %-5s %.2f  %.2f  %s [%s]  Welch [%s]" % (
            k, a["reference_mean"], a["candidate_mean"], f2(a["mean"]),
            "%s, %s" % (f2(a["ci95"][0]), f2(a["ci95"][1])) if a["ci95"] else "SD 0",
            "%s, %s" % (f2(a["welch"]["ci95"][0]), f2(a["welch"]["ci95"][1])) if a["welch"]["ci95"] else "SD 0"))
    for cfg in (cfg_ref, cfg_cand):
        t = cost["timing_ms"][cfg]
        mc = cost["model_cost"][cfg]
        L.append("== cost %-11s imgsz %s: inference %.2f ms/image (pre %.2f, post %.2f)%s" % (
            cfg, t["imgsz"], t["inference"]["mean_of_model_means"], t["preprocess"]["mean_of_model_means"],
            t["postprocess"]["mean_of_model_means"],
            "" if not mc else "; params %d (fused %d); GFLOPs unfused %s, fused %s" % (
                mc["params_unfused"]["total"], mc["params_fused_total"],
                {k: round(v, 2) for k, v in mc["gflops_unfused"].items()}, {k: round(v, 2) for k, v in mc["gflops_fused"].items()})))
    L.append("")
    L.append("== SECTION 7 OUTCOME (small stratum, paired primary): %s; paired CI [%s, %s]; Welch alongside: %s" % (
        small["verdict_primary_paired"], f2(small["paired"]["ci95"][0]), f2(small["paired"]["ci95"][1]), small["verdict_welch_alongside"]))
    L.append("   " + rec["section7_outcome"]["wording"])
    txt = "\n".join(L) + "\n"
    with open(os.path.join(work, "positive_control.txt"), "w") as f:
        f.write(txt)
    os.rename(work, args.out)
    print(txt)
    print("written:", os.path.join(args.out, "positive_control.json"))


if __name__ == "__main__":
    main()
