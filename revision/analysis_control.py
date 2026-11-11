"""P2-P4 control configuration (Paper3 revision analysis addendum v1.0, Section 7) and the
main-text summary table.

Candidate: runs/control_seed42-49 (replication seed 42 and G5 seeds 43-49). Reference: the
eight replication baselines runs/baseline_seed42-49. Both trained and evaluated at 1280.
Recall per stratum, crossover and small-scale false positives come from the public
multiseed_eval.py and crossover_fp_analysis.py, imported unchanged (hashes checked), with
their YOLO replaced by the confidence-0.001 cache reader of check_public_vs_cache.py, as in the
G1 replay. A stand-in module named "ultralytics" provides that reader, so the public scripts'
unchanged "from ultralytics import YOLO" does not load torch (as in analysis_a11_v2.py).
Statistics: compare() of analysis_replication.py, Welch primary (Section 5); the crossover and
false-positive summary is crossover_summary() of the same file. Evaluated as a third
modification under the protocol, not as an ablation of the P2 head.

Main-text summary table (Section 7): one row per comparison (+P2 replication, +P2 original,
positive control, control configuration) with Delta in the small stratum, its Welch 95% CI,
the graded verdict, Delta / SD_pooled and the original-rule verdict. The first three rows are
read from replication_analysis.json and positive_control.json; for the positive control, whose
primary analysis is paired, the paired verdict is given beside the Welch one.

Built-in checks (the script stops on any failure): baseline recall per stratum and small FP
per seed equal the G1 replay of the +P2 comparison; seed 42 of the control (recall, crossover,
small FP) equals the G1 replay of the control comparison. CPU only.
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
EXP_CF = "c90558130a28f576f2b798b0cd12474fff40d4d568d25c4cccd959211d17acb2"
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


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder with check_public_vs_cache.py and analysis_replication.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--g1-replay", required=True)
    ap.add_argument("--replication", required=True, help="replication_analysis.json (A1)")
    ap.add_argument("--positive-control", required=True, help="positive_control.json (G3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    ms, cf = os.path.join(args.repo, "evaluation", "multiseed_eval.py"), os.path.join(args.repo, "evaluation", "crossover_fp_analysis.py")
    if not (sha256_file(ms) == EXP_MS and sha256_file(cf) == EXP_CF):
        sys.exit("ERROR: the public evaluation scripts differ from the published versions")
    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    stand_in = types.ModuleType("ultralytics")
    stand_in.YOLO = cpc.CachedYOLO
    sys.modules["ultralytics"] = stand_in
    dataset = os.path.realpath(args.dataset)

    recs = {}
    for js in sorted(glob.glob(os.path.join(args.cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        recs[r["checkpoint_sha256"]] = r
    plan = {}
    for side, pat in (("reference", "runs/baseline_seed{s}"), ("candidate", "runs/control_seed{s}")):
        plan[side] = []
        for s in SEEDS:
            rel = os.path.join(pat.format(s=s), "weights", "best.pt")
            r = recs.get(sha256_file(os.path.join(args.root, rel)))
            if r is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            if not (sha256_file(r["_npz"]) == r["npz_sha256"] and r["settings"]["imgsz"] == 1280
                    and int(r["train_args"]["seed"]) == int(s) and os.path.realpath(r["dataset"]) == dataset):
                sys.exit("ERROR: cache of %s does not match (hash, imgsz, seed or dataset)" % rel)
            plan[side].append({"seed": s, "checkpoint": rel, "tag": r["tag"], "npz_sha256": r["npz_sha256"]})
    cpc.CachedYOLO.index, cpc.CachedYOLO.split, cpc.CachedYOLO.dataset = recs, "test", dataset

    work = args.out + ".partial_%d" % os.getpid()
    os.makedirs(work)
    B = [p["checkpoint"] for p in plan["reference"]]
    C = [p["checkpoint"] for p in plan["candidate"]]
    common = ["--dataset", dataset, "--weights-dir", os.path.abspath(args.root), "--imgsz", "1280", "--conf", "0.25", "--iou-match", "0.5"]
    pub = {}
    for name, path, argv in (("multiseed", ms, common + ["--thresholds", os.path.abspath(args.thresholds), "--baseline"] + B + ["--candidate"] + C + ["--seeds"] + SEEDS),
                             ("crossover", cf, common + ["--baseline"] + B + ["--candidate"] + C + ["--seeds"] + SEEDS)):
        out = os.path.join(work, "%s_control.json" % name)
        mod = load_module(path, "public_" + name)
        if "torch" in sys.modules:
            sys.exit("ERROR: torch was imported")
        buf, old = io.StringIO(), sys.argv
        sys.argv = [path] + argv + ["--out", out]
        try:
            with contextlib.redirect_stdout(buf):
                mod.main()
        finally:
            sys.argv = old
        with open(out[:-5] + ".log", "w") as f:
            f.write(buf.getvalue())
        pub[name] = {"json": json.load(open(out)), "argv": argv + ["--out", out]}
        print("[public] %s replayed, peak memory %.0f MB" % (name, peak_mb()), file=sys.stderr, flush=True)
    pm, pc = pub["multiseed"]["json"], pub["crossover"]["json"]

    g1m = json.load(open(os.path.join(args.g1_replay, "multiseed_p2.json")))
    g1c = json.load(open(os.path.join(args.g1_replay, "crossover_p2.json")))
    g1km = json.load(open(os.path.join(args.g1_replay, "multiseed_control.json")))
    g1kc = json.load(open(os.path.join(args.g1_replay, "crossover_control.json")))
    problems = []
    for k in STRATA:
        if not pm["per_seed"]["baseline"][k] == g1m["per_seed"]["baseline"][k]:
            problems.append("baseline %s recall differs from the G1 replay" % k)
        if not pm["per_seed"]["candidate"][k][0] == g1km["per_seed"]["candidate"][k][0]:
            problems.append("control seed 42 %s recall differs from the G1 replay" % k)
    for s in SEEDS:
        if not pc["per_seed"][s]["fp_small_baseline"] == g1c["per_seed"][s]["fp_small_baseline"]:
            problems.append("seed %s baseline small FP differs from G1" % s)
    for key in ("crossover_relative_area_pct", "fp_small_baseline", "fp_small_candidate", "fp_small_change_pct"):
        if not pc["per_seed"]["42"][key] == g1kc["per_seed"]["42"][key]:
            problems.append("control seed 42 %s differs from G1" % key)
    if problems:
        sys.exit("ERROR: built-in checks failed (outputs kept in %s):\n  %s" % (work, "\n  ".join(problems)))

    comp = {k: rep.compare([100 * x for x in pm["per_seed"]["baseline"][k]], [100 * x for x in pm["per_seed"]["candidate"][k]]) for k in STRATA}
    cross = rep.crossover_summary(pc, True)

    rj, pj = json.load(open(args.replication)), json.load(open(args.positive_control))

    def row(label, c, verdict, paired_verdict=None):
        w = c["welch"]
        return {"comparison": label, "delta_small": w["delta"], "welch_ci95": w["ci95"], "graded_verdict_welch": verdict,
                "delta_over_sd_pooled": w["delta"] / c["sd_pooled"], "original_rule": "met" if c["original_three_condition_rule"]["all_three"] else "not met",
                "paired_verdict_primary_for_positive_control": paired_verdict}
    table = [row("+P2, replication batch", rj["comparisons"]["replication"]["small"], rj["comparisons"]["replication"]["small"]["verdict"]),
             row("+P2, original batch", rj["comparisons"]["original"]["small"], rj["comparisons"]["original"]["small"]["verdict"]),
             row("positive control (1280 minus 640)", pj["comparisons"]["small"], pj["comparisons"]["small"]["verdict_welch_alongside"],
                 pj["comparisons"]["small"]["verdict_primary_paired"]),
             row("control configuration (P2-P4 minus baseline)", comp["small"], comp["small"]["verdict"])]

    rec = {"analysis": "P2-P4 control configuration (addendum v1.0 Section 7) and main-text summary table",
           "run_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
           "settings": {"conf": 0.25, "iou_match": 0.5, "imgsz": 1280, "units": "pp of recall", "primary": "Welch"},
           "inputs_sha256": {"multiseed_eval.py": EXP_MS, "crossover_fp_analysis.py": EXP_CF,
                             "check_public_vs_cache.py": sha256_file(os.path.join(args.tools, "check_public_vs_cache.py")),
                             "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py")),
                             "thresholds": sha256_file(args.thresholds), "replication_analysis.json": sha256_file(args.replication),
                             "positive_control.json": sha256_file(args.positive_control),
                             "g1_check_record": sha256_file(os.path.join(args.g1_replay, "check_record.json"))},
           "builtin_checks": "baseline recall and small FP equal the G1 +P2 replay; control seed 42 equals the G1 control replay",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__), "peak_memory_mb": peak_mb(),
           "per_seed_pp": {k: {"baseline": [100 * x for x in pm["per_seed"]["baseline"][k]], "control": [100 * x for x in pm["per_seed"]["candidate"][k]]} for k in STRATA},
           "comparisons": comp, "crossover_and_fp": cross, "main_text_summary_table": table,
           "public_outputs": {n: v["argv"] for n, v in pub.items()}}
    with open(os.path.join(work, "control.json"), "w") as f:
        json.dump(rec, f, indent=2)

    f2 = lambda x: "%+.2f" % x
    L = ["Control configuration (addendum Section 7): P2-P4 minus baseline, 8 seeds, pp of recall; Welch primary",
         "built-in checks passed: baseline equals the G1 +P2 replay; control seed 42 equals the G1 control replay", ""]
    L.append("per seed, small stratum: " + ", ".join("%s %.1f->%.1f (%s)" % (s, b, c, f2(c - b)) for s, b, c in zip(
        SEEDS, rec["per_seed_pp"]["small"]["baseline"], rec["per_seed_pp"]["small"]["control"])))
    for k in STRATA:
        c = comp[k]
        w, p = c["welch"], c["paired"]
        L.append("  %-6s baseline %.2f (SD %.2f)  control %.2f (SD %.2f)  delta %s  Welch CI [%s, %s] df %.1f -> %s" % (
            k, c["mean_reference"], c["sd_reference"], c["mean_candidate"], c["sd_candidate"], f2(w["delta"]),
            f2(w["ci95"][0]), f2(w["ci95"][1]), w["df"], c["verdict"]))
        L.append("         SD_pooled %.2f, delta/SD_pooled %.2f, P(one control run > one baseline run) %.2f; paired CI [%s, %s], t p %.4f, r %.2f" % (
            c["sd_pooled"], w["delta"] / c["sd_pooled"], c["prob_one_candidate_run_beats_one_reference_run_normal_model"],
            f2(p["ci95"][0]), f2(p["ci95"][1]), p["t_test_p"], p["r_between_configurations"]))
        L.append("         sign-flip p %.4f, Wilcoxon p %s, permutation p %.4f, bootstrap CI [%s, %s], Shapiro p %.2f / %.2f; original rule %s" % (
            p["exact_sign_flip_p"], "%.4f" % p["exact_wilcoxon"]["p"] if p["exact_wilcoxon"]["p"] is not None else "n/a",
            c["exact_two_sample_permutation_p"], f2(c["bootstrap_percentile_ci95"][0]), f2(c["bootstrap_percentile_ci95"][1]),
            c["shapiro_p"]["reference"], c["shapiro_p"]["candidate"], "met" if c["original_three_condition_rule"]["all_three"] else "not met"))
    L.append("== crossover (control falls to or below baseline): %d/%d seeds, locations %s, mean %s, SD %s" % (
        cross["n_with_crossover"], cross["n_seeds"], ["%.2f" % x if x is not None else "none" for x in cross["locations_pct"]],
        "%.2f%%" % cross["mean_pct"] if cross["mean_pct"] is not None else "n/a", "%.2f%%" % cross["sd_pct"] if cross["sd_pct"] is not None else "n/a"))
    L.append("== small-scale FP (baseline -> control): %s; positive %d, negative %d, undefined %d" % (
        cross["fp_small_counts"], cross["fp_n_positive"], cross["fp_n_negative"], cross["fp_n_undefined"]))
    L.append("")
    L.append("== MAIN-TEXT SUMMARY TABLE (small stratum, pp)")
    for t in table:
        L.append("  %-45s delta %s  Welch CI [%s, %s]  %s  delta/SD_pooled %.2f  original rule %s%s" % (
            t["comparison"], f2(t["delta_small"]), f2(t["welch_ci95"][0]), f2(t["welch_ci95"][1]), t["graded_verdict_welch"],
            t["delta_over_sd_pooled"], t["original_rule"],
            "" if t["paired_verdict_primary_for_positive_control"] is None else "  [paired, its primary: %s]" % t["paired_verdict_primary_for_positive_control"]))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(work, "control.txt"), "w") as f:
        f.write(txt)
    os.rename(work, args.out)
    print(txt)
    print("written:", os.path.join(args.out, "control.json"))
    print("peak memory %.0f MB" % peak_mb())


if __name__ == "__main__":
    main()
