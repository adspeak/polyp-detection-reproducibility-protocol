"""Seed subsets (Paper3 revision analysis addendum v1.0, Section 10.4; plan A3).

For k = 3-8, every one of the C(8, k) seed subsets is applied to both configurations of a
comparison (the same seeds for the reference B and the candidate C), and the Section 5
decisions are recomputed on the subset: the graded verdict (Welch 95% CI, SD_pooled), the
original three-condition rule (Delta > 0, paired t two-sided p < 0.05, Delta > SD_pooled),
and, beside them, the graded verdict with the paired-t 95% CI in place of the Welch CI.
Comparisons: original batch and replication (+P2 minus baseline; Welch primary) and positive
control (1280 minus 640; paired primary, Section 7). All three strata, small first.

Inputs are the per-seed values (pp) already stored by the A1 and G3 analyses
(replication_analysis.json, positive_control.json); nothing is recomputed from caches.
The statistics come from welch_ci() and verdict() of analysis_replication.py and scipy's
ttest_rel, as in Section 5.

Built-in check: at k = 8 (one subset, all seeds) every verdict and original-rule decision must
equal the one stored by A1 / G3, or the script stops. CPU only.
"""

import argparse
import hashlib
import importlib.util
import itertools
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
KS = (3, 4, 5, 6, 7, 8)
SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]


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


def decisions(rep, B, C):
    d, se, nu, t, ci = rep.welch_ci(B, C)
    sb, sc = float(np.std(B, ddof=1)), float(np.std(C, ddof=1))
    sdp = math.sqrt((sb ** 2 + sc ** 2) / 2)
    diff = C - B
    n = len(diff)
    md, sdd = float(diff.mean()), float(diff.std(ddof=1))
    tp = stats.t.ppf(0.975, n - 1)
    plo, phi = md - tp * sdd / math.sqrt(n), md + tp * sdd / math.sqrt(n)
    p = float(stats.ttest_rel(C, B).pvalue)
    return {"delta": d, "sd_pooled": sdp, "welch_ci": [ci[0], ci[1]], "paired_ci": [plo, phi], "paired_p": p,
            "verdict_welch": rep.verdict(ci[0], ci[1], d, sdp), "verdict_paired": rep.verdict(plo, phi, d, sdp),
            "original_rule": bool(d > 0 and p < 0.05 and d > sdp)}


def summarise(rep, B, C):
    B, C = np.asarray(B, float), np.asarray(C, float)
    out = {}
    for k in KS:
        rows = [decisions(rep, B[list(ix)], C[list(ix)]) for ix in itertools.combinations(range(len(B)), k)]
        deltas = np.array([r["delta"] for r in rows])
        n = len(rows)
        cnt = lambda key: {v: sum(1 for r in rows if r[key] == v) for v in sorted(set(r[key] for r in rows))}
        out[str(k)] = {
            "n_subsets": n,
            "verdict_welch_counts": cnt("verdict_welch"),
            "verdict_paired_counts": cnt("verdict_paired"),
            "original_rule_met": sum(r["original_rule"] for r in rows),
            "delta_gt_0": int(np.sum(deltas > 0)),
            "delta_gt_sd_pooled": sum(1 for r in rows if r["delta"] > r["sd_pooled"]),
            "paired_p_lt_0.05": sum(1 for r in rows if r["paired_p"] < 0.05),
            "undefined_statistics": sum(1 for r in rows if not all(np.isfinite(r["welch_ci"] + r["paired_ci"] + [r["paired_p"]]))),
            "delta_min": float(deltas.min()), "delta_median": float(np.median(deltas)), "delta_max": float(deltas.max()),
        }
        if k == 8:
            out["8"]["all_seeds"] = rows[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", required=True, help="replication_analysis.json (A1)")
    ap.add_argument("--positive-control", required=True, help="positive_control.json (G3)")
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)), help="Folder with analysis_replication.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    rj, pj = json.load(open(args.replication)), json.load(open(args.positive_control))

    comps = {}
    for batch in ("original", "replication"):
        comps[batch] = {"primary": "welch", "label": "+P2 minus baseline (%s batch)" % batch,
                        "data": {k: (rj["per_seed_pp"][batch][k]["B"], rj["per_seed_pp"][batch][k]["C"]) for k in STRATA},
                        "stored": {k: {"verdict_welch": rj["comparisons"][batch][k]["verdict"],
                                       "original_rule": rj["comparisons"][batch][k]["original_three_condition_rule"]["all_three"]}
                                   for k in STRATA}}
    comps["positive_control"] = {"primary": "paired", "label": "1280 minus 640 (positive control)",
                                 "data": {k: (pj["per_seed_pp"][k]["reference_640"], pj["per_seed_pp"][k]["candidate_1280"]) for k in STRATA},
                                 "stored": {k: {"verdict_welch": pj["comparisons"][k]["verdict_welch_alongside"],
                                                "verdict_paired": pj["comparisons"][k]["verdict_primary_paired"],
                                                "original_rule": pj["comparisons"][k]["original_three_condition_rule"]["all_three"]}
                                            for k in STRATA}}
    res, problems = {}, []
    for name, c in comps.items():
        res[name] = {"primary": c["primary"], "label": c["label"], "strata": {}}
        for k in STRATA:
            B, C = c["data"][k]
            if not (len(B) == 8 and len(C) == 8):
                sys.exit("ERROR: %s %s does not have 8 seeds per configuration" % (name, k))
            s = summarise(rep, B, C)
            res[name]["strata"][k] = s
            full = s["8"]["all_seeds"]
            for key, val in c["stored"][k].items():
                if not full[key] == val:
                    problems.append("%s %s %s: k = 8 gives %s, stored %s" % (name, k, key, full[key], val))
    if problems:
        sys.exit("ERROR: built-in check failed:\n  " + "\n  ".join(problems))

    rec = {"analysis": "seed subsets, addendum v1.0 Section 10.4 (plan A3)",
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "settings": {"k": list(KS), "subsets": "all C(8, k), same seeds for both configurations",
                        "verdict": "Section 5 graded verdict, Welch 95% CI; paired-t CI version beside it",
                        "original_rule": "Delta > 0, paired t two-sided p < 0.05, Delta > SD_pooled", "units": "pp of recall"},
           "inputs_sha256": {"replication_analysis.json": sha256_file(args.replication),
                             "positive_control.json": sha256_file(args.positive_control),
                             "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py"))},
           "builtin_check": "k = 8 verdicts and original-rule decisions equal those stored by A1 and G3",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__),
           "results": res}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "seed_subsets.json"), "w") as f:
        json.dump(rec, f, indent=2)

    short = {"improvement, larger than run-to-run variation": "impr>SD", "improvement, smaller than run-to-run variation": "impr<=SD",
             "no evidence of a mean difference": "none", "deterioration": "deter"}
    L = ["Seed subsets (addendum 10.4): all C(8, k) subsets, same seeds for both configurations; counts out of n subsets",
         "built-in check: k = 8 reproduces the stored A1 / G3 verdicts", ""]
    for name, r in res.items():
        L.append("== %s; primary %s" % (r["label"], r["primary"]))
        for k in STRATA:
            L.append("  %s" % k)
            for kk in KS:
                s = r["strata"][k][str(kk)]
                fmt = lambda d: ", ".join("%s %d" % (short[v], n) for v, n in d.items())
                L.append("    k=%d n=%3d | Welch: %s | paired: %s | original rule %d | Delta>0 %d, Delta>SD_pooled %d, paired p<0.05 %d | Delta %.2f / %.2f / %.2f (min / median / max)" % (
                    kk, s["n_subsets"], fmt(s["verdict_welch_counts"]), fmt(s["verdict_paired_counts"]), s["original_rule_met"],
                    s["delta_gt_0"], s["delta_gt_sd_pooled"], s["paired_p_lt_0.05"], s["delta_min"], s["delta_median"], s["delta_max"])
                         + ("" if not s["undefined_statistics"] else " | UNDEFINED (zero variance) %d" % s["undefined_statistics"]))
        L.append("")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "seed_subsets.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "seed_subsets.json"))


if __name__ == "__main__":
    main()
