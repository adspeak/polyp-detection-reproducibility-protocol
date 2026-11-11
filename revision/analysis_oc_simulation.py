"""Operating characteristics of the decision rule (Paper3 revision analysis addendum v1.0, 10.5).

Per-seed pairs (B, C) are drawn from a bivariate normal distribution: B has mean 0 and C has
mean Delta; their SDs are the replication small-stratum SDs (B = baseline, C = +P2, read from
replication_analysis.json) and their correlation is r. Grid: Delta in {0, 0.5, 1, 1.5, 2, 3, 4,
5} pp, r in {0, 0.5, 0.9}, S in {3, 5, 8, 12} seeds; 10,000 replicates per cell. Random seed
20261106: every cell uses its own numpy default_rng(20261106) (re-seeded per cell, as the
bootstrap of Section 5), so each cell is reproducible on its own. Draws: z ~ N(0, 1) of shape
(replicates, S, 2); B = s_B z1, C = Delta + s_C (r z1 + sqrt(1 - r^2) z2).

For every cell: the probability of each graded verdict (Welch 95% CI, Section 5), of the
original three-condition rule (Delta > 0, paired t p < 0.05, Delta > SD_pooled), and of a
paired-t 95% CI excluding 0 (reported as lower bound > 0 and upper bound < 0 separately).
Highlighted cases (10.5): zero effect (Delta = 0); small and stable (Delta = 1.5, r = 0.9);
large but variable (Delta = 4 with both SDs doubled; computed for every r and S).
Added after the positive-control results (deviation log, 5 Nov 2026): one cell with the observed
positive-control values (Delta, s_640, s_1280, r from positive_control.json), S = 8, same
replicates and seed. The positive control's primary analysis is paired, so its detection
probability is the paired-t CI lower bound > 0; the Welch graded verdict is reported beside it.

Built-in check: for 6 cells, all 10,000 replicates are recomputed one by one with welch_ci() and
verdict() of analysis_replication.py and scipy's ttest_rel; every verdict, every original-rule
decision and every paired-CI decision must agree with the vectorised computation, or the script
stops. CPU only.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy import stats

SEED, REPS = 20261106, 10000
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
RS = [0.0, 0.5, 0.9]
SS = [3, 5, 8, 12]
V_LARGE = "improvement, larger than run-to-run variation"
V_SMALL = "improvement, smaller than run-to-run variation"
V_NONE = "no evidence of a mean difference"
V_DET = "deterioration"
VERDICTS = [V_LARGE, V_SMALL, V_NONE, V_DET]


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


def draw(delta, sb, sc, r, S, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((reps, S, 2))
    B = sb * z[..., 0]
    C = delta + sc * (r * z[..., 0] + math.sqrt(1.0 - r * r) * z[..., 1])
    return B, C


def decide(B, C):
    """Vectorised Section 5 decisions for arrays of shape (replicates, S)."""
    S = B.shape[1]
    mb, mc = B.mean(axis=1), C.mean(axis=1)
    vb, vc = B.var(axis=1, ddof=1), C.var(axis=1, ddof=1)
    delta = mc - mb
    se = np.sqrt(vb / S + vc / S)
    nu = (vb / S + vc / S) ** 2 / ((vb / S) ** 2 / (S - 1) + (vc / S) ** 2 / (S - 1))
    t = stats.t.ppf(0.975, nu)
    lo, hi = delta - t * se, delta + t * se
    sdp = np.sqrt((vb + vc) / 2)
    verdict = np.where(lo > 0, np.where(delta > sdp, 0, 1), np.where(hi < 0, 3, 2))
    d = C - B
    md, sdd = d.mean(axis=1), d.std(axis=1, ddof=1)
    tp = stats.t.ppf(0.975, S - 1)
    plo, phi = md - tp * sdd / math.sqrt(S), md + tp * sdd / math.sqrt(S)
    p_paired = 2 * stats.t.sf(np.abs(md / (sdd / math.sqrt(S))), S - 1)
    original = (delta > 0) & (p_paired < 0.05) & (delta > sdp)
    return verdict, original, plo > 0, phi < 0


def cell(delta, sb, sc, r, S):
    B, C = draw(delta, sb, sc, r, S)
    v, orig, plo, phi = decide(B, C)
    out = {"delta": delta, "sd_B": sb, "sd_C": sc, "r": r, "S": S, "replicates": REPS, "seed": SEED}
    for i, name in enumerate(VERDICTS):
        out["P_" + name] = float(np.mean(v == i))
    out["P_welch_ci_lower_gt_0"] = out["P_" + V_LARGE] + out["P_" + V_SMALL]
    out["P_original_rule"] = float(np.mean(orig))
    out["P_paired_ci_lower_gt_0"] = float(np.mean(plo))
    out["P_paired_ci_upper_lt_0"] = float(np.mean(phi))
    out["P_paired_ci_excludes_0"] = float(np.mean(plo | phi))
    return out


def check_cell(rep, delta, sb, sc, r, S):
    """Recompute every replicate with the analysis code of Section 5; return the number of disagreements."""
    B, C = draw(delta, sb, sc, r, S)
    v, orig, plo, phi = decide(B, C)
    bad = 0
    for i in range(B.shape[0]):
        b, c = B[i], C[i]
        d, se, nu, t, ci = rep.welch_ci(b, c)
        sdp = math.sqrt((np.std(b, ddof=1) ** 2 + np.std(c, ddof=1) ** 2) / 2)
        if not rep.verdict(ci[0], ci[1], d, sdp) == VERDICTS[v[i]]:
            bad += 1
            continue
        p = float(stats.ttest_rel(c, b).pvalue)
        if not bool(d > 0 and p < 0.05 and d > sdp) == bool(orig[i]):
            bad += 1
            continue
        dd = c - b
        md, sd_d = float(dd.mean()), float(dd.std(ddof=1))
        tp = stats.t.ppf(0.975, S - 1)
        if not (md - tp * sd_d / math.sqrt(S) > 0) == bool(plo[i]) or not (md + tp * sd_d / math.sqrt(S) < 0) == bool(phi[i]):
            bad += 1
    return bad


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
    rs = rj["comparisons"]["replication"]["small"]
    sb, sc = float(rs["sd_reference"]), float(rs["sd_candidate"])
    ps = pj["comparisons"]["small"]
    pc = {"delta": float(ps["welch"]["delta"]), "sd_B": float(ps["sd_reference"]), "sd_C": float(ps["sd_candidate"]),
          "r": float(ps["paired"]["r_between_configurations"]), "S": 8}
    if not (ps["n_reference"] == 8 and ps["n_candidate"] == 8):
        sys.exit("ERROR: positive control is not 8 against 8 seeds")

    checks = [(0.0, 0, 3), (1.5, 0.9, 8), (3.0, 0.5, 5), (5.0, 0.0, 12), (0.5, 0.9, 3), (2.0, 0.5, 8)]
    nbad = {"%s_r%s_S%d" % c: check_cell(rep, c[0], sb, sc, c[1], c[2]) for c in checks}
    if any(nbad.values()):
        sys.exit("ERROR: vectorised decisions disagree with the Section 5 code: %s" % nbad)

    grid = [cell(d, sb, sc, r, S) for d in DELTAS for r in RS for S in SS]
    large_variable = [cell(4.0, 2 * sb, 2 * sc, r, S) for r in RS for S in SS]
    pc_cell = cell(pc["delta"], pc["sd_B"], pc["sd_C"], pc["r"], pc["S"])

    rec = {
        "analysis": "operating characteristics of the rule, addendum v1.0 Section 10.5",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"replicates_per_cell": REPS, "seed": SEED, "seeding": "numpy default_rng(20261106) re-seeded for every cell",
                     "draws": "z ~ N(0,1), shape (replicates, S, 2); B = s_B z1; C = Delta + s_C (r z1 + sqrt(1-r^2) z2)",
                     "verdict": "Welch 95% CI and SD_pooled = sqrt((s_B^2 + s_C^2)/2), Section 5",
                     "original_rule": "Delta > 0, paired t two-sided p < 0.05, Delta > SD_pooled",
                     "paired_ci": "paired t 95% CI", "monte_carlo_se_max": 0.5 / math.sqrt(REPS),
                     "grid": {"delta_pp": DELTAS, "r": RS, "S": SS}},
        "replication_small_sds": {"sd_B_baseline": sb, "sd_C_p2": sc},
        "positive_control_observed": pc,
        "inputs_sha256": {"replication_analysis.json": sha256_file(args.replication),
                          "positive_control.json": sha256_file(args.positive_control),
                          "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py"))},
        "builtin_check_disagreements": nbad,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "this_script_sha256": sha256_file(__file__),
        "grid": grid,
        "highlight_large_but_variable_sds_doubled": large_variable,
        "added_positive_control_cell": pc_cell,
    }
    os.makedirs(args.out)
    with open(os.path.join(args.out, "oc_simulation.json"), "w") as f:
        json.dump(rec, f, indent=2)
    import csv
    keys = list(grid[0].keys())
    with open(os.path.join(args.out, "oc_simulation_cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set"] + keys)
        for name, rows in (("grid", grid), ("large_but_variable", large_variable), ("positive_control_observed", [pc_cell])):
            for row in rows:
                w.writerow([name] + [row[k] for k in keys])

    g = {(c["delta"], c["r"], c["S"]): c for c in grid}
    L = ["Operating characteristics (addendum 10.5): %d replicates per cell, seed %d re-seeded per cell; Monte Carlo SE <= %.3f" % (
        REPS, SEED, 0.5 / math.sqrt(REPS)),
         "replication small-stratum SDs: s_B (baseline) %.3f, s_C (+P2) %.3f pp; built-in check: 0 disagreements in %d x %d replicates" % (
             sb, sc, len(checks), REPS), ""]
    for title, key in (("P(Welch CI lower bound > 0) = P(evidence of improvement, either size label)", "P_welch_ci_lower_gt_0"),
                       ("P(original three-condition rule met)", "P_original_rule"),
                       ("P(paired-t CI lower bound > 0)", "P_paired_ci_lower_gt_0")):
        L.append("== " + title)
        L.append("   r    S  " + "".join("  D=%-4s" % ("%g" % d) for d in DELTAS))
        for r in RS:
            for S in SS:
                L.append("  %.1f %3d  " % (r, S) + "".join("  %.3f " % g[(d, r, S)][key] for d in DELTAS))
        L.append("")
    L.append("== zero effect (Delta = 0): P(Welch CI lower > 0), P(Welch CI upper < 0), P(paired CI excludes 0), P(original rule)")
    for r in RS:
        for S in SS:
            c = g[(0.0, r, S)]
            L.append("  r %.1f S %2d: %.3f  %.3f  %.3f  %.3f" % (r, S, c["P_welch_ci_lower_gt_0"], c["P_" + V_DET],
                                                              c["P_paired_ci_excludes_0"], c["P_original_rule"]))
    L.append("== small and stable (Delta = 1.5, r = 0.9): graded verdict probabilities [larger, smaller, none, deterioration]; original rule; paired CI lower > 0")
    for S in SS:
        c = g[(1.5, 0.9, S)]
        L.append("  S %2d: [%s]  original %.3f  paired %.3f" % (S, ", ".join("%.3f" % c["P_" + v] for v in VERDICTS),
                                                            c["P_original_rule"], c["P_paired_ci_lower_gt_0"]))
    L.append("== large but variable (Delta = 4, SDs doubled: %.2f, %.2f): [larger, smaller, none, deterioration]; original; paired lower > 0" % (2 * sb, 2 * sc))
    for c in large_variable:
        L.append("  r %.1f S %2d: [%s]  original %.3f  paired %.3f" % (c["r"], c["S"], ", ".join("%.3f" % c["P_" + v] for v in VERDICTS),
                                                                   c["P_original_rule"], c["P_paired_ci_lower_gt_0"]))
    L.append("== ADDED (deviation log): observed positive control, Delta %.3f, s_640 %.3f, s_1280 %.3f, r %.3f, S 8" % (
        pc["delta"], pc["sd_B"], pc["sd_C"], pc["r"]))
    L.append("  paired-t CI lower > 0 (its primary analysis): %.3f; Welch graded [larger, smaller, none, deterioration]: [%s]; original rule %.3f" % (
        pc_cell["P_paired_ci_lower_gt_0"], ", ".join("%.3f" % pc_cell["P_" + v] for v in VERDICTS), pc_cell["P_original_rule"]))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "oc_simulation.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "oc_simulation.json"))


if __name__ == "__main__":
    main()
