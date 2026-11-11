"""Replication analysis (Paper3 revision analysis addendum v1.0, Sections 5 and 6).

Inputs
  replication: the multiseed and crossover JSON files that the public scripts produced
               from the confidence-0.001 cache (public_check/<stamp>/replay/), whose
               equality with the real runs is recorded in replay/check_record.json;
  original:    the recorded per-seed values of the original batch in the public
               repository (results/multiseed_8seeds.json, results/crossover_fp_8seeds.json).
All recall values are converted to percentage points. Nothing here is chosen by its result:
every statistic, interval and rule below is the one fixed in the addendum.

Bootstrap: percentile CI, 10,000 resamples of seeds within each group, numpy
default_rng(20261106), re-seeded for every comparison so that each one is reproducible
on its own. Exact Wilcoxon signed-rank: zero differences dropped, mid-ranks for ties,
all sign patterns enumerated; two-sided p = share of patterns at least as far from the
null mean as observed. All exact p values use a tolerance of 1e-9 for "at least as extreme".
"""

import argparse
import hashlib
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
SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
BOOT_N, BOOT_SEED, TOL = 10000, 20261106, 1e-9


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def welch_ci(B, C):
    n, m = len(B), len(C)
    vb, vc = np.var(B, ddof=1), np.var(C, ddof=1)
    se = math.sqrt(vb / n + vc / m)
    nu = (vb / n + vc / m) ** 2 / ((vb / n) ** 2 / (n - 1) + (vc / m) ** 2 / (m - 1))
    t = stats.t.ppf(0.975, nu)
    d = float(np.mean(C) - np.mean(B))
    return d, se, nu, t, (d - t * se, d + t * se)


def verdict(lo, hi, delta, sd_pooled):
    if lo > 0:
        return "improvement, larger than run-to-run variation" if delta > sd_pooled else \
            "improvement, smaller than run-to-run variation"
    if hi < 0:
        return "deterioration"
    return "no evidence of a mean difference"


def signflip_p(d):
    obs = abs(np.mean(d))
    cnt = tot = 0
    for s in itertools.product((1.0, -1.0), repeat=len(d)):
        tot += 1
        cnt += abs(np.mean(np.array(s) * d)) >= obs - TOL
    return cnt / tot


def wilcoxon_exact(d):
    nz = d[np.abs(d) > 0]
    if len(nz) == 0:
        return {"p": None, "n_used": 0, "n_zero": int(len(d))}
    r = stats.rankdata(np.abs(nz))
    tplus = float(r[nz > 0].sum())
    mu = float(r.sum()) / 2
    obs = abs(tplus - mu)
    cnt = tot = 0
    for s in itertools.product((True, False), repeat=len(nz)):
        tot += 1
        cnt += abs(float(r[np.array(s)].sum()) - mu) >= obs - TOL
    return {"p": cnt / tot, "T_plus": tplus, "n_used": int(len(nz)), "n_zero": int(len(d) - len(nz))}


def perm2_p(B, C):
    allv = np.concatenate([B, C])
    n, m = len(B), len(C)
    obs = abs(np.mean(C) - np.mean(B))
    tot = allv.sum()
    cnt = k = 0
    for idx in itertools.combinations(range(n + m), m):
        k += 1
        sc = allv[list(idx)].sum()
        cnt += abs(sc / m - (tot - sc) / n) >= obs - TOL
    return cnt / k


def bootstrap_ci(B, C):
    rng = np.random.default_rng(BOOT_SEED)
    n, m = len(B), len(C)
    out = np.empty(BOOT_N)
    for i in range(BOOT_N):
        out[i] = C[rng.integers(0, m, m)].mean() - B[rng.integers(0, n, n)].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def compare(B, C):
    """All statistics of Section 5 for one comparison (B = reference, C = candidate), in pp."""
    B, C = np.asarray(B, float), np.asarray(C, float)
    n, m = len(B), len(C)
    delta, se, nu, t, ci = welch_ci(B, C)
    sb, sc = float(np.std(B, ddof=1)), float(np.std(C, ddof=1))
    sdp = math.sqrt((sb ** 2 + sc ** 2) / 2)
    pi_half = t * math.sqrt((1 + 1 / n) * sb ** 2 + (1 + 1 / m) * sc ** 2)
    out = {
        "n_reference": n, "n_candidate": m,
        "mean_reference": float(B.mean()), "mean_candidate": float(C.mean()),
        "sd_reference": sb, "sd_candidate": sc, "sd_pooled": sdp,
        "welch": {"delta": delta, "se": se, "df": nu, "t_crit": float(t), "ci95": [float(ci[0]), float(ci[1])]},
        "verdict": verdict(ci[0], ci[1], delta, sdp),
        "prob_one_candidate_run_beats_one_reference_run_normal_model": float(stats.norm.cdf(delta / (math.sqrt(2) * sdp))),
        "prediction_interval_welch": [delta - pi_half, delta + pi_half],
        "cohens_d": delta / sdp,
        "exact_two_sample_permutation_p": perm2_p(B, C),
        "bootstrap_percentile_ci95": bootstrap_ci(B, C),
        "shapiro_p": {"reference": float(stats.shapiro(B).pvalue), "candidate": float(stats.shapiro(C).pvalue)},
    }
    if n == m:
        d = C - B
        md, sd_d = float(d.mean()), float(d.std(ddof=1))
        se_d = sd_d / math.sqrt(n)
        tp = stats.t.ppf(0.975, n - 1)
        out["paired"] = {
            "per_seed_difference": [float(x) for x in d],
            "mean": md, "sd": sd_d, "se": se_d, "df": n - 1,
            "ci95": [md - tp * se_d, md + tp * se_d],
            "t_test_p": float(stats.ttest_rel(C, B).pvalue),
            "r_between_configurations": float(stats.pearsonr(B, C)[0]),
            "exact_sign_flip_p": signflip_p(d),
            "exact_wilcoxon": wilcoxon_exact(d),
            "d_z": md / sd_d,
            "prediction_interval": [md - tp * sd_d * math.sqrt(1 + 1 / n), md + tp * sd_d * math.sqrt(1 + 1 / n)],
        }
        out["shapiro_p"]["paired_differences"] = float(stats.shapiro(d).pvalue)
        out["original_three_condition_rule"] = {
            "delta_gt_0": delta > 0, "paired_t_p_lt_0.05": out["paired"]["t_test_p"] < 0.05,
            "delta_gt_sd_pooled": delta > sdp,
            "all_three": bool(delta > 0 and out["paired"]["t_test_p"] < 0.05 and delta > sdp)}
    return out


def satterthwaite(terms):
    """terms: list of (variance of a mean, its degrees of freedom) -> (se, df)."""
    v = sum(x for x, _ in terms)
    df = v ** 2 / sum(x ** 2 / k for x, k in terms)
    return math.sqrt(v), df


def batch_difference(rep, orig):
    Br, Cr, Bo, Co = [np.asarray(x, float) for x in (rep["B"], rep["C"], orig["B"], orig["C"])]
    D = float((Cr.mean() - Br.mean()) - (Co.mean() - Bo.mean()))
    res = {"D": D}
    se, df = satterthwaite([(np.var(x, ddof=1) / len(x), len(x) - 1) for x in (Br, Cr, Bo, Co)])
    t = stats.t.ppf(0.975, df)
    res["primary_four_independent_groups"] = {"se": se, "df": df, "ci95": [D - t * se, D + t * se]}
    do = Co - Bo
    se2, df2 = satterthwaite([(np.var(Br, ddof=1) / len(Br), len(Br) - 1), (np.var(Cr, ddof=1) / len(Cr), len(Cr) - 1),
                              (np.var(do, ddof=1) / len(do), len(do) - 1)])
    t2 = stats.t.ppf(0.975, df2)
    res["sensitivity_original_paired"] = {"se": se2, "df": df2, "ci95": [D - t2 * se2, D + t2 * se2]}
    return res


def crossover_summary(js, counts):
    cr = js.get("crossovers") or []
    valid = [c for c in cr if c is not None]
    out = {"n_seeds": len(cr), "n_with_crossover": len(valid), "locations_pct": cr,
           "mean_pct": float(np.mean(valid)) if valid else None,
           "sd_pct": float(np.std(valid, ddof=1)) if len(valid) > 1 else None}
    if counts:
        ch = []
        for s in SEEDS:
            p = js["per_seed"][s]
            fb, fc = p["fp_small_baseline"], p["fp_small_candidate"]
            ch.append(None if fb == 0 else (fc - fb) / fb * 100)
        out["fp_small_counts"] = {s: [js["per_seed"][s]["fp_small_baseline"], js["per_seed"][s]["fp_small_candidate"]] for s in SEEDS}
    else:
        ch = list(js.get("fp_changes") or [])
        out["fp_note"] = "recorded changes; the public code writes 0 when the baseline count is 0"
    defined = [c for c in ch if c is not None]
    out["fp_small_change_pct"] = ch
    out["fp_n_positive"] = sum(c > 0 for c in defined)
    out["fp_n_negative"] = sum(c < 0 for c in defined)
    out["fp_n_undefined"] = len(ch) - len(defined)
    out["fp_sign_changes_between_seeds"] = out["fp_n_positive"] > 0 and out["fp_n_negative"] > 0
    out["fp_mean_pct"] = float(np.mean(defined)) if defined else None
    out["fp_sd_pct"] = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-multiseed", required=True)
    ap.add_argument("--rep-crossover", required=True)
    ap.add_argument("--check-record", required=True)
    ap.add_argument("--orig-multiseed", required=True)
    ap.add_argument("--orig-crossover", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    chk = json.load(open(args.check_record))
    if not chk.get("all_identical"):
        sys.exit("ERROR: the cache check did not pass; no analysis")
    rm, rc = json.load(open(args.rep_multiseed)), json.load(open(args.rep_crossover))
    om, oc = json.load(open(args.orig_multiseed)), json.load(open(args.orig_crossover))
    if not (list(rm["seeds"]) == SEEDS and list(om["seeds"]) == SEEDS):
        sys.exit("ERROR: seeds differ from 42-49")
    batches = {
        "replication": {k: {"B": [100 * x for x in rm["per_seed"]["baseline"][k]],
                            "C": [100 * x for x in rm["per_seed"]["candidate"][k]]} for k in STRATA},
        "original": {k: {"B": [100 * x for x in om["data"]["baseline"][k]],
                         "C": [100 * x for x in om["data"]["p2"][k]]} for k in STRATA},
    }
    for b in batches.values():
        for k in STRATA:
            if not (len(b[k]["B"]) == 8 and len(b[k]["C"]) == 8):
                sys.exit("ERROR: expected 8 seeds per group")

    comp = {b: {k: compare(batches[b][k]["B"], batches[b][k]["C"]) for k in STRATA} for b in batches}
    dif = {k: batch_difference(batches["replication"][k], batches["original"][k]) for k in STRATA}
    means = {}
    for cfg, key in (("baseline", "B"), ("p2", "C")):
        means[cfg] = {}
        for k in STRATA:
            r, o = np.asarray(batches["replication"][k][key]), np.asarray(batches["original"][k][key])
            d, se, nu, t, ci = welch_ci(o, r)
            means[cfg][k] = {"replication_mean": float(r.mean()), "original_mean": float(o.mean()),
                             "replication_sd": float(r.std(ddof=1)), "original_sd": float(o.std(ddof=1)),
                             "replication_minus_original": d, "welch_ci95": [float(ci[0]), float(ci[1])]}
    agree = {k: comp["replication"][k]["verdict"] == comp["original"][k]["verdict"] for k in STRATA}

    lo, hi = comp["replication"]["small"]["welch"]["ci95"]
    if lo > 0:
        scen, crit = "B", "replication Welch 95% CI lower bound > 0 (small stratum)"
    elif hi < 0:
        scen, crit = "C", "replication Welch 95% CI upper bound < 0 (small stratum)"
    else:
        scen, crit = "A", "replication Welch 95% CI contains 0 (small stratum)"
    Dp = dif["small"]["primary_four_independent_groups"]["ci95"]
    scen_note = None
    if scen in ("B", "C"):
        scen_note = ("CI for D excludes 0: the batches differ, the original conclusion was not reproduced"
                     if (Dp[0] > 0 or Dp[1] < 0) else
                     "CI for D contains 0: the verdicts differ, but whether the effects differ remains uncertain")

    rec = {
        "analysis": "replication analysis, addendum v1.0 Sections 5-6",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"units": "percentage points of recall", "ci": "Welch 95%", "bootstrap": [BOOT_N, BOOT_SEED],
                     "exact_tolerance": TOL, "wilcoxon_zero_method": "drop zeros, mid-ranks"},
        "inputs": {p: sha256_file(p) for p in (args.rep_multiseed, args.rep_crossover, args.check_record,
                                                args.orig_multiseed, args.orig_crossover)},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "this_script_sha256": sha256_file(__file__),
        "per_seed_pp": batches,
        "comparisons": comp,
        "batch_difference_D": dif,
        "configuration_means_replication_vs_original": means,
        "graded_verdicts_agree": agree,
        "crossover_and_fp": {"replication": crossover_summary(rc, True), "original": crossover_summary(oc, False)},
        "scenario": {"letter": scen, "criterion": crit, "small_ci95": [lo, hi], "note": scen_note},
    }
    os.makedirs(args.out)
    with open(os.path.join(args.out, "replication_analysis.json"), "w") as f:
        json.dump(rec, f, indent=2)

    def f2(x):
        return "%+.2f" % x
    lines = ["Replication analysis (addendum v1.0, Sections 5-6); values in pp of recall", ""]
    for b in ("replication", "original"):
        lines.append("== %s batch: +P2 minus baseline, 8 seeds each" % b)
        for k in STRATA:
            c = comp[b][k]
            w, p = c["welch"], c["paired"]
            lines.append("  %-6s baseline %.2f (SD %.2f)  +P2 %.2f (SD %.2f)  delta %s  Welch CI [%s, %s] df %.1f  -> %s" % (
                k, c["mean_reference"], c["sd_reference"], c["mean_candidate"], c["sd_candidate"], f2(w["delta"]),
                f2(w["ci95"][0]), f2(w["ci95"][1]), w["df"], c["verdict"]))
            lines.append("         SD_pooled %.2f; P(one +P2 run > one baseline run) %.2f; prediction interval [%s, %s]" % (
                c["sd_pooled"], c["prob_one_candidate_run_beats_one_reference_run_normal_model"],
                f2(c["prediction_interval_welch"][0]), f2(c["prediction_interval_welch"][1])))
            lines.append("         paired: mean %s CI [%s, %s] t-test p %.3f, r %.2f, sign-flip p %.3f, Wilcoxon p %s; "
                         "permutation p %.3f; d %.2f, d_z %.2f; bootstrap CI [%s, %s]" % (
                             f2(p["mean"]), f2(p["ci95"][0]), f2(p["ci95"][1]), p["t_test_p"], p["r_between_configurations"],
                             p["exact_sign_flip_p"],
                             "%.3f" % p["exact_wilcoxon"]["p"] if p["exact_wilcoxon"]["p"] is not None else "n/a",
                             c["exact_two_sample_permutation_p"], c["cohens_d"], p["d_z"],
                             f2(c["bootstrap_percentile_ci95"][0]), f2(c["bootstrap_percentile_ci95"][1])))
            lines.append("         paired prediction interval [%s, %s]; Shapiro p: B %.2f, C %.2f, diff %.2f; "
                         "original 3-condition rule: %s" % (
                             f2(p["prediction_interval"][0]), f2(p["prediction_interval"][1]), c["shapiro_p"]["reference"],
                             c["shapiro_p"]["candidate"], c["shapiro_p"]["paired_differences"],
                             "met" if c["original_three_condition_rule"]["all_three"] else "not met"))
        lines.append("")
    lines.append("== D = delta(replication) - delta(original)")
    for k in STRATA:
        d = dif[k]
        a, s = d["primary_four_independent_groups"], d["sensitivity_original_paired"]
        lines.append("  %-6s D %s  primary CI [%s, %s] (df %.1f)  sensitivity CI [%s, %s] (df %.1f)" % (
            k, f2(d["D"]), f2(a["ci95"][0]), f2(a["ci95"][1]), a["df"], f2(s["ci95"][0]), f2(s["ci95"][1]), s["df"]))
    lines.append("")
    lines.append("== configuration means, replication minus original (Welch CI)")
    for cfg in ("baseline", "p2"):
        for k in STRATA:
            x = means[cfg][k]
            lines.append("  %-8s %-6s rep %.2f (SD %.2f)  orig %.2f (SD %.2f)  diff %s  CI [%s, %s]" % (
                cfg, k, x["replication_mean"], x["replication_sd"], x["original_mean"], x["original_sd"],
                f2(x["replication_minus_original"]), f2(x["welch_ci95"][0]), f2(x["welch_ci95"][1])))
    lines.append("  graded verdicts agree between batches: %s" % agree)
    lines.append("")
    for b in ("replication", "original"):
        x = rec["crossover_and_fp"][b]
        lines.append("== %s: crossovers %d/%d, mean %s, SD %s; small-FP change %s; positive %d, negative %d, undefined %d; sign changes: %s" % (
            b, x["n_with_crossover"], x["n_seeds"], "%.2f%%" % x["mean_pct"] if x["mean_pct"] is not None else "n/a",
            "%.2f%%" % x["sd_pct"] if x["sd_pct"] is not None else "n/a",
            ["%+.0f%%" % c if c is not None else "undef" for c in x["fp_small_change_pct"]],
            x["fp_n_positive"], x["fp_n_negative"], x["fp_n_undefined"], x["fp_sign_changes_between_seeds"]))
    lines.append("")
    lines.append("== SCENARIO %s: %s; small-stratum replication CI [%s, %s]%s" % (
        scen, crit, f2(lo), f2(hi), ("; " + scen_note) if scen_note else ""))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "replication_analysis.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "replication_analysis.json"))


if __name__ == "__main__":
    main()
