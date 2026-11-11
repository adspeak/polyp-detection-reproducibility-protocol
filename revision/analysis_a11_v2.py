"""Crossover definition and one-to-one matching (Paper3 revision analysis addendum v1.0,
Sections 10.1 and 10.2; plan A11). Replication batch, +P2 against baseline, seeds 42-49.

10.1 Crossover definition. Per-lesion records come from the public crossover_fp_analysis.py
(eval_full, sliding_recall and find_crossover imported unchanged, hash checked) with its YOLO
replaced by the cache reader of check_public_vs_cache.py, exactly as in the G1 replay, and the
settings stored in the G1 replay (grid, window, minimum lesions, confidence 0.25, IoU 0.5).
  * primary: the public rule, first crossing from candidate above to at or below baseline;
  * submitted-text definition: every crossing between consecutive defined points, in either
    direction (d0 > 0 and d1 <= 0, or d0 < 0 and d1 >= 0), located by the same linear
    interpolation; the crossing closest to (5/35)^2 = 2.04% relative area on the log scale;
  * per seed: number of crossings; whether any interpolation spans an undefined grid point;
    if so, the primary crossover is also recomputed using adjacent defined grid points only;
  * all 8 x 8 = 64 baseline-candidate combinations, beside the 8 same-seed pairs: share with a
    crossover, median, IQR, mean and SD of the locations (primary definition); share of
    positive small-scale false-positive changes (undefined when the baseline count is 0).
10.2 One-to-one matching. Cached predictions with confidence > 0.25 (deviation log entry 1;
no box has confidence exactly 0.25) are matched greedily in descending confidence (stable
order); each prediction takes the unmatched box with the highest IoU >= 0.5. Recall per stratum
(training-set terciles) and false positives (unmatched predictions; small-scale: predicted
relative area below the public script's p33 setting, 0.0187 in G1) are analysed with compare() of
analysis_replication.py (Section 5). Cache and ground-truth loaders and box_iou are those of
analysis_ensemble_v2.py, already validated against the public scripts in A5 (k = 1 check).

Built-in checks (the script stops on any failure):
  * primary crossover and small-scale FP of every seed equal the G1 replay;
  * coverage recall per stratum (any prediction with IoU >= 0.5) and coverage small-scale FP from
    the 10.2 code equal the G1 replay of multiseed_eval.py and crossover_fp_analysis.py;
  * one-to-one hits never exceed coverage hits, and one-to-one FP never falls below coverage FP.
CPU only.

v2 (6 Nov 2026): v1 was killed by the system (exit code 137) on the 1-core, no-GPU instance
before writing anything, most likely for memory: loading the public module imported
ultralytics and with it torch. v2 registers a stand-in module named "ultralytics" whose YOLO is
the cache reader before the public module is loaded, so its unchanged line
"from ultralytics import YOLO" binds the cache reader directly (the G1 replay replaced YOLO
the same way, after import). It also reads the caches with load_split_once(), the same logic as load_split() of
analysis_ensemble_v2.py except that each array is read from the npz file once per model:
load_split() indexes the lazily loaded npz inside the per-image loop, so every image slice
keeps its own full copy of the array (correct values, but memory grows with the number of
images). The built-in checks against the G1 replay cover this loader. Progress lines and the
peak memory are printed.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
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
DIM = (5.0 / 35.0) ** 2 * 100  # 2.04 %, the diminutive boundary in relative-area percent


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


def crossings(yb, yp, xs):
    """All crossings between consecutive defined points, either direction (submitted-text rule)."""
    diff = yp - yb
    idx = np.nonzero(~np.isnan(diff))[0]
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        d0, d1 = diff[a], diff[b]
        if (d0 > 0 and d1 <= 0) or (d0 < 0 and d1 >= 0):
            x = xs[a] + (xs[b] - xs[a]) * d0 / (d0 - d1)
            out.append({"x": float(x), "direction": "down" if d0 > 0 else "up", "spans_undefined": bool(b - a > 1)})
    return out


def primary_adjacent_only(yb, yp, xs):
    """The public rule restricted to pairs of adjacent grid points that are both defined."""
    diff = yp - yb
    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if np.isnan(d0) or np.isnan(d1):
            continue
        if d0 > 0 and d1 <= 0:
            return float(xs[i] + (xs[i + 1] - xs[i]) * d0 / (d0 - d1))
    return None


def primary_segment_spans(yb, yp, xs):
    """Whether the public rule's crossing interpolates across an undefined grid point."""
    diff = yp - yb
    idx = np.nonzero(~np.isnan(diff))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if diff[a] > 0 and diff[b] <= 0:
            return bool(b - a > 1)
    return None


def load_split_once(cache_root, split, tags):
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
        off, xy, cf_ = z["offsets"], z["xyxy"], z["conf"]
        preds.append([(xy[off[i]:off[i + 1]], cf_[off[i]:off[i + 1]]) for i in range(len(n))])
    return names, hw, preds


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def loc_summary(vals):
    v = [x for x in vals if x is not None]
    out = {"n": len(vals), "n_with_crossover": len(v)}
    if v:
        a = np.asarray(v)
        out.update({"median": float(np.median(a)), "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
                    "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None})
    return out


def fp_change(fb, fc):
    return None if fb == 0 else (fc - fb) / fb * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder with check_public_vs_cache.py, analysis_replication.py, analysis_ensemble_v2.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--weights-dir", required=True, help="Folder with baseline_seedNN.pt and p2_seedNN.pt (as in G1)")
    ap.add_argument("--g1-replay", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    ms = os.path.join(args.repo, "evaluation", "multiseed_eval.py")
    cfp = os.path.join(args.repo, "evaluation", "crossover_fp_analysis.py")
    if not (sha256_file(ms) == EXP_MS and sha256_file(cfp) == EXP_CF):
        sys.exit("ERROR: the public evaluation scripts differ from the published versions")
    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    ens = load_module(os.path.join(args.tools, "analysis_ensemble_v2.py"), "ens")
    stand_in = types.ModuleType("ultralytics")
    stand_in.YOLO = cpc.CachedYOLO
    sys.modules["ultralytics"] = stand_in
    cf = load_module(cfp, "public_crossover")
    if "torch" in sys.modules:
        sys.exit("ERROR: torch was imported")
    dataset = os.path.realpath(args.dataset)
    g1m = json.load(open(os.path.join(args.g1_replay, "multiseed_p2.json")))
    g1c = json.load(open(os.path.join(args.g1_replay, "crossover_p2.json")))
    g1r = json.load(open(os.path.join(args.g1_replay, "check_record.json")))
    if not (g1r.get("all_identical") and list(g1m["seeds"]) == SEEDS and list(g1c["seeds"]) == SEEDS):
        sys.exit("ERROR: the G1 replay is not the expected one (all_identical, seeds 42-49)")
    st = g1c["settings"]
    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])

    # caches: index by checkpoint hash for the public code; tags for the 10.2 code
    import glob
    recs = {}
    for js in sorted(glob.glob(os.path.join(args.cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        recs[r["checkpoint_sha256"]] = r
    plan = {}
    for c in ("baseline", "p2"):
        for s in SEEDS:
            w = os.path.join(args.weights_dir, "%s_seed%s.pt" % (c, s))
            r = recs.get(sha256_file(w))
            if r is None:
                sys.exit("ERROR: no test cache for %s" % w)
            if not (sha256_file(r["_npz"]) == r["npz_sha256"] and r["settings"]["imgsz"] == st["imgsz"]
                    and os.path.realpath(r["dataset"]) == dataset and int(r["train_args"]["seed"]) == int(s)):
                sys.exit("ERROR: cache of %s does not match (hash, imgsz, dataset or seed)" % w)
            plan[(c, s)] = {"weight": w, "tag": r["tag"], "npz_sha256": r["npz_sha256"]}
    cpc.CachedYOLO.index, cpc.CachedYOLO.split, cpc.CachedYOLO.dataset = recs, "test", dataset
    cf.YOLO = cpc.CachedYOLO

    # ---------- 10.1 ----------
    xs = np.logspace(np.log10(st["grid_lo_pct"]), np.log10(st["grid_hi_pct"]), st["grid_points"])
    curves, fps = {}, {}
    for n_done, (key, p) in enumerate(plan.items(), 1):
        print("[10.1] %2d/16 %s_seed%s  peak memory %.0f MB" % (n_done, key[0], key[1], peak_mb()), file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rr, fp = cf.eval_full(p["weight"], dataset, st["imgsz"], st["conf"], st["iou_match"], st["p33"], st["p67"])
        curves[key] = cf.sliding_recall(rr, xs, st["window_w"], st["min_lesions"])
        fps[key] = int(fp[0])
    problems = []
    per_seed = {}
    for s in SEEDS:
        yb, yp = curves[("baseline", s)], curves[("p2", s)]
        cr = cf.find_crossover(yb, yp, xs)
        cr = None if cr is None else float(cr)
        if not cr == g1c["per_seed"][s]["crossover_relative_area_pct"]:
            problems.append("seed %s crossover %s, G1 %s" % (s, cr, g1c["per_seed"][s]["crossover_relative_area_pct"]))
        if not (fps[("baseline", s)] == g1c["per_seed"][s]["fp_small_baseline"] and fps[("p2", s)] == g1c["per_seed"][s]["fp_small_candidate"]):
            problems.append("seed %s small FP differs from G1" % s)
        cx = crossings(yb, yp, xs)
        alt = min(cx, key=lambda c: abs(math.log(c["x"]) - math.log(DIM))) if cx else None
        any_span = any(c["spans_undefined"] for c in cx)
        per_seed[s] = {"primary": cr, "primary_spans_undefined": primary_segment_spans(yb, yp, xs),
                       "n_crossings": len(cx), "crossings": cx, "any_interpolation_spans_undefined": any_span,
                       "submitted_text_definition": None if alt is None else alt["x"],
                       "submitted_text_direction": None if alt is None else alt["direction"],
                       "primary_adjacent_defined_only": primary_adjacent_only(yb, yp, xs) if any_span else None,
                       "n_undefined_grid_points": {"baseline": int(np.isnan(yb).sum()), "p2": int(np.isnan(yp).sum())}}
    combos = []
    for sb, sc in itertools.product(SEEDS, SEEDS):
        cr = cf.find_crossover(curves[("baseline", sb)], curves[("p2", sc)], xs)
        combos.append({"baseline_seed": sb, "p2_seed": sc, "same_seed": sb == sc, "crossover": None if cr is None else float(cr),
                       "fp_small_baseline": fps[("baseline", sb)], "fp_small_p2": fps[("p2", sc)],
                       "fp_small_change_pct": fp_change(fps[("baseline", sb)], fps[("p2", sc)])})

    def combo_summary(rows):
        ch = [r["fp_small_change_pct"] for r in rows]
        d = [c for c in ch if c is not None]
        return {"crossover": loc_summary([r["crossover"] for r in rows]),
                "fp_change": {"n": len(ch), "n_undefined": len(ch) - len(d), "n_positive": sum(1 for c in d if c > 0),
                              "share_positive_of_defined": (sum(1 for c in d if c > 0) / len(d)) if d else None}}
    s101 = {"settings_from_g1_replay": st, "diminutive_boundary_pct": DIM, "per_seed": per_seed,
            "same_seed_pairs": combo_summary([r for r in combos if r["same_seed"]]),
            "all_64_combinations": combo_summary(combos), "combinations": combos,
            "submitted_text_definition_summary": loc_summary([per_seed[s]["submitted_text_definition"] for s in SEEDS])}

    # ---------- 10.2 ----------
    tags = [plan[("baseline", s)]["tag"] for s in SEEDS] + [plan[("p2", s)]["tag"] for s in SEEDS]
    print("[10.2] loading 16 caches  peak memory %.0f MB" % peak_mb(), file=sys.stderr, flush=True)
    names, hw, preds = load_split_once(args.cache, "test", tags)
    gt = ens.load_gt(dataset, "test", names, hw, p33, p67)
    one2one = {}
    for t, pr in zip(tags, preds):
        cov_hit = {k: 0 for k in STRATA}; o2o_hit = {k: 0 for k in STRATA}; tot = {k: 0 for k in STRATA}
        cov_fp_small = o2o_fp = o2o_fp_small = cov_fp = 0
        for (boxes, strata, W, H), (xy, cf_) in zip(gt, pr):
            keep = cf_ > st["conf"]
            pb, pc = xy[keep], cf_[keep]
            M = np.array([ens.box_iou(g, pb) for g in boxes]) if (len(boxes) and len(pb)) else np.zeros((len(boxes), len(pb)))
            area = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) / (W * H) if len(pb) else np.zeros(0)
            okm = M >= st["iou_match"]
            for gi, s in enumerate(strata):
                tot[s] += 1
                if len(pb) and okm[gi].any():
                    cov_hit[s] += 1
            unmatched_cov = ~okm.any(axis=0) if len(boxes) else np.ones(len(pb), dtype=bool)
            cov_fp += int(unmatched_cov.sum())
            cov_fp_small += int((unmatched_cov & (area < st["p33"])).sum())
            taken = np.zeros(len(boxes), dtype=bool)
            pmatched = np.zeros(len(pb), dtype=bool)
            for pi in np.argsort(-pc, kind="mergesort"):
                if not len(boxes):
                    break
                cand = np.nonzero(okm[:, pi] & ~taken)[0]
                if len(cand):
                    gi = cand[np.argmax(M[cand, pi])]
                    taken[gi] = True
                    pmatched[pi] = True
            for gi, s in enumerate(strata):
                if taken[gi]:
                    o2o_hit[s] += 1
            o2o_fp += int((~pmatched).sum())
            o2o_fp_small += int((~pmatched & (area < st["p33"])).sum())
        one2one[t] = {"total": tot, "coverage_hits": cov_hit, "one_to_one_hits": o2o_hit,
                      "coverage_fp": cov_fp, "coverage_fp_small": cov_fp_small,
                      "one_to_one_fp": o2o_fp, "one_to_one_fp_small": o2o_fp_small}
    for c, key in (("baseline", "baseline"), ("p2", "candidate")):
        for i, s in enumerate(SEEDS):
            o = one2one[plan[(c, s)]["tag"]]
            for k in STRATA:
                cov = o["coverage_hits"][k] / o["total"][k] if o["total"][k] > 0 else 0.0
                if not cov == g1m["per_seed"][key][k][i]:
                    problems.append("%s seed %s %s coverage recall %r, G1 %r" % (c, s, k, cov, g1m["per_seed"][key][k][i]))
                if o["one_to_one_hits"][k] > o["coverage_hits"][k]:
                    problems.append("%s seed %s %s one-to-one hits exceed coverage hits" % (c, s, k))
            if not o["coverage_fp_small"] == fps[(c, s)]:
                problems.append("%s seed %s coverage small FP %d, public %d" % (c, s, o["coverage_fp_small"], fps[(c, s)]))
            if o["one_to_one_fp"] < o["coverage_fp"]:
                problems.append("%s seed %s one-to-one FP below coverage FP" % (c, s))
    if problems:
        sys.exit("ERROR: built-in checks failed:\n  " + "\n  ".join(problems))
    comp = {}
    for k in STRATA:
        B = [100.0 * one2one[plan[("baseline", s)]["tag"]]["one_to_one_hits"][k] / one2one[plan[("baseline", s)]["tag"]]["total"][k] for s in SEEDS]
        C = [100.0 * one2one[plan[("p2", s)]["tag"]]["one_to_one_hits"][k] / one2one[plan[("p2", s)]["tag"]]["total"][k] for s in SEEDS]
        comp["recall_" + k] = dict(rep.compare(B, C), per_seed_baseline=B, per_seed_p2=C)
    for m in ("one_to_one_fp", "one_to_one_fp_small"):
        B = [float(one2one[plan[("baseline", s)]["tag"]][m]) for s in SEEDS]
        C = [float(one2one[plan[("p2", s)]["tag"]][m]) for s in SEEDS]
        comp[m] = dict(rep.compare(B, C), per_seed_baseline=B, per_seed_p2=C)
    s102 = {"per_model": one2one, "comparisons": comp,
            "note": "for false positives a positive difference means more false positives with +P2; the graded verdict labels do not apply"}

    rec = {"analysis": "crossover definition and one-to-one matching, addendum v1.0 Sections 10.1-10.2 (plan A11), replication batch",
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "inputs": {"plan": {"%s_seed%s" % k: v for k, v in plan.items()},
                      "sha256": {"multiseed_eval.py": EXP_MS, "crossover_fp_analysis.py": EXP_CF,
                                 "check_public_vs_cache.py": sha256_file(os.path.join(args.tools, "check_public_vs_cache.py")),
                                 "analysis_replication.py": sha256_file(os.path.join(args.tools, "analysis_replication.py")),
                                 "analysis_ensemble_v2.py": sha256_file(os.path.join(args.tools, "analysis_ensemble_v2.py")),
                                 "thresholds": sha256_file(args.thresholds),
                                 "g1_check_record": sha256_file(os.path.join(args.g1_replay, "check_record.json"))}},
           "builtin_checks": "primary crossover and small FP per seed equal G1; coverage recall per stratum and coverage small FP equal G1; one-to-one <= coverage hits, >= coverage FP",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__), "peak_memory_mb_before_writing": peak_mb(),
           "section_10_1": s101, "section_10_2": s102}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "a11.json"), "w") as f:
        json.dump(rec, f, indent=2)

    f2 = lambda x: "none" if x is None else "%.2f" % x
    L = ["A11 (addendum 10.1-10.2), replication batch, +P2 minus baseline",
         "built-in checks passed: primary crossover, small FP and coverage recall equal the G1 replay", "",
         "== 10.1 per seed: primary | submitted-text definition (closest to %.2f%%) | crossings | any span over undefined | primary, adjacent defined only" % DIM]
    for s in SEEDS:
        p = per_seed[s]
        L.append("  seed %s: %s | %s (%s) | %d | %s | %s" % (
            s, f2(p["primary"]), f2(p["submitted_text_definition"]), p["submitted_text_direction"] or "-", p["n_crossings"],
            "yes" if p["any_interpolation_spans_undefined"] else "no",
            f2(p["primary_adjacent_defined_only"]) if p["any_interpolation_spans_undefined"] else "not needed"))
    for lab, sm in (("primary, same-seed pairs", s101["same_seed_pairs"]["crossover"]),
                    ("submitted-text definition, same-seed pairs", s101["submitted_text_definition_summary"]),
                    ("primary, all 64 combinations", s101["all_64_combinations"]["crossover"])):
        L.append("  %s: %d/%d with a crossover; median %s, IQR [%s, %s], mean %s, SD %s" % (
            lab, sm["n_with_crossover"], sm["n"], f2(sm.get("median")), f2(sm.get("q25")), f2(sm.get("q75")), f2(sm.get("mean")), f2(sm.get("sd"))))
    for lab, sm in (("same-seed pairs", s101["same_seed_pairs"]["fp_change"]), ("all 64 combinations", s101["all_64_combinations"]["fp_change"])):
        L.append("  small-scale FP change, %s: %d positive of %d defined (%d undefined)" % (lab, sm["n_positive"], sm["n"] - sm["n_undefined"], sm["n_undefined"]))
    L += ["", "== 10.2 one-to-one matching (Section 5 statistics; pp of recall, counts of FP)"]
    for k in ("recall_small", "recall_medium", "recall_large", "one_to_one_fp", "one_to_one_fp_small"):
        c = comp[k]
        w = c["welch"]
        L.append("  %-20s baseline %.2f (SD %.2f)  +P2 %.2f (SD %.2f)  delta %+.2f  Welch CI [%+.2f, %+.2f]%s  paired CI [%+.2f, %+.2f]" % (
            k, c["mean_reference"], c["sd_reference"], c["mean_candidate"], c["sd_candidate"], w["delta"], w["ci95"][0], w["ci95"][1],
            ("  -> " + c["verdict"]) if k.startswith("recall") else "", c["paired"]["ci95"][0], c["paired"]["ci95"][1]))
    lost = {k: [one2one[plan[(c, s)]["tag"]]["coverage_hits"][k] - one2one[plan[(c, s)]["tag"]]["one_to_one_hits"][k]
                for c in ("baseline", "p2") for s in SEEDS] for k in STRATA}
    L.append("  hits lost by one-to-one matching (16 models): " + "; ".join("%s %d-%d" % (k, min(v), max(v)) for k, v in lost.items()))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "a11.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "a11.json"))
    print("peak memory %.0f MB" % peak_mb())


if __name__ == "__main__":
    main()
