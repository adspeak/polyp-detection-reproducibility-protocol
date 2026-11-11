"""G7 planning simulation (supplementary pre-registration S1, Section 7). Planning numbers, not results.

Model and code: those of addendum Section 10 (analysis_oc_simulation.py, imported unchanged and
hash-checked): per-seed pairs (B, C) bivariate normal, B mean 0, C mean Delta, SDs and
correlation r observed in the original positive control (read from positive_control.json, as
analysis_oc_simulation.py does); 10,000 replicates; random seed 20261106 through its draw().
Each replicate draws 16 pairs: pairs 1-8 are the first batch, pairs 9-16 the second batch.
Decisions through its decide(): Welch 95% CI (graded verdict of Section 5) and paired-t 95% CI.

Events, per method (paired / Welch), the same method at both stages:
  improvement = CI lower bound > 0; deterioration = CI upper bound < 0; any = either.
Procedures:
  fixed 8        : first batch only;
  fixed 16       : all 16 pairs, tested once;
  two-stage alone: first batch tested; if it shows no evidence in either direction, the second
                   batch is added and tested alone; positive if either stage is;
  two-stage pool : as above, but the second stage tests all 16 pairs.
Reported for Delta = 0: P(false improvement) and P(two-sided false positive); for the observed
Delta: P(evidence of improvement). The simulation quantifies error rates; it corrects nothing.

Built-in check: cell() of analysis_oc_simulation.py for the observed values at S = 8 must equal,
key by key, the positive-control cell stored in oc_simulation.json (A4). CPU only.
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

EXPECT_OC = "6f8821661124d1ddf7da86722c5a3d264bfcb4a2b1b2b14a50a282a064f52c32"
PROCS = ["fixed 8 pairs", "fixed 16 pairs", "two-stage, second batch alone", "two-stage, pooled"]


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


def events(oc, B, C):
    verdict, _orig, plo, phi = oc.decide(B, C)
    w_imp, w_det = verdict <= 1, verdict == 3
    return {"paired": (plo, phi), "welch": (w_imp, w_det)}


def procedures(oc, delta, sb, sc, r):
    B, C = oc.draw(delta, sb, sc, r, 16)
    e1 = events(oc, B[:, :8], C[:, :8])
    e2 = events(oc, B[:, 8:], C[:, 8:])
    e16 = events(oc, B, C)
    out = {}
    for m in ("paired", "welch"):
        i1, d1 = e1[m]
        i2, d2 = e2[m]
        i16, d16 = e16[m]
        a1 = i1 | d1
        go = ~a1
        res = {
            PROCS[0]: (i1, a1),
            PROCS[1]: (i16, i16 | d16),
            PROCS[2]: (i1 | (go & i2), a1 | (go & (i2 | d2))),
            PROCS[3]: (i1 | (go & i16), a1 | (go & (i16 | d16))),
        }
        out[m] = {p: {"P_improvement": float(np.mean(v[0])), "P_two_sided": float(np.mean(v[1]))} for p, v in res.items()}
        out[m]["P_second_stage_reached"] = float(np.mean(go))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True, help="Folder with analysis_oc_simulation.py and analysis_replication.py")
    ap.add_argument("--positive-control", required=True, help="positive_control.json (G3)")
    ap.add_argument("--oc-json", required=True, help="oc_simulation.json (A4)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    ocp = os.path.join(args.tools, "analysis_oc_simulation.py")
    if not sha256_file(ocp) == EXPECT_OC:
        sys.exit("ERROR: analysis_oc_simulation.py is not the A4 version (SHA-256 %s)" % sha256_file(ocp))
    oc = load_module(ocp, "oc")
    pj = json.load(open(args.positive_control))
    ps = pj["comparisons"]["small"]
    delta, sb, sc = float(ps["welch"]["delta"]), float(ps["sd_reference"]), float(ps["sd_candidate"])
    r = float(ps["paired"]["r_between_configurations"])

    a4 = json.load(open(args.oc_json))["added_positive_control_cell"]
    mine = oc.cell(delta, sb, sc, r, 8)
    bad = [k for k in a4 if not mine.get(k) == a4[k]]
    if bad:
        sys.exit("ERROR: built-in check failed, the S = 8 cell differs from A4 in %s" % bad)

    res = {"delta_0": procedures(oc, 0.0, sb, sc, r), "delta_observed": procedures(oc, delta, sb, sc, r)}
    rec = {"analysis": "G7 planning simulation, supplementary pre-registration S1 Section 7 (planning numbers, not results)",
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "parameters": {"delta_observed": delta, "sd_640": sb, "sd_1280": sc, "r": r, "replicates": oc.REPS,
                          "seed": oc.SEED, "pairs_drawn": 16, "first_batch": "pairs 1-8", "second_batch": "pairs 9-16",
                          "continue_rule": "second stage only if the first batch shows no evidence in either direction (same method)",
                          "monte_carlo_se_max": 0.5 / math.sqrt(oc.REPS)},
           "builtin_check": {"A4_cell_reproduced_exactly": True, "A4_paired": a4["P_paired_ci_lower_gt_0"],
                             "A4_welch": a4["P_welch_ci_lower_gt_0"]},
           "inputs_sha256": {"analysis_oc_simulation.py": EXPECT_OC, "positive_control.json": sha256_file(args.positive_control),
                             "oc_simulation.json": sha256_file(args.oc_json)},
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
           "this_script_sha256": sha256_file(os.path.abspath(__file__)),
           "results": res}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "g7_planning.json"), "w") as f:
        json.dump(rec, f, indent=2)
    L = ["G7 planning simulation (S1 Section 7): Delta %.3f, SD 640 %.3f, SD 1280 %.3f, r %.3f; %d replicates, seed %d; MC SE <= %.3f" % (
        delta, sb, sc, r, oc.REPS, oc.SEED, 0.5 / math.sqrt(oc.REPS)),
         "built-in check passed: S = 8 cell equals A4 (paired %.3f, Welch %.3f)" % (a4["P_paired_ci_lower_gt_0"], a4["P_welch_ci_lower_gt_0"]), "",
         "== Delta = 0                      false improvement      two-sided false positive",
         "   procedure                       paired    Welch        paired    Welch"]
    z = res["delta_0"]
    for p in PROCS:
        L.append("   %-30s  %.4f    %.4f       %.4f    %.4f" % (p, z["paired"][p]["P_improvement"], z["welch"][p]["P_improvement"],
                                                            z["paired"][p]["P_two_sided"], z["welch"][p]["P_two_sided"]))
    L.append("   second stage reached: paired %.4f, Welch %.4f" % (z["paired"]["P_second_stage_reached"], z["welch"]["P_second_stage_reached"]))
    L += ["", "== Delta = %.3f                  evidence of improvement" % delta, "   procedure                       paired    Welch"]
    o = res["delta_observed"]
    for p in PROCS:
        L.append("   %-30s  %.4f    %.4f" % (p, o["paired"][p]["P_improvement"], o["welch"][p]["P_improvement"]))
    L.append("   second stage reached: paired %.4f, Welch %.4f" % (o["paired"]["P_second_stage_reached"], o["welch"]["P_second_stage_reached"]))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "g7_planning.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "g7_planning.json"))


if __name__ == "__main__":
    main()
