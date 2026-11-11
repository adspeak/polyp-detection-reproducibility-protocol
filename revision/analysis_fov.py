"""Field-of-view assumption (Paper3 revision analysis addendum v1.0, Section 10.3; plan A6).

For F = 25, 30, 35, 40, 45 and 50 mm the clinical strata are a < (5/F)^2 (diminutive),
(5/F)^2 <= a < (10/F)^2 and a >= (10/F)^2, with a the relative box area w*h of the YOLO label
(as in the public scripts). The Welch verdict for the diminutive stratum is recomputed for every
comparison with per-box records, with the box counts per stratum:
  replication          baseline (runs/baseline_seed42-49) vs +P2 (runs/p2_seed42-49), 1280;
  positive control     640 (runs_g3/baseline640_seed42-49) vs 1280 (runs/baseline_seed42-49);
  control configuration baseline vs control (runs/control_seed42-49), 1280.
A box counts as detected when any cached prediction with confidence > 0.25 (deviation log entry
1) has IoU >= 0.5 with it (coverage recall, as in the public multiseed_eval.py). The statistics
are compare() of analysis_replication.py (Section 5); paired results are reported beside the
Welch verdict, which is the one Section 10.3 asks for (also for the positive control, whose
primary analysis in Section 7 is paired). The original batch has only per-stratum recall and is
not re-stratified.

Built-in checks (the script stops on any failure):
  * the tercile strata of the label reader equal those of load_gt() of analysis_ensemble_v2.py;
  * with the training-set terciles, recall per stratum equals the G1 replay (baseline and +P2,
    seeds 42-49; control seed 42) and the positive-control record (640 and 1280, seeds 42-49).
Caches are read once per array (as load_split_once() in analysis_a11_v2.py). CPU only.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import sys
from datetime import datetime, timezone

import numpy as np
import scipy

SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
FS = [25, 30, 35, 40, 45, 50]
STRATA = ("diminutive", "small_6_9mm", "large_10mm")
TERC = ("small", "medium", "large")


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


def load_once(cache_root, tag):
    js = os.path.join(cache_root, "test", tag + ".json")
    r = json.load(open(js))
    npz = js[:-5] + ".npz"
    if not sha256_file(npz) == r["npz_sha256"]:
        raise SystemExit("ERROR: %s changed after it was written" % npz)
    z = np.load(npz)
    names = [str(x) for x in z["names"]]
    off, xy, cf = z["offsets"], z["xyxy"], z["conf"]
    return r, names, z["orig_hw"], [(xy[off[i]:off[i + 1]], cf[off[i]:off[i + 1]]) for i in range(len(names))]


def load_labels(dataset, names, hw):
    """Per image: list of (pixel box, relative area a = w*h), parsed as in the public scripts."""
    out = []
    for n, (h, w) in zip(names, hw):
        W, H = int(w), int(h)
        lp = os.path.join(dataset, "labels", "test", n[:-4] + ".txt")
        boxes = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w2, hh = map(float, p)
                    boxes.append(([(cx - w2 / 2) * W, (cy - hh / 2) * H, (cx + w2 / 2) * W, (cy + hh / 2) * H], w2 * hh))
        out.append(boxes)
    return out


def hits_for(box_iou, labels, preds, conf, iou_m):
    """Coverage hit (0/1) for every ground-truth box, in label order."""
    h = []
    for boxes, (xy, cf) in zip(labels, preds):
        pb = xy[cf > conf]
        for g, _ in boxes:
            io = box_iou(g, pb)
            h.append(1 if (len(io) > 0 and io.max() >= iou_m) else 0)
    return np.array(h, dtype=int)


def strata_of(areas, lo, hi, names):
    return np.where(areas < lo, names[0], np.where(areas < hi, names[1], names[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Paper3 root; run folders are relative to it")
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder with analysis_replication.py and analysis_ensemble_v2.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--g1-replay", required=True)
    ap.add_argument("--positive-control", required=True, help="positive_control.json (G3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    rep = load_module(os.path.join(args.tools, "analysis_replication.py"), "rep")
    ens = load_module(os.path.join(args.tools, "analysis_ensemble_v2.py"), "ens")
    dataset = os.path.realpath(args.dataset)
    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])
    g1m = json.load(open(os.path.join(args.g1_replay, "multiseed_p2.json")))
    g1k = json.load(open(os.path.join(args.g1_replay, "multiseed_control.json")))
    g1c = json.load(open(os.path.join(args.g1_replay, "crossover_p2.json")))
    pcj = json.load(open(args.positive_control))
    conf, iou_m = float(g1c["settings"]["conf"]), float(g1c["settings"]["iou_match"])

    runs = {"baseline": "runs/baseline_seed{s}", "p2": "runs/p2_seed{s}", "control": "runs/control_seed{s}",
            "baseline640": "runs_g3/baseline640_seed{s}"}
    imgsz = {"baseline": 1280, "p2": 1280, "control": 1280, "baseline640": 640}
    by_sha = {}
    for js in sorted(glob.glob(os.path.join(args.cache, "test", "*.json"))):
        r = json.load(open(js))
        by_sha[r["checkpoint_sha256"]] = r["tag"]
    hits, names0, hw0, labels, inputs = {}, None, None, None, {}
    for cfg, pat in runs.items():
        for s in SEEDS:
            ck = os.path.join(args.root, pat.format(s=s), "weights", "best.pt")
            tag = by_sha.get(sha256_file(ck))
            if tag is None:
                sys.exit("ERROR: no test cache for %s" % ck)
            r, names, hw, preds = load_once(args.cache, tag)
            if not (r["settings"]["imgsz"] == imgsz[cfg] and int(r["train_args"]["seed"]) == int(s)
                    and os.path.realpath(r["dataset"]) == dataset):
                sys.exit("ERROR: %s cache does not match (imgsz, seed or dataset)" % tag)
            if names0 is None:
                names0, hw0 = names, hw
                labels = load_labels(dataset, names, hw)
                gt = ens.load_gt(dataset, "test", names, hw, p33, p67)
                mine = [("small" if a < p33 else ("medium" if a < p67 else "large")) for boxes in labels for _, a in boxes]
                theirs = [s_ for (_, strata, _, _) in gt for s_ in strata]
                if not mine == theirs:
                    sys.exit("ERROR: label reader strata differ from load_gt()")
            elif not (names == names0 and np.array_equal(hw, hw0)):
                sys.exit("ERROR: %s covers different images" % tag)
            hits[(cfg, s)] = hits_for(ens.box_iou, labels, preds, conf, iou_m)
            inputs[tag] = r["npz_sha256"]
        print("[cache] %s done, peak memory %.0f MB" % (cfg, peak_mb()), file=sys.stderr, flush=True)
    areas = np.array([a for boxes in labels for _, a in boxes])
    terc = strata_of(areas, p33, p67, TERC)

    def recall(cfg, s, st, which):
        sel = st == which
        return float(hits[(cfg, s)][sel].sum()) / float(sel.sum()) if sel.sum() else None

    problems = []
    for i, s in enumerate(SEEDS):
        for k in TERC:
            for cfg, key in (("baseline", "baseline"), ("p2", "candidate")):
                if not recall(cfg, s, terc, k) == g1m["per_seed"][key][k][i]:
                    problems.append("%s seed %s %s recall differs from G1" % (cfg, s, k))
            if not 100 * recall("baseline640", s, terc, k) == pcj["per_seed_pp"][k]["reference_640"][i]:
                problems.append("640 seed %s %s differs from the positive-control record" % (s, k))
            if not 100 * recall("baseline", s, terc, k) == pcj["per_seed_pp"][k]["candidate_1280"][i]:
                problems.append("1280 seed %s %s differs from the positive-control record" % (s, k))
    for k in TERC:
        if not recall("control", "42", terc, k) == g1k["per_seed"]["candidate"][k][0]:
            problems.append("control seed 42 %s differs from G1" % k)
    if problems:
        sys.exit("ERROR: built-in checks failed:\n  " + "\n  ".join(problems))

    comparisons = {"replication": ("baseline", "p2", "+P2 minus baseline"),
                   "positive_control": ("baseline640", "baseline", "1280 minus 640"),
                   "control_configuration": ("baseline", "control", "control minus baseline")}
    res = {}
    for F in FS:
        lo, hi = (5.0 / F) ** 2, (10.0 / F) ** 2
        st = strata_of(areas, lo, hi, STRATA)
        counts = {k: int((st == k).sum()) for k in STRATA}
        row = {"F_mm": F, "thresholds_relative_area": [lo, hi], "box_counts": counts, "comparisons": {}}
        for name, (b, c, lab) in comparisons.items():
            row["comparisons"][name] = {}
            for k in STRATA:
                if counts[k] == 0:
                    row["comparisons"][name][k] = None
                    continue
                B = [100 * recall(b, s, st, k) for s in SEEDS]
                C = [100 * recall(c, s, st, k) for s in SEEDS]
                row["comparisons"][name][k] = dict(rep.compare(B, C), per_seed_reference=B, per_seed_candidate=C, label=lab)
        res[str(F)] = row

    rec = {"analysis": "field-of-view assumption, addendum v1.0 Section 10.3 (plan A6)",
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "settings": {"F_mm": FS, "strata": "a < (5/F)^2; (5/F)^2 <= a < (10/F)^2; a >= (10/F)^2; a = w*h of the label",
                        "detection": "coverage: any prediction with confidence > %g and IoU >= %g" % (conf, iou_m),
                        "verdict": "Welch (Section 10.3); paired beside it", "units": "pp of recall"},
           "comparisons": {k: v[2] for k, v in comparisons.items()},
           "inputs_sha256": dict(inputs, thresholds=sha256_file(args.thresholds),
                                 positive_control_json=sha256_file(args.positive_control),
                                 g1_check_record=sha256_file(os.path.join(args.g1_replay, "check_record.json")),
                                 analysis_replication=sha256_file(os.path.join(args.tools, "analysis_replication.py")),
                                 analysis_ensemble_v2=sha256_file(os.path.join(args.tools, "analysis_ensemble_v2.py"))),
           "builtin_checks": "tercile recall equals G1 (baseline, +P2, control seed 42) and the positive-control record",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__), "peak_memory_mb": peak_mb(),
           "results": res}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "fov.json"), "w") as f:
        json.dump(rec, f, indent=2)

    L = ["FOV sensitivity (addendum 10.3): coverage recall at conf > %g, IoU >= %g; Welch verdict for the diminutive stratum" % (conf, iou_m),
         "built-in checks passed: tercile recall equals G1 and the positive-control record", ""]
    for F in FS:
        row = res[str(F)]
        L.append("== F = %d mm: diminutive a < %.4f, 6-9 mm < %.4f; boxes: diminutive %d, 6-9 mm %d, >= 10 mm %d" % (
            F, row["thresholds_relative_area"][0], row["thresholds_relative_area"][1],
            row["box_counts"]["diminutive"], row["box_counts"]["small_6_9mm"], row["box_counts"]["large_10mm"]))
        for name in comparisons:
            c = row["comparisons"][name]["diminutive"]
            if c is None:
                L.append("  %-22s no diminutive boxes" % name)
                continue
            w, p = c["welch"], c["paired"]
            L.append("  %-22s ref %.2f  cand %.2f  delta %+.2f  Welch CI [%+.2f, %+.2f] -> %s | paired CI [%+.2f, %+.2f]" % (
                name, c["mean_reference"], c["mean_candidate"], w["delta"], w["ci95"][0], w["ci95"][1], c["verdict"],
                p["ci95"][0], p["ci95"][1]))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "fov.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "fov.json"))
    print("peak memory %.0f MB" % peak_mb())


if __name__ == "__main__":
    main()
