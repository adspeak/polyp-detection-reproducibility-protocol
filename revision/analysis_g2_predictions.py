"""Same-seed repeats, G2: cached test predictions (Paper3 revision analysis addendum v1.0,
Section 8), and the three kinds of variation described side by side.

Pairs: replication run (runs/NAME) against its G2 repeat (runs_g2/NAME) for baseline_seed42,
baseline_seed43, p2_seed42 and p2_seed43; both cached once at confidence 0.001, imgsz 1280,
on the same instance. Per pair:
  * identical predictions: per image, the cached arrays (boxes, confidences, classes) are
    compared bit for bit; reported as the share of images whose predictions are identical, and
    as the number of predictions that occur in both runs with identical values (exact multiset
    match per image) relative to each run's total. Done for all cached predictions
    (confidence > 0.001) and at the operating point (confidence > 0.25);
  * recall per stratum (training-set terciles, coverage: any prediction with confidence > 0.25
    and IoU >= 0.5) of each run and the difference, repeat minus replication, in pp.
The cache files themselves are not compared by hash: they also store the inference times.
Three kinds of variation, small-stratum recall, described side by side (no decomposition):
(i) same-seed repeats (above); (ii) seed-to-seed SD within the replication batch; (iii) the
difference between the original and the replication batch (Section 6), read from
replication_analysis.json.
Built-in check: the recall of the four replication runs equals the G1 replay. The label reader
and the coverage rule are those of analysis_fov.py. CPU only.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np

NAMES = ["baseline_seed42", "baseline_seed43", "p2_seed42", "p2_seed43"]
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


def load_full(cache_root, tag):
    js = os.path.join(cache_root, "test", tag + ".json")
    r = json.load(open(js))
    npz = js[:-5] + ".npz"
    if not sha256_file(npz) == r["npz_sha256"]:
        raise SystemExit("ERROR: %s changed after it was written" % npz)
    z = np.load(npz)
    names = [str(x) for x in z["names"]]
    off, xy, cf, cl = z["offsets"], z["xyxy"], z["conf"], z["cls"]
    per = [(xy[off[i]:off[i + 1]], cf[off[i]:off[i + 1]], cl[off[i]:off[i + 1]]) for i in range(len(names))]
    return r, names, z["orig_hw"], per


def rows_bytes(xy, cf, cl):
    return [xy[i].tobytes() + cf[i:i + 1].tobytes() + cl[i:i + 1].tobytes() for i in range(len(cf))]


def compare_preds(pa, pb, thr):
    same_img = 0
    common = na = nb = 0
    for (xa, ca, la), (xb, cb, lb) in zip(pa, pb):
        ka, kb = ca > thr, cb > thr
        xa, ca, la, xb, cb, lb = xa[ka], ca[ka], la[ka], xb[kb], cb[kb], lb[kb]
        if np.array_equal(xa, xb) and np.array_equal(ca, cb) and np.array_equal(la, lb):
            same_img += 1
        ra, rb = rows_bytes(xa, ca, la), rows_bytes(xb, cb, lb)
        na, nb = na + len(ra), nb + len(rb)
        cnt = {}
        for r in ra:
            cnt[r] = cnt.get(r, 0) + 1
        for r in rb:
            if cnt.get(r, 0) > 0:
                cnt[r] -= 1
                common += 1
    n = len(pa)
    return {"images": n, "images_identical": same_img, "share_images_identical": same_img / n,
            "predictions_replication": na, "predictions_repeat": nb, "predictions_in_both": common,
            "share_of_replication_predictions": common / na if na else None,
            "share_of_repeat_predictions": common / nb if nb else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)), help="Folder with analysis_fov.py")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--g1-replay", required=True)
    ap.add_argument("--replication", required=True, help="replication_analysis.json (A1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    fov = load_module(os.path.join(args.tools, "analysis_fov.py"), "fov")
    ens = load_module(os.path.join(args.tools, "analysis_ensemble_v2.py"), "ens")
    dataset = os.path.realpath(args.dataset)
    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])
    g1m = json.load(open(os.path.join(args.g1_replay, "multiseed_p2.json")))
    g1c = json.load(open(os.path.join(args.g1_replay, "crossover_p2.json")))
    conf, iou_m = float(g1c["settings"]["conf"]), float(g1c["settings"]["iou_match"])
    rj = json.load(open(args.replication))

    labels, names0, hw0 = None, None, None
    res, inputs, problems = {}, {}, []
    for n in NAMES:
        runs = {}
        for proj in ("runs", "runs_g2"):
            tag = "%s__%s" % (proj, n)
            r, names, hw, per = load_full(args.cache, tag)
            ck = os.path.join(args.root, proj, n, "weights", "best.pt")
            if not (r["checkpoint_sha256"] == sha256_file(ck) and r["settings"]["imgsz"] == 1280
                    and os.path.realpath(r["dataset"]) == dataset and int(r["train_args"]["seed"]) == int(n[-2:])):
                sys.exit("ERROR: cache %s does not belong to %s (checkpoint, imgsz, dataset or seed)" % (tag, ck))
            if names0 is None:
                names0, hw0 = names, hw
                labels = fov.load_labels(dataset, names, hw)
            elif not (names == names0 and np.array_equal(hw, hw0)):
                sys.exit("ERROR: %s covers different images" % tag)
            runs[proj] = (r, per)
            inputs[tag] = r["npz_sha256"]
        areas = np.array([a for boxes in labels for _, a in boxes])
        terc = fov.strata_of(areas, p33, p67, TERC)
        rec_ = {}
        for proj in ("runs", "runs_g2"):
            per = runs[proj][1]
            h = fov.hits_for(ens.box_iou, labels, [(x, c) for x, c, _ in per], conf, iou_m)
            rec_[proj] = {k: float(h[terc == k].sum()) / float((terc == k).sum()) for k in TERC}
        cfg, s = n.split("_seed")
        key = "baseline" if cfg == "baseline" else "candidate"
        i = list(g1m["seeds"]).index(s)
        for k in TERC:
            if not rec_["runs"][k] == g1m["per_seed"][key][k][i]:
                problems.append("%s %s recall differs from G1" % (n, k))
        res[n] = {"all_cached_predictions": compare_preds(runs["runs"][1], runs["runs_g2"][1], 0.001 - 1e-12),
                  "operating_point_0.25": compare_preds(runs["runs"][1], runs["runs_g2"][1], conf),
                  "recall_pp": {"replication": {k: 100 * v for k, v in rec_["runs"].items()},
                                "repeat": {k: 100 * v for k, v in rec_["runs_g2"].items()},
                                "repeat_minus_replication": {k: 100 * (rec_["runs_g2"][k] - rec_["runs"][k]) for k in TERC}},
                  "hosts": [runs["runs"][0].get("host"), runs["runs_g2"][0].get("host")]}
    if problems:
        sys.exit("ERROR: built-in check failed:\n  " + "\n  ".join(problems))
    rs = rj["comparisons"]["replication"]["small"]
    bd = {cfg: {k: rj["configuration_means_replication_vs_original"][cfg]["small"][k]
                for k in ("replication_minus_original", "welch_ci95", "replication_mean", "original_mean")}
          for cfg in ("baseline", "p2")}
    three = {"i_same_seed_repeat_small_pp": {n: res[n]["recall_pp"]["repeat_minus_replication"]["small"] for n in NAMES},
             "ii_seed_to_seed_sd_replication_small_pp": {"baseline": rs["sd_reference"], "p2": rs["sd_candidate"]},
             "iii_replication_minus_original_small_pp": bd}
    rec = {"analysis": "G2 same-seed repeats, cached test predictions, and the three kinds of variation (addendum v1.0 Section 8)",
           "run_utc": datetime.now(timezone.utc).isoformat(),
           "settings": {"identical": "bitwise equality of boxes, confidences and classes per image; exact multiset match for predictions",
                        "recall": "coverage at confidence > %g, IoU >= %g, training-set terciles" % (conf, iou_m)},
           "inputs_sha256": dict(inputs, analysis_fov=sha256_file(os.path.join(args.tools, "analysis_fov.py")),
                                 replication_analysis=sha256_file(args.replication)),
           "builtin_check": "recall of the four replication runs equals the G1 replay",
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
           "this_script_sha256": sha256_file(__file__), "pairs": res, "three_kinds_of_variation": three}
    os.makedirs(args.out)
    with open(os.path.join(args.out, "g2_predictions.json"), "w") as f:
        json.dump(rec, f, indent=2)
    L = ["G2 cached test predictions (addendum Section 8): replication run vs same-seed repeat", "built-in check passed: replication recall equals G1", ""]
    for n in NAMES:
        a, o, r_ = res[n]["all_cached_predictions"], res[n]["operating_point_0.25"], res[n]["recall_pp"]
        L.append("  %-16s all cached: images identical %d/%d; predictions in both %d of %d / %d" % (
            n, a["images_identical"], a["images"], a["predictions_in_both"], a["predictions_replication"], a["predictions_repeat"]))
        L.append("  %-16s conf > 0.25: images identical %d/%d; predictions in both %d of %d / %d" % (
            "", o["images_identical"], o["images"], o["predictions_in_both"], o["predictions_replication"], o["predictions_repeat"]))
        L.append("  %-16s recall (pp) replication / repeat / difference: %s" % ("", "; ".join(
            "%s %.2f / %.2f / %+.2f" % (k, r_["replication"][k], r_["repeat"][k], r_["repeat_minus_replication"][k]) for k in TERC)))
    L.append("")
    L.append("== three kinds of variation, small-stratum recall (pp), side by side, no decomposition")
    L.append("  (i)   same-seed repeat minus replication run: " + ", ".join("%s %+.2f" % (n, v) for n, v in three["i_same_seed_repeat_small_pp"].items()))
    L.append("  (ii)  seed-to-seed SD in the replication batch: baseline %.2f, +P2 %.2f" % (rs["sd_reference"], rs["sd_candidate"]))
    L.append("  (iii) replication minus original batch: " + ", ".join("%s %+.2f [%+.2f, %+.2f]" % (
        c, bd[c]["replication_minus_original"], bd[c]["welch_ci95"][0], bd[c]["welch_ci95"][1]) for c in ("baseline", "p2")))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "g2_predictions.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "g2_predictions.json"))


if __name__ == "__main__":
    main()
