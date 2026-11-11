"""COCO-style AP and model cost (Paper3 revision analysis addendum v1.0, Section 10, items 6 and 7).

AP (item 6): pycocotools on the cached confidence-0.001 test predictions, at most 100 per
image (the highest-scoring ones), COCO area ranges in original pixels. Ground truth comes
from the YOLO test labels converted to pixel boxes with the image sizes stored in the
cache; area = box width x height in pixels. Reported per model: AP, AP50, AP_S, AP_M,
AP_L and AR_S (x100). Summarised per configuration by mean and SD, and +P2 or control
minus baseline by Welch 95% CI (when the candidate has at least 2 runs).

Model cost (item 7, needs torch and ultralytics): parameter counts by backbone, neck and
head (unfused, as trained) and in total for the fused model used at inference; GFLOPs at
640 and at the training input size, unfused and fused (Ultralytics get_flops); mean
inference time per test image from the one-time inference (cache records).

Tags are <project>__<run>; configuration = run name before "_seed".
"""

import argparse
import contextlib
import copy
import glob
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import stats

METRICS = ("AP", "AP50", "AP_S", "AP_M", "AP_L", "AR_S")
STAT_INDEX = {"AP": 0, "AP50": 1, "AP_S": 3, "AP_M": 4, "AP_L": 5, "AR_S": 9}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def load_caches(cache_root, split, pattern):
    recs = []
    for js in sorted(glob.glob(os.path.join(cache_root, split, "*.json"))):
        r = json.load(open(js))
        if not re.search(pattern, r["tag"]):
            continue
        npz = js[:-5] + ".npz"
        if not sha256_file(npz) == r["npz_sha256"]:
            sys.exit("ERROR: %s changed after it was written" % npz)
        r["_npz"] = npz
        recs.append(r)
    return recs


def build_gt(dataset, split, names, hw):
    images, anns = [], []
    aid = 0
    n_boxes = 0
    for i, (n, (h, w)) in enumerate(zip(names, hw)):
        images.append({"id": i + 1, "file_name": n, "height": int(h), "width": int(w)})
        lp = os.path.join(dataset, "labels", split, n[:-4] + ".txt")
        if not os.path.exists(lp):
            continue
        for ln in open(lp):
            p = ln.split()
            if not len(p) == 5:
                continue
            cx, cy, bw, bh = [float(v) for v in p[1:]]
            x, y, pw, ph = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            aid += 1
            n_boxes += 1
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "bbox": [x, y, pw, ph],
                         "area": pw * ph, "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "polyp"}]}, n_boxes


def coco_eval(gt_dict, z):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gt = COCO()
        gt.dataset = copy.deepcopy(gt_dict)
        gt.createIndex()
        off, xyxy, conf = z["offsets"], z["xyxy"].astype(np.float64), z["conf"].astype(np.float64)
        rows = []
        for i in range(len(off) - 1):
            a, b = int(off[i]), int(off[i + 1])
            if b == a:
                continue
            idx = np.argsort(-conf[a:b], kind="mergesort")[:100] + a
            bx = xyxy[idx]
            rows.append(np.column_stack([np.full(len(idx), i + 1), bx[:, 0], bx[:, 1], bx[:, 2] - bx[:, 0],
                                         bx[:, 3] - bx[:, 1], conf[idx], np.ones(len(idx))]))
        det = gt.loadRes(np.concatenate(rows)) if rows else None
        if det is None:
            return {k: 0.0 for k in METRICS}, 0
        ev = COCOeval(gt, det, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {k: float(s[STAT_INDEX[k]]) * 100 for k in METRICS}, int(sum(len(r) for r in rows))


def welch(B, C):
    B, C = np.asarray(B, float), np.asarray(C, float)
    n, m = len(B), len(C)
    if n < 2 or m < 2:
        return None
    vb, vc = B.var(ddof=1), C.var(ddof=1)
    se = math.sqrt(vb / n + vc / m)
    if se == 0:
        return {"delta": float(C.mean() - B.mean()), "se": 0.0, "df": None, "ci95": None}
    nu = (vb / n + vc / m) ** 2 / ((vb / n) ** 2 / (n - 1) + (vc / m) ** 2 / (m - 1))
    t = stats.t.ppf(0.975, nu)
    d = float(C.mean() - B.mean())
    return {"delta": d, "se": se, "df": nu, "ci95": [d - t * se, d + t * se]}


def model_cost(best):
    import torch
    from ultralytics.utils.torch_utils import get_flops
    ck = torch.load(best, map_location="cpu")
    model = ck["model"].float().eval()
    imgsz = int((ck.get("train_args") or {}).get("imgsz"))
    nb = len(model.yaml["backbone"])
    layers = list(model.model)
    count = lambda mods: int(sum(p.numel() for m in mods for p in m.parameters()))
    out = {"checkpoint": best, "train_imgsz": imgsz, "stride": [int(x) for x in model.stride.tolist()],
           "params_unfused": {"backbone": count(layers[:nb]), "neck": count(layers[nb:-1]), "head": count(layers[-1:]),
                              "total": count(layers)},
           "gflops_unfused": {"640": float(get_flops(model, 640)), str(imgsz): float(get_flops(model, imgsz))}}
    fused = copy.deepcopy(model).fuse(verbose=False)
    out["params_fused_total"] = int(sum(p.numel() for p in fused.parameters()))
    out["gflops_fused"] = {"640": float(get_flops(fused, 640)), str(imgsz): float(get_flops(fused, imgsz))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Cache root, e.g. $P3/evaluation/cache/ldpolyp")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tags", default=r"^runs__", help="Regex selecting cache tags (default: the replication runs).")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--cost", action="store_true", help="Also compute parameters and GFLOPs (needs torch, ultralytics).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)
    recs = load_caches(args.cache, args.split, args.tags)
    if not recs:
        sys.exit("ERROR: no caches match %s" % args.tags)
    z0 = np.load(recs[0]["_npz"])
    names, hw = [str(n) for n in z0["names"]], z0["orig_hw"]
    for r in recs[1:]:
        z = np.load(r["_npz"])
        if not ([str(n) for n in z["names"]] == names and np.array_equal(z["orig_hw"], hw)):
            sys.exit("ERROR: %s covers different images" % r["tag"])
    gt, n_boxes = build_gt(os.path.abspath(args.dataset), args.split, names, hw)
    ranges = {"small_lt_32sq": 0, "medium": 0, "large_gt_96sq": 0}
    for a in gt["annotations"]:
        ar = a["area"]
        ranges["small_lt_32sq" if ar < 32 ** 2 else ("medium" if ar <= 96 ** 2 else "large_gt_96sq")] += 1
    print("ground truth: %d images, %d boxes; COCO area ranges in pixels: %s" % (len(names), n_boxes, ranges))

    per_model, groups = {}, {}
    for r in recs:
        m, nd = coco_eval(gt, np.load(r["_npz"]))
        cfg = r["tag"].split("__", 1)[1].split("_seed")[0]
        per_model[r["tag"]] = dict(m, detections_used=nd, config=cfg, seed=r["train_args"].get("seed"))
        groups.setdefault(cfg, []).append(r["tag"])
        print("  %-28s " % r["tag"] + "  ".join("%s %.2f" % (k, m[k]) for k in METRICS))

    summary = {}
    for cfg, tags in groups.items():
        summary[cfg] = {k: {"n": len(tags), "mean": float(np.mean([per_model[t][k] for t in tags])),
                            "sd": float(np.std([per_model[t][k] for t in tags], ddof=1)) if len(tags) > 1 else None}
                        for k in METRICS}
    contrasts = {}
    if args.reference in groups:
        for cfg, tags in groups.items():
            if cfg == args.reference:
                continue
            contrasts[cfg + "_minus_" + args.reference] = {
                k: welch([per_model[t][k] for t in groups[args.reference]], [per_model[t][k] for t in tags]) for k in METRICS}

    timing = {}
    for cfg, tags in groups.items():
        tr = [next(r for r in recs if r["tag"] == t)["timing_ms_per_image"] for t in tags]
        timing[cfg] = {part: {"mean_of_model_means": float(np.mean([x[part]["mean"] for x in tr])),
                              "sd_of_model_means": float(np.std([x[part]["mean"] for x in tr], ddof=1)) if len(tr) > 1 else None,
                              "mean_without_first_image": float(np.mean([x[part]["mean_without_first"] for x in tr]))}
                       for part in ("preprocess", "inference", "postprocess")}
        timing[cfg]["imgsz"] = next(r for r in recs if r["tag"] == tags[0])["settings"]["imgsz"]
        timing[cfg]["gpu"] = next(r for r in recs if r["tag"] == tags[0])["versions"].get("gpu")

    cost = None
    if args.cost:
        cost = {}
        for cfg, tags in groups.items():
            r0 = next(r for r in recs if r["tag"] == tags[0])
            cost[cfg] = model_cost(r0["checkpoint"])
            totals = set()
            for t in tags:
                r = next(x for x in recs if x["tag"] == t)
                import torch
                totals.add(int(sum(p.numel() for p in torch.load(r["checkpoint"], map_location="cpu")["model"].parameters())))
            cost[cfg]["params_total_all_runs"] = sorted(totals)

    rec = {
        "analysis": "COCO-style AP and model cost, addendum v1.0 Section 10 items 6-7",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"split": args.split, "max_dets": 100, "iou_type": "bbox", "area_ranges": "COCO default, original pixels",
                     "values": "x100", "tags_regex": args.tags, "reference": args.reference},
        "inputs": {r["tag"]: {"npz_sha256": r["npz_sha256"], "checkpoint_sha256": r["checkpoint_sha256"]} for r in recs},
        "ground_truth": {"images": len(names), "boxes": n_boxes, "coco_area_ranges": ranges,
                         "labels_sha256": recs[0]["labels_sha256"]},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pycocotools": pkg_version("pycocotools"),
                     "torch": pkg_version("torch"), "ultralytics": pkg_version("ultralytics"), "platform": platform.platform()},
        "this_script_sha256": sha256_file(__file__),
        "per_model": per_model, "summary": summary, "contrasts_welch": contrasts, "timing_ms": timing, "model_cost": cost,
    }
    os.makedirs(args.out)
    with open(os.path.join(args.out, "ap_cost.json"), "w") as f:
        json.dump(rec, f, indent=2)
    lines = ["== COCO-style AP (x100), mean (SD) per configuration"]
    for cfg in groups:
        lines.append("  %-9s n=%d  " % (cfg, len(groups[cfg])) + "  ".join(
            "%s %.2f%s" % (k, summary[cfg][k]["mean"], " (%.2f)" % summary[cfg][k]["sd"] if summary[cfg][k]["sd"] is not None else "")
            for k in METRICS))
    for name, c in contrasts.items():
        lines.append("  %s (Welch 95%% CI):" % name)
        for k in METRICS:
            w = c[k]
            lines.append("     %-5s %s" % (k, "n/a (fewer than 2 runs)" if w is None else "%+.2f [%+.2f, %+.2f]" % (
                w["delta"], w["ci95"][0], w["ci95"][1]) if w["ci95"] else "%+.2f (SD 0)" % w["delta"]))
    lines.append("== inference time per test image (ms; mean of the model means)")
    for cfg in groups:
        t = timing[cfg]
        lines.append("  %-9s imgsz %s: preprocess %.2f, inference %.2f, postprocess %.2f (without the first image: inference %.2f) on %s" % (
            cfg, t["imgsz"], t["preprocess"]["mean_of_model_means"], t["inference"]["mean_of_model_means"],
            t["postprocess"]["mean_of_model_means"], t["inference"]["mean_without_first_image"], t["gpu"]))
    if cost:
        lines.append("== model cost")
        for cfg, c in cost.items():
            p = c["params_unfused"]
            lines.append("  %-9s params backbone %d, neck %d, head %d, total %d (fused %d; all runs %s); GFLOPs unfused %s, fused %s" % (
                cfg, p["backbone"], p["neck"], p["head"], p["total"], c["params_fused_total"], c["params_total_all_runs"],
                {k: round(v, 2) for k, v in c["gflops_unfused"].items()}, {k: round(v, 2) for k, v in c["gflops_fused"].items()}))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "ap_cost.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "ap_cost.json"))


if __name__ == "__main__":
    main()
