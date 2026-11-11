#!/usr/bin/env python3
"""A10 video-cluster bootstrap (Paper3 D14; frozen appendix v1.0 Section 10.8).

Exploratory supplementary analysis, CPU only, reads the G1 cached test-set
predictions. Resamples the 60 test videos with replacement and recomputes the
main comparison (retrained +P2 vs baseline, 16 models) per size stratum with
the same coverage-recall definition as the primary analysis.

Rules (frozen 2026-11-06):
  1. Resampling unit = test video (video id = image name up to the LAST '_').
  2. Each of the 10,000 resamples draws 60 videos with replacement; the same
     video sample (with multiplicities) is used for all 16 models and all
     three strata.
  3. Per-model per-stratum recall = hit boxes / boxes in the drawn videos;
     hit = coverage recall (confidence > 0.25, IoU >= 0.5); strata = frozen
     training-set terciles; Delta = mean(+P2 eight seeds) - mean(baseline
     eight seeds), identical to A1.
  4. Small stratum is primary; medium and large are supplementary.
  5. Report the full 60-video Delta and the 2.5/97.5 percentile interval over
     resamples; a resample with no boxes in a stratum is excluded and counted.
  6. Built-in check: full-set (each video once) recall must equal the G1
     cache replay through the published multiseed_eval.py.
  7. Intervals condition on the 16 trained models and only reflect test-video
     sampling variability; they do not change the main verdict.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

SEEDS = ["42", "43", "44", "45", "46", "47", "48", "49"]
STRATA = ("small", "medium", "large")
CONF = 0.25
IOU_MATCH = 0.5
NBOOT = 10000
RNG_SEED = 20261106
N_VIDEOS = 60
EXP_MS = "3cac5bf9bcbcf37844547bdc0430060ff13fc767940cd139a4bda99abab43559"


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


def load_cache_index(cache, dataset):
    """Replay the G1 cache index exactly as the G7 analysis does."""
    recs = {}
    for js in sorted(glob.glob(os.path.join(cache, "test", "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        if not sha256_file(r["_npz"]) == r["npz_sha256"]:
            sys.exit("ERROR: %s changed after it was written" % r["_npz"])
        if not os.path.realpath(r["dataset"]) == os.path.realpath(dataset):
            continue
        recs[r["checkpoint_sha256"]] = r
    return recs


def box_iou(b, bs):
    if len(bs) == 0:
        return np.array([])
    x1 = np.maximum(b[0], bs[:, 0])
    y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2])
    y2 = np.minimum(b[3], bs[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def video_id(image_path):
    stem = os.path.basename(image_path)[:-4]
    return stem.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PAPER3_ROOT")
    ap.add_argument("--repo", required=True, help="repo checkout")
    ap.add_argument("--tools", required=True, help="dir containing check_public_vs_cache.py")
    ap.add_argument("--cache", required=True, help="evaluation/cache/ldpolyp")
    ap.add_argument("--dataset", required=True, help="data/ldpolyp")
    ap.add_argument("--thresholds", required=True, help="data/size_thresholds.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit("ERROR: %s exists; refusing to overwrite" % args.out)

    cpc = load_module(os.path.join(args.tools, "check_public_vs_cache.py"), "cpc")
    recs = load_cache_index(args.cache, args.dataset)
    cpc.CachedYOLO.index = recs
    cpc.CachedYOLO.split = "test"
    cpc.CachedYOLO.dataset = os.path.realpath(args.dataset)

    thr = np.load(args.thresholds, allow_pickle=True).item()
    p33, p67 = float(thr["p33"]), float(thr["p67"])
    dataset_real = os.path.realpath(args.dataset)

    models = {}
    for side, pattern in (
        ("baseline", os.path.join("runs", "baseline_seed{seed}")),
        ("p2", os.path.join("runs", "p2_seed{seed}")),
    ):
        for seed in SEEDS:
            rel = os.path.join(pattern.format(seed=seed), "weights", "best.pt")
            abspath = os.path.join(args.root, rel)
            sha = sha256_file(abspath)
            rec = recs.get(sha)
            if rec is None:
                sys.exit("ERROR: no test cache for %s" % rel)
            if int(rec["train_args"]["seed"]) != int(seed):
                sys.exit("ERROR: cache of %s has seed mismatch" % rel)
            models[(side, seed)] = {
                "rel": rel,
                "abspath": abspath,
                "sha256": sha,
                "rec": rec,
                "imgsz": int(rec["settings"]["imgsz"]),
            }

    images = sorted(glob.glob(os.path.join(dataset_real, "images", "test", "*.jpg")))
    gt_by_video = {}
    for ip in images:
        vid = video_id(ip)
        name = os.path.basename(ip)[:-4]
        lp = os.path.join(dataset_real, "labels", "test", name + ".txt")
        gts = []
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w2, h = map(float, p)
                    ra = w2 * h
                    st = "small" if ra < p33 else ("medium" if ra < p67 else "large")
                    gts.append((cx, cy, w2, h, st))
        gt_by_video.setdefault(vid, []).append((ip, gts))
    videos = sorted(gt_by_video.keys())
    if len(videos) != N_VIDEOS:
        sys.exit("ERROR: expected %d test videos, found %d" % (N_VIDEOS, len(videos)))

    # per model, per video, per stratum: [hit, total]
    counts = {key: {v: {s: [0, 0] for s in STRATA} for v in videos} for key in models}
    full_recall = {}
    for key, info in models.items():
        m = cpc.CachedYOLO(info["abspath"])
        for vid, items in gt_by_video.items():
            for ip, gts in items:
                r = m.predict(ip, conf=CONF, imgsz=info["imgsz"], verbose=False)[0]
                W, H = r.orig_shape[1], r.orig_shape[0]
                pr = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                for cx, cy, w2, h, st in gts:
                    g = [(cx - w2 / 2) * W, (cy - h / 2) * H,
                         (cx + w2 / 2) * W, (cy + h / 2) * H]
                    io = box_iou(g, pr)
                    hit = 1 if (len(io) > 0 and float(io.max()) >= IOU_MATCH) else 0
                    counts[key][vid][st][0] += hit
                    counts[key][vid][st][1] += 1
        for s in STRATA:
            h = sum(counts[key][v][s][0] for v in videos)
            t = sum(counts[key][v][s][1] for v in videos)
            full_recall[(key, s)] = (h / t) if t > 0 else float("nan")

    # Built-in check against the published multiseed_eval.py replay.
    ms_path = os.path.join(args.repo, "evaluation", "multiseed_eval.py")
    if not sha256_file(ms_path) == EXP_MS:
        sys.exit("ERROR: multiseed_eval.py differs from the published version")
    ms = load_module(ms_path, "ms")
    ms.YOLO = cpc.CachedYOLO
    worst = 0.0
    for key, info in models.items():
        ref = ms.eval_size(info["abspath"], dataset_real,
                           p33, p67, info["imgsz"], CONF, IOU_MATCH)
        for s in STRATA:
            mine = full_recall[(key, s)]
            if not (np.isfinite(mine) and np.isfinite(ref[s])):
                sys.exit("ERROR: non-finite recall in built-in check for %s" % (key,))
            worst = max(worst, abs(mine - ref[s]))
    if worst > 1e-9:
        sys.exit("ERROR: built-in recall check failed (max abs diff %.3g)" % worst)

    # Bootstrap over videos.
    rng = np.random.default_rng(RNG_SEED)
    samples = rng.multinomial(N_VIDEOS, np.ones(N_VIDEOS) / N_VIDEOS, size=NBOOT)
    result = {"full": {}, "bootstrap": {}}
    for s in STRATA:
        HB = np.array([[counts[("baseline", seed)][v][s][0] for v in videos]
                       for seed in SEEDS], dtype=float)
        TB = np.array([[counts[("baseline", seed)][v][s][1] for v in videos]
                       for seed in SEEDS], dtype=float)
        HP = np.array([[counts[("p2", seed)][v][s][0] for v in videos]
                       for seed in SEEDS], dtype=float)
        TP = np.array([[counts[("p2", seed)][v][s][1] for v in videos]
                       for seed in SEEDS], dtype=float)

        def recalls(H, T):
            denom = T @ samples.T
            num = H @ samples.T
            out = num / np.where(denom > 0, denom, np.nan)
            return out

        RB = recalls(HB, TB)  # 8 x NBOOT
        RP = recalls(HP, TP)
        delta = RP.mean(axis=0) - RB.mean(axis=0)
        excluded = int(np.isnan(delta).sum())
        if excluded:
            delta = delta[np.isfinite(delta)]
        lo, hi = np.percentile(delta, [2.5, 97.5]) if len(delta) else (np.nan, np.nan)

        fb = np.mean([full_recall[("baseline", seed), s] for seed in SEEDS])
        fp = np.mean([full_recall[("p2", seed), s] for seed in SEEDS])
        full_delta = fp - fb
        result["full"][s] = {
            "baseline_mean_recall": float(fb),
            "p2_mean_recall": float(fp),
            "delta": float(full_delta),
            "delta_pp": float(100 * full_delta),
        }
        result["bootstrap"][s] = {
            "delta_pp_2p5": float(100 * lo),
            "delta_pp_97p5": float(100 * hi),
            "resamples": NBOOT,
            "excluded_no_boxes": excluded,
        }

    os.makedirs(args.out)
    record = {
        "analysis": "A10 video-cluster bootstrap (D14, appendix 10.8; exploratory)",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "conf": CONF, "iou_match": IOU_MATCH, "n_boot": NBOOT,
            "rng_seed": RNG_SEED, "n_videos": N_VIDEOS, "seeds": SEEDS,
            "p33": p33, "p67": p67,
        },
        "inputs_sha256": {
            "multiseed_eval.py": sha256_file(ms_path),
            "thresholds": sha256_file(args.thresholds),
            "this_script": sha256_file(__file__),
        },
        "models": {
            "%s_seed%s" % (side, seed): {
                "checkpoint": info["rel"], "checkpoint_sha256": info["sha256"],
                "imgsz": info["imgsz"],
                "full_recall": {s: float(full_recall[((side, seed), s)]) for s in STRATA},
            }
            for (side, seed), info in models.items()
        },
        "result": result,
    }
    with open(os.path.join(args.out, "a10.json"), "w") as f:
        json.dump(record, f, indent=2)

    lines = ["A10 video-cluster bootstrap (D14; exploratory); recall Delta in pp", ""]
    for s in STRATA:
        lines.append("%-6s full Delta %+.2f pp; bootstrap 2.5%%/97.5%% = [%+.2f, %+.2f] pp (%d resamples, %d excluded)"
                     % (s, result["full"][s]["delta_pp"],
                        result["bootstrap"][s]["delta_pp_2p5"],
                        result["bootstrap"][s]["delta_pp_97p5"],
                        NBOOT, result["bootstrap"][s]["excluded_no_boxes"]))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "a10.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print("written:", os.path.join(args.out, "a10.json"))


if __name__ == "__main__":
    main()
