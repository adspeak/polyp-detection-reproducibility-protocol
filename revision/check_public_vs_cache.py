"""Exact-match check of the cache against the public evaluation scripts (addendum v1.0, Section 4).

The public scripts evaluation/multiseed_eval.py and evaluation/crossover_fp_analysis.py were
run unchanged (real inference at confidence 0.25). Here the same scripts are imported from the
repository and run again with the same arguments, with one change: their YOLO is replaced by
a reader of the confidence-0.001 cache that keeps the boxes with confidence > the requested
threshold (Ultralytics NMS keeps conf > conf_thres). Every number they compute (recall per
stratum, false-positive counts, crossovers) is then compared with the real run exactly.

Each checkpoint passed to the scripts is matched to its cache through the SHA-256 of the file,
so the cache that is replayed is the cache of that exact checkpoint.
"""

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Arr(object):
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes(object):
    def __init__(self, xyxy, conf):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)


class _Result(object):
    def __init__(self, hw, xyxy, conf):
        self.orig_shape = (int(hw[0]), int(hw[1]))
        self.boxes = _Boxes(xyxy, conf)


class CachedYOLO(object):
    """Stands in for ultralytics.YOLO inside the public scripts."""
    index = {}      # checkpoint sha256 -> cache record
    split = "test"
    dataset = None
    used = {}       # tag -> set of image names requested

    def __init__(self, weight):
        sha = sha256_file(weight)
        if sha not in CachedYOLO.index:
            raise RuntimeError("no %s cache for checkpoint %s (sha256 %s)" % (CachedYOLO.split, weight, sha[:16]))
        self.rec = CachedYOLO.index[sha]
        z = np.load(self.rec["_npz"])
        names, off = [str(n) for n in z["names"]], z["offsets"]
        xyxy, conf, hw = z["xyxy"], z["conf"], z["orig_hw"]
        self.by_name = {n: (hw[i], xyxy[off[i]:off[i + 1]], conf[off[i]:off[i + 1]]) for i, n in enumerate(names)}
        CachedYOLO.used.setdefault(self.rec["tag"], set())

    def predict(self, source, conf=0.25, imgsz=None, verbose=False, **kw):
        if kw:
            raise RuntimeError("unexpected predict arguments %s" % sorted(kw))
        if not imgsz == self.rec["settings"]["imgsz"]:
            raise RuntimeError("%s: imgsz %s requested, cache made at %s" % (self.rec["tag"], imgsz, self.rec["settings"]["imgsz"]))
        if conf < self.rec["settings"]["conf"]:
            raise RuntimeError("confidence %s is below the cache threshold" % conf)
        want = os.path.join(CachedYOLO.dataset, "images", CachedYOLO.split)
        if not os.path.realpath(os.path.dirname(source)) == os.path.realpath(want):
            raise RuntimeError("image %s is not in %s" % (source, want))
        name = os.path.basename(source)
        hw, b, c = self.by_name[name]
        keep = c > conf
        CachedYOLO.used[self.rec["tag"]].add(name)
        return [_Result(hw, b[keep], c[keep])]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def diff(a, b, path=""):
    """List of (path, real, replay) where the two JSON values differ (NaN equals NaN)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + "/" + k, a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                out += diff(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if not len(a) == len(b):
            out.append((path, "len %d" % len(a), "len %d" % len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, "%s[%d]" % (path, i))
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif not (type(a) == type(b) and a == b):
        out.append((path, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cache", required=True, help="Cache root (contains <split>/<tag>.npz and .json).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--jobs", required=True, help="jobs.json written when the public scripts were run.")
    ap.add_argument("--out", required=True, help="Folder for the replayed outputs and the check record.")
    args = ap.parse_args()

    os.makedirs(args.out)
    jobs = json.load(open(args.jobs))
    recs = []
    for js in sorted(glob.glob(os.path.join(args.cache, args.split, "*.json"))):
        r = json.load(open(js))
        r["_npz"] = js[:-5] + ".npz"
        if not sha256_file(r["_npz"]) == r["npz_sha256"]:
            sys.exit("ERROR: %s changed after it was written" % r["_npz"])
        recs.append(r)
    CachedYOLO.index = {r["checkpoint_sha256"]: r for r in recs}
    CachedYOLO.split = args.split
    datasets = sorted(set(r["dataset"] for r in recs))
    if not len(datasets) == 1:
        sys.exit("ERROR: caches from more than one dataset: %s" % datasets)
    CachedYOLO.dataset = datasets[0]

    # boxes at or near the 0.25 threshold, where > and >= could differ
    near = {"exactly_0.25": 0, "within_1e-6": 0}
    for r in recs:
        c = np.load(r["_npz"])["conf"]
        near["exactly_0.25"] += int((c == np.float32(0.25)).sum())
        near["within_1e-6"] += int((np.abs(c.astype(np.float64) - 0.25) < 1e-6).sum())

    scripts = {"multiseed": os.path.join(args.repo, "evaluation", "multiseed_eval.py"),
               "crossover": os.path.join(args.repo, "evaluation", "crossover_fp_analysis.py")}
    results, all_ok = [], True
    for i, job in enumerate(jobs):
        script = scripts[job["script"]]
        if not sha256_file(script) == job["script_sha256"]:
            sys.exit("ERROR: %s differs from the script used for the real run" % script)
        mod = load_module(script, "public_%s_%d" % (job["script"], i))
        mod.YOLO = CachedYOLO
        replay = os.path.join(args.out, os.path.basename(job["out"]))
        argv = list(job["argv"])
        argv[argv.index("--out") + 1] = replay
        buf = io.StringIO()
        old = sys.argv
        sys.argv = [script] + argv
        try:
            with contextlib.redirect_stdout(buf):
                mod.main()
        finally:
            sys.argv = old
        with open(replay[:-5] + ".log", "w") as f:
            f.write(buf.getvalue())
        real = json.load(open(job["out"]))
        rep = json.load(open(replay))
        d = diff(real, rep)
        ok = not d
        all_ok = all_ok and ok
        results.append({"job": job["name"], "real": job["out"], "real_sha256": sha256_file(job["out"]),
                        "replay": replay, "identical": ok, "differences": [[p, repr(a), repr(b)] for p, a, b in d[:50]],
                        "n_differences": len(d)})
        print("%-22s %s" % (job["name"], "identical" if ok else "DIFFERENT in %d values, e.g. %s" % (len(d), d[:3])))

    unused = {t: len(v) for t, v in CachedYOLO.used.items()}
    rec = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "replayed = public scripts with YOLO replaced by the conf-0.001 cache, boxes kept if conf > threshold",
        "split": args.split, "dataset": CachedYOLO.dataset,
        "caches": {r["tag"]: {"npz_sha256": r["npz_sha256"], "checkpoint_sha256": r["checkpoint_sha256"]} for r in recs},
        "images_requested_per_model": unused,
        "boxes_near_threshold": near,
        "scripts_sha256": {k: sha256_file(v) for k, v in scripts.items()},
        "this_script_sha256": sha256_file(__file__),
        "jobs": results, "all_identical": all_ok,
    }
    with open(os.path.join(args.out, "check_record.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("boxes with confidence exactly 0.25: %d; within 1e-6 of 0.25: %d" % (near["exactly_0.25"], near["within_1e-6"]))
    print("images requested per model:", sorted(set(unused.values())))
    print("RESULT:", "all identical" if all_ok else "MISMATCH - resolve before any further analysis")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
