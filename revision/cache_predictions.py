"""One-time inference cache (Paper3 revision analysis addendum v1.0, Section 4).

Each checkpoint predicts every image of each requested split once, at confidence 0.001,
at its training input size (read from the checkpoint), with the Ultralytics default NMS
(IoU 0.7, at most 300 detections per image). Images are taken in the order the public
evaluation scripts use: sorted(glob(<dataset>/images/<split>/*.jpg)), one image per call.

Per (model, split) the script writes
  <out>/<split>/<tag>.npz   names, orig_hw, offsets, xyxy (float32, original pixels),
                            conf (float32), cls, speed_ms (preprocess, inference, postprocess)
  <out>/<split>/<tag>.json  settings, input hashes, versions, GPU, timing summary
where <tag> = <project folder>__<run folder>, e.g. runs__baseline_seed42.

A cache that exists is never regenerated or overwritten (the cache is generated once).
The script refuses to run while a training process is running, because the timings and
the GPU memory would be shared with training.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONF, IOU, MAX_DET = 0.001, 0.7, 300
TRAIN_RE = re.compile(r"^[^ ]*python[0-9.]* .*train_paper3\.py( |$)")  # a python process running the training script


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_processes():
    found = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(os.path.join(d, "cmdline"), "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if TRAIN_RE.search(cmd) and not str(os.getpid()) == os.path.basename(d):
            found.append("%s %s" % (os.path.basename(d), cmd.strip()[:160]))
    return found


def gpu_apps():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_root_of(data_yaml):
    """Directory a training data.yaml points to (its 'path' key, else its own folder)."""
    try:
        import yaml
        d = yaml.safe_load(open(data_yaml)) or {}
        p = d.get("path")
    except Exception:
        p = None
    base = Path(data_yaml).parent
    if not p:
        return base.resolve()
    p = Path(p)
    return (p if p.is_absolute() else base / p).resolve()


def label_digest(dataset, split, names):
    parts = []
    for n in names:
        lp = os.path.join(dataset, "labels", split, n[:-4] + ".txt")
        parts.append("%s\t%s\n" % (n, open(lp).read() if os.path.exists(lp) else "<no label file>"))
    return sha256_text("".join(parts))


def versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch
        v.update(torch=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        v.update(torch=None, cuda=None, cudnn=None, gpu=None)
    try:
        import ultralytics
        v["ultralytics"] = ultralytics.__version__
    except Exception:
        v["ultralytics"] = None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Run folders, each with weights/best.pt.")
    ap.add_argument("--dataset", required=True, help="YOLO dataset folder with images/<split> and labels/<split>.")
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--out", required=True, help="Cache root, e.g. $P3/evaluation/cache/ldpolyp")
    ap.add_argument("--doi", required=True, help="Figshare DOI of the frozen addendum (recorded in every cache).")
    ap.add_argument("--dry-run", action="store_true", help="Check inputs and print the plan; no inference.")
    args = ap.parse_args()

    busy = training_processes()
    if busy and not args.dry_run:
        sys.exit("ERROR: a training process is running; pause the queue first:\n  " + "\n  ".join(busy))
    if busy:
        print("note: training is running (allowed in a dry run, which does no inference)")
    dataset = str(Path(args.dataset).resolve())
    lists = {}
    for s in args.splits:
        names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(dataset, "images", s, "*.jpg")))]
        others = [p for p in os.listdir(os.path.join(dataset, "images", s)) if not p.endswith((".jpg", ".npy"))]
        if not names:
            sys.exit("ERROR: no .jpg images in %s/images/%s" % (dataset, s))
        if others:
            print("WARNING: %d non-.jpg files in images/%s are ignored, as in the public scripts: %s" % (len(others), s, others[:3]))
        lists[s] = names
        print("split %-5s %d images (sorted *.jpg, as in the public scripts)" % (s, len(names)))

    from ultralytics import YOLO  # imported after the checks

    plan = []
    for rd in args.run_dirs:
        rd = Path(rd).resolve()
        best = rd / "weights" / "best.pt"
        if not best.exists() or not (rd / "recovery_metadata.json").exists():
            sys.exit("ERROR: %s is not a finished run (best.pt or recovery_metadata.json missing)" % rd)
        tag = "%s__%s" % (rd.parent.name, rd.name)
        m = YOLO(str(best))
        targs = (m.ckpt or {}).get("train_args", {})
        imgsz = targs.get("imgsz")
        if not isinstance(imgsz, int):
            sys.exit("ERROR: %s: training input size not found in the checkpoint (%r)" % (tag, imgsz))
        droot = dataset_root_of(targs.get("data", ""))
        if not str(droot) == dataset:
            sys.exit("ERROR: %s was trained on %s, not on %s" % (tag, droot, dataset))
        plan.append((tag, rd, best, m, imgsz, targs))
        todo = [s for s in args.splits if not (Path(args.out) / s / (tag + ".npz")).exists()]
        print("model %-34s imgsz %4s seed %s -> to cache: %s" % (tag, imgsz, targs.get("seed"), todo or "nothing (exists)"))
    if args.dry_run:
        print("dry run: no inference")
        return

    ver = versions()
    this_sha = sha256_file(__file__)
    for tag, rd, best, m, imgsz, targs in plan:
        best_sha = sha256_file(best)
        for s in args.splits:
            od = Path(args.out) / s
            od.mkdir(parents=True, exist_ok=True)
            npz, js = od / (tag + ".npz"), od / (tag + ".json")
            if npz.exists():
                print("exists, not regenerated: %s" % npz)
                continue
            names = lists[s]
            apps_before = gpu_apps()
            t0 = datetime.now(timezone.utc).isoformat()
            w0 = time.time()
            hw, offsets, xyxy, conf, cls, speed = [], [0], [], [], [], []
            for n in names:
                r = m.predict(os.path.join(dataset, "images", s, n), conf=CONF, iou=IOU, max_det=MAX_DET,
                              imgsz=imgsz, verbose=False)[0]
                b = r.boxes
                bx = b.xyxy.cpu().numpy().astype(np.float32, copy=False) if b is not None else np.zeros((0, 4), np.float32)
                cf = b.conf.cpu().numpy().astype(np.float32, copy=False) if b is not None else np.zeros((0,), np.float32)
                cl = b.cls.cpu().numpy().astype(np.int16) if b is not None else np.zeros((0,), np.int16)
                hw.append(r.orig_shape[:2])
                xyxy.append(bx)
                conf.append(cf)
                cls.append(cl)
                offsets.append(offsets[-1] + len(cf))
                speed.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
            wall = time.time() - w0
            arr = dict(names=np.array(names), orig_hw=np.array(hw, dtype=np.int32), offsets=np.array(offsets, dtype=np.int64),
                       xyxy=np.concatenate(xyxy).reshape(-1, 4), conf=np.concatenate(conf), cls=np.concatenate(cls),
                       speed_ms=np.array(speed, dtype=np.float64))
            tmp = od / (tag + ".tmp.npz")
            np.savez_compressed(tmp, **arr)
            per_img = np.diff(arr["offsets"])
            sp = arr["speed_ms"]
            rec = {
                "tag": tag, "run_dir": str(rd), "checkpoint": str(best), "checkpoint_sha256": best_sha,
                "checkpoint_bytes": best.stat().st_size,
                "train_args": {k: targs.get(k) for k in ("model", "data", "imgsz", "seed", "epochs", "batch", "patience")},
                "split": s, "dataset": dataset, "n_images": len(names),
                "image_list_sha256": sha256_text("\n".join(names)),
                "labels_sha256": label_digest(dataset, s, names),
                "settings": {"conf": CONF, "iou": IOU, "max_det": MAX_DET, "imgsz": imgsz, "batch": 1, "half": False,
                             "image_order": "sorted(glob(images/<split>/*.jpg))"},
                "n_boxes": int(len(arr["conf"])), "max_boxes_per_image": int(per_img.max()) if len(per_img) else 0,
                "images_at_max_det": int((per_img >= MAX_DET).sum()),
                "boxes_with_conf_exactly_0.25": int((arr["conf"] == np.float32(0.25)).sum()),
                "timing_ms_per_image": {k: {"mean": float(sp[:, i].mean()), "median": float(np.median(sp[:, i])),
                                            "mean_without_first": float(sp[1:, i].mean()) if len(sp) > 1 else None}
                                        for i, k in enumerate(("preprocess", "inference", "postprocess"))},
                "wall_seconds": wall,
                "gpu_apps_before": apps_before,
                "started_utc": t0, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(), "platform": platform.platform(),
                "versions": ver, "addendum_doi": args.doi, "script_sha256": this_sha,
                "npz_sha256": sha256_file(tmp),
            }
            os.replace(tmp, npz)
            with open(js, "w") as f:
                json.dump(rec, f, indent=2)
            print("cached %-34s %-5s %4d images %6d boxes, inference %.1f ms/image, %.0f s" % (
                tag, s, len(names), rec["n_boxes"], rec["timing_ms_per_image"]["inference"]["mean"], wall))


if __name__ == "__main__":
    main()
