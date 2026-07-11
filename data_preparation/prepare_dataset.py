"""
Prepare the LDPolypVideo dataset for detection experiments.

This script converts the official LDPolypVideo release into a YOLO-format
detection dataset, using a VIDEO-LEVEL train/val/test split to prevent
frame-level leakage (near-duplicate frames of the same lesion must not appear
in both training and test sets). This corresponds to Section 4.2 of the paper.

Key settings (fixed for reproducibility):
    - Random seed: 42 (controls the train/val video split)
    - Frame sampling stride: 10 (one frame in every ten per video)
    - Val ratio: 0.15 of the TrainValid videos are held out for validation
    - The 60 official Test videos are kept intact as the test set

Expected input layout (official LDPolypVideo structure):
    <ldpolyp_root>/
        TrainValid/Images/<video>/*.jpg
        TrainValid/Annotations/<video>/*.txt
        Test/Images/<video>/*.jpg
        Test/Annotations/<video>/*.txt

Each annotation .txt begins with an integer count k, followed by k lines of
"x1 y1 x2 y2" absolute pixel coordinates.

Usage:
    python prepare_dataset.py --ldpolyp-root /path/to/LDPolypVideo --out ./ldpolyp
"""

import argparse
import glob
import os
import random
import shutil

import yaml
from PIL import Image

SEED = 42
STRIDE = 10
VAL_RATIO = 0.15


def list_videos(base, split):
    d = os.path.join(base, split, "Images")
    return sorted(v for v in os.listdir(d) if os.path.isdir(os.path.join(d, v)))


def convert(base, split, video_set, target, out):
    """Convert one split's videos into YOLO-format images + labels under `out`."""
    img_root = os.path.join(base, split, "Images")
    ann_root = os.path.join(base, split, "Annotations")
    n_img = n_polyp = n_empty = n_box = n_bad_box = 0

    for v in sorted(video_set):
        frames = sorted(glob.glob(os.path.join(img_root, v, "*.jpg")))[::STRIDE]
        for ip in frames:
            stem = os.path.splitext(os.path.basename(ip))[0]
            ap = os.path.join(ann_root, v, stem + ".txt")
            try:
                W, H = Image.open(ip).size
            except Exception:
                continue

            lines = []
            if os.path.exists(ap):
                raw = [x.strip() for x in open(ap) if x.strip()]
                if raw:
                    try:
                        k = int(raw[0])
                    except ValueError:
                        k = 0
                    for b in raw[1 : 1 + k]:
                        p = b.split()
                        if len(p) < 4:
                            continue
                        x1, y1, x2, y2 = map(float, p[:4])
                        # normalize + clip to [0, 1] + drop degenerate boxes
                        xmin, xmax = sorted([x1, x2])
                        ymin, ymax = sorted([y1, y2])
                        xmin = max(0, min(xmin, W))
                        xmax = max(0, min(xmax, W))
                        ymin = max(0, min(ymin, H))
                        ymax = max(0, min(ymax, H))
                        bw, bh = xmax - xmin, ymax - ymin
                        if bw < 1 or bh < 1:
                            n_bad_box += 1
                            continue
                        cx = (xmin + xmax) / 2 / W
                        cy = (ymin + ymax) / 2 / H
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")

            # unique filename: <video>_<frame> to avoid same-named frames colliding
            name = f"{v}_{stem}"
            shutil.copy(ip, os.path.join(out, f"images/{target}", name + ".jpg"))
            with open(os.path.join(out, f"labels/{target}", name + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_img += 1
            n_box += len(lines)
            n_polyp += len(lines) > 0
            n_empty += len(lines) == 0

    print(
        f"[{target}] images={n_img}  with_polyp={n_polyp}  empty={n_empty}  "
        f"boxes={n_box}  dropped_degenerate={n_bad_box}"
    )
    return n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldpolyp-root", required=True,
                    help="Path to the official LDPolypVideo release root.")
    ap.add_argument("--out", default="./ldpolyp",
                    help="Output directory for the YOLO-format dataset.")
    args = ap.parse_args()

    random.seed(SEED)
    base, out = args.ldpolyp_root, args.out

    if os.path.exists(out):
        shutil.rmtree(out)
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # video-level split (leakage control)
    tv_videos = list_videos(base, "TrainValid")
    random.shuffle(tv_videos)
    n_val = max(1, int(len(tv_videos) * VAL_RATIO))
    val_vids = set(tv_videos[:n_val])
    train_vids = set(tv_videos[n_val:])
    test_vids = list_videos(base, "Test")
    print(f"Split: train_videos={len(train_vids)}  val_videos={len(val_vids)}  "
          f"test_videos={len(test_vids)}")

    convert(base, "TrainValid", train_vids, "train", out)
    convert(base, "TrainValid", val_vids, "val", out)
    convert(base, "Test", test_vids, "test", out)

    data = {"path": os.path.abspath(out), "train": "images/train",
            "val": "images/val", "test": "images/test",
            "nc": 1, "names": ["polyp"]}
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print("\ndata.yaml written:")
    print(open(os.path.join(out, "data.yaml")).read())


if __name__ == "__main__":
    main()
