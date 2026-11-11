"""Build the Kvasir-SEG YOLO dataset for the PeerJ revision (addendum v1.0, Section 11).

Membership: the role column of Paper2's split table (newsplit_20261002_152404.csv),
Kvasir-SEG rows only (700 train / 150 val / 150 test; 666 groups, none across roles).
Boxes: one box per image, produced by mask_to_yolo() imported unchanged from the
original tools/prepare_kvasir_seg.py, so the box rule is exactly the original one.
Strata: terciles of the 700 training boxes, stored in size_thresholds.npy with the
same keys as data_preparation/compute_size_thresholds.py writes.

The script never overwrites an existing output directory and never touches the
existing kvasir_yolo dataset.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

EXPECT_ROLES = {"train": 700, "val": 150, "test": 150}
FOV_MM = 35.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask_to_yolo(prepare_script):
    spec = importlib.util.spec_from_file_location("prepare_kvasir_seg_original", prepare_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mask_to_yolo


def n_components(mask_path, image_path):
    """8-connected components of mask > 127, after the same resize as mask_to_yolo."""
    mask = np.asarray(Image.open(mask_path).convert("L"))
    image = Image.open(image_path)
    if not mask.shape == (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize(image.size, resample=Image.Resampling.NEAREST))
    _, n = ndimage.label(mask > 127, structure=np.ones((3, 3), dtype=int))
    return int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvasir-root", required=True, help="Directory with images/ and masks/ (or a Kvasir-SEG/ subfolder).")
    ap.add_argument("--assign", required=True, help="Paper2 split table (newsplit_20261002_152404.csv).")
    ap.add_argument("--expect-md5", required=True, help="MD5 of the split table fixed in the addendum.")
    ap.add_argument("--prepare-script", required=True, help="Original tools/prepare_kvasir_seg.py (box rule).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit("ERROR: %s already exists; refusing to overwrite" % out)
    got = md5(args.assign)
    if not got == args.expect_md5:
        sys.exit("ERROR: split table MD5 %s does not match the addendum (%s)" % (got, args.expect_md5))

    src = Path(args.kvasir_root)
    if (src / "Kvasir-SEG").is_dir():
        src = src / "Kvasir-SEG"
    images_dir, masks_dir = src / "images", src / "masks"
    if not (images_dir.is_dir() and masks_dir.is_dir()):
        sys.exit("ERROR: expected %s and %s" % (images_dir, masks_dir))

    with open(args.assign, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seg = [r for r in rows if str(r["id"]).startswith("seg/") and r["role"] in EXPECT_ROLES]
    roles = dict(Counter(r["role"] for r in seg))
    groups = {}
    for r in seg:
        groups.setdefault(r["group"], set()).add(r["role"])
    cross = sum(1 for v in groups.values() if len(v) > 1)
    names = [r["name"] for r in seg]
    print("split table: %d Kvasir-SEG images %s, %d groups, %d across roles" % (len(seg), roles, len(groups), cross))
    if not roles == EXPECT_ROLES or cross or not len(set(names)) == len(names):
        sys.exit("ERROR: split counts, duplicate names or groups differ from the addendum")

    mask_to_yolo = load_mask_to_yolo(args.prepare_script)
    bad_md5, missing = [], []
    for r in seg:
        img = images_dir / r["name"]
        msk = masks_dir / (Path(r["name"]).stem + ".jpg")
        if not img.exists() or not msk.exists():
            missing.append(r["name"])
        elif not md5(img) == r["md5"]:
            bad_md5.append(r["name"])
    print("images or masks missing: %d; image MD5 different from the split table: %d" % (len(missing), len(bad_md5)))
    if missing or bad_md5:
        print("  examples:", (missing + bad_md5)[:5])
        sys.exit("ERROR: source images do not match the split table")

    for s in EXPECT_ROLES:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    areas = {s: [] for s in EXPECT_ROLES}
    splits = {s: [] for s in EXPECT_ROLES}
    multi = Counter()        # images with more than one 8-connected component (recounted here)
    multi_p2 = Counter()     # images with n_boxes > 1 in the Paper2 table (data_v3 labels)
    agree = Counter()        # recount == Paper2 n_boxes
    empty = []
    for r in sorted(seg, key=lambda r: r["name"]):
        s, name = r["role"], r["name"]
        img = images_dir / name
        msk = masks_dir / (Path(name).stem + ".jpg")
        shutil.copy2(img, out / "images" / s / name)
        label = mask_to_yolo(msk, img)
        (out / "labels" / s / (Path(name).stem + ".txt")).write_text((label + "\n") if label else "", encoding="ascii")
        splits[s].append(Path(name).stem)
        nc = n_components(msk, img)
        multi[s] += int(nc > 1)
        nb = str(r.get("n_boxes", "")).strip()
        if nb:
            multi_p2[s] += int(int(float(nb)) > 1)
            agree[s] += int(int(float(nb)) == nc)
        if label is None:
            empty.append(name)
            continue
        p = label.split()
        areas[s].append(float(p[3]) * float(p[4]))
    if empty:
        print("WARNING: images whose mask has no foreground (empty label file):", empty)

    tr = np.array(areas["train"])
    p33, p67 = float(np.percentile(tr, 100 / 3)), float(np.percentile(tr, 200 / 3))
    c5, c10 = (5 / FOV_MM) ** 2, (10 / FOV_MM) ** 2
    thr = {
        "p33": p33, "p67": p67,
        "clinical_diminutive": float(c5), "clinical_large": float(c10),
        "n_boxes_train": int(len(tr)),
        "bin_counts": {"small": int((tr < p33).sum()), "medium": int(((tr >= p33) & (tr < p67)).sum()),
                       "large": int((tr >= p67).sum())},
        "frac_below_5mm": float((tr < c5).mean()),
        "fov_mm": FOV_MM,
        "source": "terciles of the Kvasir-SEG training boxes, data_v3 roles (addendum v1.0, Section 11)",
    }
    np.save(out / "size_thresholds.npy", thr, allow_pickle=True)
    counts = {}
    for s in EXPECT_ROLES:
        a = np.array(areas[s])
        counts[s] = {"images": len(splits[s]), "boxes": int(len(a)),
                     "small": int((a < p33).sum()), "medium": int(((a >= p33) & (a < p67)).sum()),
                     "large": int((a >= p67).sum()), "below_5mm_F35": int((a < c5).sum()),
                     "images_more_than_one_component_recount": int(multi[s]),
                     "images_more_than_one_box_paper2_table": int(multi_p2[s]),
                     "images_recount_equals_paper2_n_boxes": int(agree[s])}
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n- polyp\n" % out.resolve(),
        encoding="utf-8")
    labels_sha = {s: hashlib.sha256("".join(
        "%s\t%s\n" % (stem, (out / "labels" / s / (stem + ".txt")).read_text(encoding="ascii"))
        for stem in splits[s]).encode()).hexdigest() for s in EXPECT_ROLES}
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "addendum": "Paper3 revision analysis addendum v1.0, Section 11",
        "split_table": {"path": str(Path(args.assign).resolve()), "md5": got, "sha256": sha256(args.assign)},
        "kvasir_root": str(src.resolve()),
        "box_rule": {"function": "mask_to_yolo from the original prepare script", "script": str(Path(args.prepare_script).resolve()),
                     "script_sha256": sha256(args.prepare_script)},
        "this_script_sha256": sha256(__file__),
        "component_rule": "scipy.ndimage.label on mask > 127, 8-connectivity, no minimum area",
        "groups": len(groups), "groups_across_roles": cross,
        "image_md5_checked_against_split_table": len(seg),
        "labels_sha256_by_split": labels_sha,
        "thresholds": thr, "counts": counts, "splits": splits,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("thresholds: p33 = %.6f, p67 = %.6f (from %d training boxes; train strata %s)" % (
        p33, p67, len(tr), thr["bin_counts"]))
    for s in EXPECT_ROLES:
        c = counts[s]
        print("  %-5s images %3d boxes %3d | small %3d medium %3d large %3d | below 5 mm %3d | "
              ">1 component: recount %2d, Paper2 table %2d; recount = Paper2 n_boxes in %3d" % (
                  s, c["images"], c["boxes"], c["small"], c["medium"], c["large"], c["below_5mm_F35"],
                  c["images_more_than_one_component_recount"], c["images_more_than_one_box_paper2_table"],
                  c["images_recount_equals_paper2_n_boxes"]))
    print("output:", out.resolve())


if __name__ == "__main__":
    main()
