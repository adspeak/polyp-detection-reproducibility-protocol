#!/usr/bin/env python3
"""Rebuild the Kvasir-SEG 800/200 YOLO dataset used in the paper.

The historical project records identify the split used by the authors:
shuffle the sorted image/mask stems with seed 42, use the first 800 for
training and the last 200 for validation/evaluation. The recorded example
label also shows that boxes use exclusive mask bounds:

cx=(xmin+xmax)/2, cy=(ymin+ymax)/2, w=xmax-xmin, h=ymax-ymin.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_image(images_dir, stem):
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for mask stem {stem}")


def mask_to_yolo(mask_path, image_path):
    mask = np.asarray(Image.open(mask_path).convert("L"))
    image = Image.open(image_path)
    if mask.shape != (image.height, image.width):
        mask_image = Image.fromarray(mask).resize(
            image.size, resample=Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image)

    y, x = np.where(mask > 127)
    if not len(x):
        return None

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    width, height = image.size
    box_width = xmax - xmin
    box_height = ymax - ymin
    cx = (xmin + xmax) / 2 / width
    cy = (ymin + ymax) / 2 / height
    return f"0 {cx:.8f} {cy:.8f} {box_width / width:.8f} {box_height / height:.8f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kvasir-root",
        required=True,
        help="Directory containing Kvasir-SEG/images and Kvasir-SEG/masks.",
    )
    parser.add_argument("--out", required=True, help="Output YOLO dataset.")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0:
        raise ValueError("train-frac must be positive and val-frac non-negative")
    if args.train_frac + args.val_frac > 1:
        raise ValueError("train-frac + val-frac must not exceed 1")

    source = Path(args.kvasir_root)
    nested = source / "Kvasir-SEG"
    if nested.exists():
        source = nested
    images_dir = source / "images"
    masks_dir = source / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {source}/images and {source}/masks"
        )

    masks = sorted(masks_dir.glob("*.jpg"))
    if not masks:
        raise FileNotFoundError(f"No .jpg masks found under {masks_dir}")

    names = [path.stem for path in masks]
    random.Random(args.seed).shuffle(names)
    n_total = len(names)
    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)
    splits = {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in splits:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    counts = {}
    for split, stems in splits.items():
        box_count = 0
        for stem in stems:
            image_path = find_image(images_dir, stem)
            shutil.copy2(image_path, out / "images" / split / image_path.name)
            label = mask_to_yolo(masks_dir / f"{stem}.jpg", image_path)
            label_path = out / "labels" / split / f"{stem}.txt"
            label_path.write_text((label + "\n") if label else "", encoding="ascii")
            box_count += int(label is not None)
        counts[split] = {"images": len(stems), "boxes": box_count}

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["polyp"],
    }
    if splits["test"]:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "status": "reconstruction from recovered historical split rules",
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "counts": counts,
        "splits": splits,
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Output: {out.resolve()}")
    for split, counts_for_split in counts.items():
        if not counts_for_split["images"]:
            continue
        print(
            f"  {split:5s}: {counts_for_split['images']} images, "
            f"{counts_for_split['boxes']} boxes"
        )
    print(f"Manifest: {(out / 'split_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
