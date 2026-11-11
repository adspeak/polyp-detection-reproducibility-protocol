#!/usr/bin/env python3
"""Evaluate Kvasir-SEG seed-42 detection weights by training-set tertiles."""

import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.asarray([])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (
        boxes[:, 3] - boxes[:, 1]
    )
    return inter / (area_box + area_boxes - inter + 1e-9)


def evaluate(weight, dataset, p33, p67, split, imgsz, conf, iou_match):
    model = YOLO(weight)
    hits = {key: 0 for key in ("small", "medium", "large")}
    totals = {key: 0 for key in ("small", "medium", "large")}

    images = sorted(glob.glob(f"{dataset}/images/{split}/*.jpg"))
    for image_path in images:
        label_path = (
            f"{dataset}/labels/{split}/"
            f"{os.path.basename(image_path)[:-4]}.txt"
        )
        ground_truth = []
        if os.path.exists(label_path):
            with open(label_path, encoding="ascii") as stream:
                for line in stream:
                    parts = line.split()
                    if len(parts) == 5:
                        _, cx, cy, width, height = map(float, parts)
                        ground_truth.append((cx, cy, width, height))
        if not ground_truth:
            continue

        result = model.predict(
            image_path, conf=conf, imgsz=imgsz, verbose=False
        )[0]
        image_width = result.orig_shape[1]
        image_height = result.orig_shape[0]
        predictions = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else np.zeros((0, 4))
        )

        for cx, cy, width, height in ground_truth:
            area = width * height
            stratum = (
                "small"
                if area < p33
                else ("medium" if area < p67 else "large")
            )
            totals[stratum] += 1
            box = [
                (cx - width / 2) * image_width,
                (cy - height / 2) * image_height,
                (cx + width / 2) * image_width,
                (cy + height / 2) * image_height,
            ]
            overlaps = box_iou(box, predictions)
            if len(overlaps) and overlaps.max() >= iou_match:
                hits[stratum] += 1

    recall = {
        key: hits[key] / totals[key] if totals[key] else 0.0
        for key in totals
    }
    return recall, totals, len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--out", default="kvasir_eval.json")
    args = parser.parse_args()

    thresholds = np.load(
        args.thresholds, allow_pickle=True
    ).item()
    p33, p67 = thresholds["p33"], thresholds["p67"]
    baseline, baseline_totals, image_count = evaluate(
        args.baseline,
        args.dataset,
        p33,
        p67,
        args.split,
        args.imgsz,
        args.conf,
        args.iou_match,
    )
    candidate, candidate_totals, _ = evaluate(
        args.candidate,
        args.dataset,
        p33,
        p67,
        args.split,
        args.imgsz,
        args.conf,
        args.iou_match,
    )

    print(f"Kvasir thresholds: p33={p33 * 100:.3f}%, p67={p67 * 100:.3f}%")
    print(f"Images: {image_count}, GT counts: {baseline_totals}")
    print(f"{'model':12s} {'small':>9s} {'medium':>9s} {'large':>9s}")
    for name, values in (
        ("baseline", baseline),
        ("candidate", candidate),
    ):
        print(
            f"{name:12s} {values['small'] * 100:8.1f}% "
            f"{values['medium'] * 100:8.1f}% "
            f"{values['large'] * 100:8.1f}%"
        )
    delta = (candidate["small"] - baseline["small"]) * 100
    print(f"Small-stratum difference: {delta:+.1f} pp")

    output = {
        "thresholds": {
            "p33": float(p33),
            "p67": float(p67),
        },
        "settings": {
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou_match": args.iou_match,
        },
        "baseline": baseline,
        "candidate": candidate,
        "gt_counts": baseline_totals,
        "small_difference_pp": delta,
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
