#!/usr/bin/env python3
"""Evaluate seed-group ensembles with weighted box fusion."""

import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(per_model_boxes, per_model_scores, iou_thr=0.55):
    order = sorted(
        range(len(per_model_boxes)),
        key=lambda index: per_model_scores[index],
        reverse=True,
    )
    clusters = []
    for index in order:
        box = per_model_boxes[index]
        score = per_model_scores[index]
        best_cluster = None
        best_iou = iou_thr
        for cluster in clusters:
            overlap = iou(cluster["box"], box)
            if overlap >= best_iou:
                best_iou = overlap
                best_cluster = cluster
        if best_cluster is None:
            clusters.append(
                {"box": list(box), "sum": score, "count": 1}
            )
        else:
            weight = score
            total = best_cluster["sum"] + weight
            best_cluster["box"] = [
                (best_cluster["box"][axis] * best_cluster["sum"] + box[axis] * weight)
                / total
                for axis in range(4)
            ]
            best_cluster["sum"] = total
            best_cluster["count"] += 1

    boxes = np.asarray([cluster["box"] for cluster in clusters])
    scores = np.asarray(
        [cluster["sum"] / cluster["count"] for cluster in clusters]
    )
    return boxes, scores


def predict_all(weights, image_path, imgsz, conf):
    boxes = []
    scores = []
    for weight in weights:
        result = YOLO(weight).predict(
            image_path, conf=conf, imgsz=imgsz, verbose=False
        )[0]
        if result.boxes is None:
            continue
        boxes.extend(result.boxes.xyxy.cpu().numpy().tolist())
        scores.extend(result.boxes.conf.cpu().numpy().tolist())
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,))
    return weighted_box_fusion(boxes, scores)


def recall_iou(box, boxes):
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


def evaluate_group(
    weights, dataset, split, p33, p67, imgsz, conf, iou_match
):
    hits = {key: 0 for key in ("small", "medium", "large")}
    totals = {key: 0 for key in ("small", "medium", "large")}
    for image_path in sorted(glob.glob(f"{dataset}/images/{split}/*.jpg")):
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

        boxes, scores = predict_all(
            weights, image_path, imgsz, conf
        )
        result = YOLO(weights[0]).predict(
            image_path, conf=conf, imgsz=imgsz, verbose=False
        )[0]
        image_width = result.orig_shape[1]
        image_height = result.orig_shape[0]

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
            overlaps = recall_iou(box, boxes)
            if len(overlaps) and overlaps.max() >= iou_match:
                hits[stratum] += 1

    return {
        key: hits[key] / totals[key] if totals[key] else 0.0
        for key in totals
    }, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--out", default="ensemble_eval.json")
    args = parser.parse_args()

    thresholds = np.load(
        args.thresholds, allow_pickle=True
    ).item()
    p33, p67 = thresholds["p33"], thresholds["p67"]

    baseline_flat = list(args.baseline)
    candidate_flat = list(args.candidate)
    groups = [tuple(int(value) for value in group.split(",")) for group in args.groups]
    assert len(baseline_flat) == len(candidate_flat), (
        "Baseline and candidate lists must have the same length."
    )
    baseline_by_seed = dict(enumerate(baseline_flat, start=42))
    candidate_by_seed = dict(enumerate(candidate_flat, start=42))

    results = {"groups": []}
    group_deltas = {"small": [], "medium": [], "large": []}
    for group in groups:
        baseline_weights = [baseline_by_seed[seed] for seed in group]
        candidate_weights = [candidate_by_seed[seed] for seed in group]
        baseline_recall, totals = evaluate_group(
            baseline_weights,
            args.dataset,
            args.split,
            p33,
            p67,
            args.imgsz,
            args.conf,
            args.iou_match,
        )
        candidate_recall, _ = evaluate_group(
            candidate_weights,
            args.dataset,
            args.split,
            p33,
            p67,
            args.imgsz,
            args.conf,
            args.iou_match,
        )
        entry = {
            "seeds": group,
            "baseline": baseline_recall,
            "candidate": candidate_recall,
            "gt_counts": totals,
            "difference_pp": {
                key: (candidate_recall[key] - baseline_recall[key]) * 100
                for key in baseline_recall
            },
        }
        results["groups"].append(entry)
        for key in group_deltas:
            group_deltas[key].append(entry["difference_pp"][key])

    summary = {}
    for key, values in group_deltas.items():
        values = np.asarray(values)
        summary[key] = {
            "mean_pp": float(values.mean()),
            "sd_pp": float(values.std(ddof=1)),
        }
    results["summary"] = summary

    print("Ensemble group results (candidate - baseline)")
    for entry in results["groups"]:
        print(
            f"  seeds {tuple(entry['seeds'])}: "
            + ", ".join(
                f"{key}={entry['difference_pp'][key]:+.1f} pp"
                for key in ("small", "medium", "large")
            )
        )
    for key, values in summary.items():
        print(
            f"  {key}: {values['mean_pp']:+.2f} +/- {values['sd_pp']:.2f} pp"
        )

    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
