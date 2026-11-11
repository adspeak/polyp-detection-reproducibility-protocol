#!/usr/bin/env python3
"""Recompute COCO AP_S, AP_L, and AR_S for baseline and candidate weights."""

import argparse
import contextlib
import glob
import io
import json
import os

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def build_ground_truth(dataset, split, output_path):
    image_paths = sorted(glob.glob(f"{dataset}/images/{split}/*.jpg"))
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "polyp"}],
    }
    image_id_map = {}
    annotation_id = 0

    for image_id, image_path in enumerate(image_paths):
        width, height = Image.open(image_path).size
        basename = os.path.basename(image_path)
        image_id_map[basename] = image_id
        coco["images"].append(
            {
                "id": image_id,
                "file_name": basename,
                "width": width,
                "height": height,
            }
        )
        label_path = (
            f"{dataset}/labels/{split}/"
            f"{os.path.splitext(basename)[0]}.txt"
        )
        if not os.path.exists(label_path):
            continue
        with open(label_path, encoding="ascii") as stream:
            for line in stream:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, box_width, box_height = map(float, parts)
                x = (cx - box_width / 2) * width
                y = (cy - box_height / 2) * height
                width_px = box_width * width
                height_px = box_height * height
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x, y, width_px, height_px],
                        "area": width_px * height_px,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(coco, stream)
    return image_paths, image_id_map, coco


def evaluate_weight(
    weight,
    image_paths,
    image_id_map,
    gt_path,
    imgsz,
    conf,
    detections_path,
):
    model = YOLO(weight)
    detections = []
    for image_path in image_paths:
        image_id = image_id_map[os.path.basename(image_path)]
        result = model.predict(
            image_path, conf=conf, imgsz=imgsz, verbose=False
        )[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "score": float(score),
                }
            )
    if not detections:
        raise RuntimeError(f"No detections produced by {weight}")

    with open(detections_path, "w", encoding="utf-8") as stream:
        json.dump(detections, stream)
    ground_truth = COCO(gt_path)
    predictions = ground_truth.loadRes(detections_path)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(ground_truth, predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--out", default="coco_ap.json")
    args = parser.parse_args()

    gt_path = os.path.join(os.path.dirname(args.out), "coco_gt.json")
    image_paths, image_id_map, coco = build_ground_truth(
        args.dataset, args.split, gt_path
    )
    print(
        f"COCO GT: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} boxes"
    )

    stats = {}
    for name, weight in (
        ("baseline", args.baseline),
        ("candidate", args.candidate),
    ):
        detections_path = os.path.join(
            os.path.dirname(args.out), f"{name}_coco_dt.json"
        )
        stats[name] = evaluate_weight(
            weight,
            image_paths,
            image_id_map,
            gt_path,
            args.imgsz,
            args.conf,
            detections_path,
        )

    baseline = stats["baseline"]
    candidate = stats["candidate"]
    summary = {
        "stats_order": [
            "AP",
            "AP50",
            "AP75",
            "AP_small",
            "AP_medium",
            "AP_large",
            "AR_1",
            "AR_10",
            "AR_100",
            "AR_small",
            "AR_medium",
            "AR_large",
        ],
        "baseline": baseline,
        "candidate": candidate,
        "differences_pp": {
            "AP_small": (candidate[3] - baseline[3]) * 100,
            "AP_large": (candidate[5] - baseline[5]) * 100,
            "AR_small": (candidate[9] - baseline[9]) * 100,
        },
        "settings": {
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
        },
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    def metric(value):
        return "N/A" if value < 0 else f"{value * 100:.1f}%"

    print(
        f"{'model':12s} {'AP':>7s} {'AP50':>7s} {'AP_s':>7s} "
        f"{'AP_m':>7s} {'AP_l':>7s} {'AR_s':>7s}"
    )
    for name, values in stats.items():
        print(
            f"{name:12s} {metric(values[0]):>7s} "
            f"{metric(values[1]):>7s} {metric(values[3]):>7s} "
            f"{metric(values[4]):>7s} {metric(values[5]):>7s} "
            f"{metric(values[9]):>7s}"
        )
    print(
        "Candidate minus baseline: "
        f"AP_s={summary['differences_pp']['AP_small']:+.1f} pp, "
        f"AP_l={summary['differences_pp']['AP_large']:+.1f} pp, "
        f"AR_s={summary['differences_pp']['AR_small']:+.1f} pp"
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
