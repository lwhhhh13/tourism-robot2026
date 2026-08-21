#!/usr/bin/env python3
"""用 presence CSV 统计每种颜色的检出率、精确率和误检率。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.color_detector import ColorDetector


COLORS = ("pink", "yellow", "brown")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV columns: image,pink,yellow,brown")
    parser.add_argument("labels", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "perception.yaml")
    parser.add_argument("--failure-dir", type=Path, default=ROOT / "failure_cases" / "perception")
    parser.add_argument("--report", type=Path, default=ROOT / "perception_metrics.json")
    args = parser.parse_args()
    detector = ColorDetector.from_yaml(args.config)
    counts = {color: {key: 0 for key in ("tp", "fp", "fn", "tn")} for color in COLORS}
    failures = []
    with args.labels.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row_index, row in enumerate(rows):
        image_path = (args.labels.parent / row["image"]).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        predicted = {item.color for item in detector.detect(image)}
        failed = False
        for color in COLORS:
            expected = parse_bool(row.get(color, "0"))
            observed = color in predicted
            key = "tp" if expected and observed else "fn" if expected else "fp" if observed else "tn"
            counts[color][key] += 1
            failed |= key in {"fp", "fn"}
        if failed:
            args.failure_dir.mkdir(parents=True, exist_ok=True)
            destination = args.failure_dir / f"{row_index:04d}_{image_path.name}"
            shutil.copy2(image_path, destination)
            failures.append({"image": str(image_path), "predicted": sorted(predicted)})
    metrics = {}
    for color, item in counts.items():
        tp, fp, fn, tn = (item[key] for key in ("tp", "fp", "fn", "tn"))
        metrics[color] = {
            **item,
            "recall_success_rate": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
        }
    report = {"images": len(rows), "metrics": metrics, "failures": failures}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
