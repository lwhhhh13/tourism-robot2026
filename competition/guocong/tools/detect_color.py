#!/usr/bin/env python3
"""在一张保存的 BGR 图片上运行 HSV 检测并输出标注图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.color_detector import ColorDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "perception.yaml")
    parser.add_argument("--output", type=Path, default=Path("detections.png"))
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    detector = ColorDetector.from_yaml(args.config)
    detections = detector.detect(bgr)
    if not cv2.imwrite(str(args.output), detector.draw_bgr(bgr, detections)):
        raise OSError(args.output)
    print(json.dumps([item.to_dict() for item in detections], ensure_ascii=False, indent=2))
    print(f"Annotated image: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
