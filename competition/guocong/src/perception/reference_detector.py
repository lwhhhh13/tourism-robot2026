from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceDetection:
    object_class: str
    color: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[int, int]
    area_px: int
    confidence: float


class ReferenceDetector:
    """白色立方体/长方体障碍物的可调基线检测，仅用于场景校验。"""

    def __init__(self, config: Mapping):
        cfg = config.get("reference_detector", config)
        self.enabled = bool(cfg.get("enabled", True))
        self.input_order = str(cfg.get("input_color_order", "BGR")).upper()
        self.lower = np.asarray(cfg.get("lower_hsv", [0, 0, 175]), np.uint8)
        self.upper = np.asarray(cfg.get("upper_hsv", [179, 70, 255]), np.uint8)
        self.min_area_px = int(cfg.get("min_area_px", 2500))
        self.max_area_fraction = float(cfg.get("max_area_fraction", 0.35))
        self.cube_max_aspect = float(cfg.get("cube_max_aspect", 1.35))

    def detect(self, image: np.ndarray) -> List[ReferenceDetection]:
        if not self.enabled:
            return []
        conversion = cv2.COLOR_BGR2HSV if self.input_order == "BGR" else cv2.COLOR_RGB2HSV
        hsv = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), conversion)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        height, width = image.shape[:2]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        output: List[ReferenceDetection] = []
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            if area < self.min_area_px or area > self.max_area_fraction * height * width:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            object_class = "cube" if aspect <= self.cube_max_aspect else "shelf_obstacle"
            u, v = (int(round(value)) for value in centroids[label])
            fill = area / max(w * h, 1)
            confidence = float(np.clip(0.5 * fill + 0.5 * min(1.0, area / (4 * self.min_area_px)), 0, 1))
            output.append(ReferenceDetection(object_class, "white", (x, y, x + w - 1, y + h - 1),
                                              (u, v), area, confidence))
        return sorted(output, key=lambda item: item.confidence, reverse=True)

    @staticmethod
    def draw_bgr(image_bgr: np.ndarray, detections: Sequence[ReferenceDetection]) -> np.ndarray:
        output = image_bgr.copy()
        for item in detections:
            x1, y1, x2, y2 = item.bbox_xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(output, f"white_{item.object_class} {item.confidence:.2f}",
                        (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return output
