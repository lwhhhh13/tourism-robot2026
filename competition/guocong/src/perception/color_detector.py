from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ColorDetection:
    color: str
    object_class: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[int, int]
    area_px: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "object_class": self.object_class,
            "bbox_xyxy": list(self.bbox_xyxy),
            "center_uv": list(self.center_uv),
            "area_px": self.area_px,
            "confidence": self.confidence,
        }


class ColorDetector:
    """无需联网和模型权重的 pink/yellow/brown HSV 连通域检测器。"""

    def __init__(self, config: Mapping):
        cfg = config.get("detector", config)
        self.input_order = str(cfg.get("input_color_order", "BGR")).upper()
        if self.input_order not in {"BGR", "RGB"}:
            raise ValueError("input_color_order must be BGR or RGB")
        self.hsv_ranges = cfg["hsv_ranges"]
        self.min_area_px = int(cfg.get("min_area_px", 500))
        self.max_area_fraction = float(cfg.get("max_area_fraction", 0.60))
        self.min_fill_ratio = float(cfg.get("min_fill_ratio", 0.30))
        morphology = cfg.get("morphology", {})
        self.open_kernel = int(morphology.get("open_kernel", 3))
        self.close_kernel = int(morphology.get("close_kernel", 7))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ColorDetector":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(yaml.safe_load(stream))

    @staticmethod
    def _ranges(spec: Mapping) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for item in spec.get("ranges", [spec]):
            yield np.asarray(item["lower"], np.uint8), np.asarray(item["upper"], np.uint8)

    def mask_for_color(self, image: np.ndarray, color: str) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}")
        if color not in self.hsv_ranges:
            raise KeyError(f"Unknown color {color!r}; configured: {sorted(self.hsv_ranges)}")
        conversion = cv2.COLOR_BGR2HSV if self.input_order == "BGR" else cv2.COLOR_RGB2HSV
        hsv = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), conversion)
        output = np.zeros(image.shape[:2], np.uint8)
        for lower, upper in self._ranges(self.hsv_ranges[color]):
            output = cv2.bitwise_or(output, cv2.inRange(hsv, lower, upper))
        if self.open_kernel > 1:
            kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
        return output

    def detect(self, image: np.ndarray, colors: Sequence[str] | None = None) -> List[ColorDetection]:
        height, width = image.shape[:2]
        detections: List[ColorDetection] = []
        for color in colors or tuple(self.hsv_ranges):
            mask = self.mask_for_color(image, color)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, count):
                x, y, w, h, area = (int(value) for value in stats[label])
                if area < self.min_area_px or area > self.max_area_fraction * height * width:
                    continue
                fill = area / max(w * h, 1)
                if fill < self.min_fill_ratio:
                    continue
                u, v = (int(round(value)) for value in centroids[label])
                size_score = min(1.0, area / max(4.0 * self.min_area_px, 1.0))
                confidence = float(np.clip(0.65 * fill + 0.35 * size_score, 0.0, 1.0))
                detections.append(
                    ColorDetection(color, "box", (x, y, x + w - 1, y + h - 1), (u, v), area, confidence)
                )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    @staticmethod
    def draw_bgr(image_bgr: np.ndarray, detections: Sequence[ColorDetection]) -> np.ndarray:
        output = image_bgr.copy()
        colors = {"pink": (180, 105, 255), "yellow": (0, 220, 240), "brown": (40, 70, 120)}
        for item in detections:
            x1, y1, x2, y2 = item.bbox_xyxy
            color = colors.get(item.color, (0, 255, 0))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{item.color} {item.confidence:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return output
