from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .color_detector import ColorDetector


@dataclass(frozen=True)
class WristRefinement:
    color: str
    center_uv: tuple[int, int]
    error_px: tuple[float, float]
    error_normalized: tuple[float, float]
    confidence: float
    aligned: bool


class WristRefiner:
    """手眼 RGB 二维精定位；腕部无深度，因此只输出目标相对图像中心误差。"""

    def __init__(self, detector: ColorDetector, tolerance_px: float = 18.0):
        self.detector = detector
        self.tolerance_px = float(tolerance_px)

    def refine(self, rgb: np.ndarray, target_color: str) -> Optional[WristRefinement]:
        detections = self.detector.detect(rgb, colors=[target_color])
        if not detections:
            return None
        target = detections[0]
        height, width = rgb.shape[:2]
        du = float(target.center_uv[0] - (width - 1) / 2.0)
        dv = float(target.center_uv[1] - (height - 1) / 2.0)
        return WristRefinement(
            color=target.color, center_uv=target.center_uv, error_px=(du, dv),
            error_normalized=(du / max(width / 2.0, 1.0), dv / max(height / 2.0, 1.0)),
            confidence=target.confidence,
            aligned=abs(du) <= self.tolerance_px and abs(dv) <= self.tolerance_px,
        )
