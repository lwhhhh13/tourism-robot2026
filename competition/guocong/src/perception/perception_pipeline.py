from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np

from src.common.types import ObjectObservation, SceneState

from .color_detector import ColorDetection, ColorDetector
from .reference_detector import ReferenceDetection, ReferenceDetector
from .rgbd_localizer import CameraIntrinsics, RGBDLocalizer


class PerceptionPipeline:
    """目标/参考物检测 + RGB-D 定位，输出统一 SceneState。"""

    def __init__(self, detector: ColorDetector, localizer: RGBDLocalizer,
                 reference_detector: Optional[ReferenceDetector] = None):
        self.detector = detector
        self.localizer = localizer
        self.reference_detector = reference_detector

    def _observations(self, image: np.ndarray) -> Iterable[ColorDetection | ReferenceDetection]:
        yield from self.detector.detect(image)
        if self.reference_detector is not None:
            yield from self.reference_detector.detect(image)

    def process(self, rgb: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics,
                world_from_camera_optical: np.ndarray, *, timestamp: Optional[float] = None,
                camera: str = "head_camera") -> SceneState:
        state = SceneState(timestamp=time.time() if timestamp is None else float(timestamp))
        counters: dict[str, int] = {}
        for detection in self._observations(rgb):
            key = f"{detection.color}_{detection.object_class}"
            index = counters.get(key, 0)
            counters[key] = index + 1
            depth_m = None
            point_camera = None
            point_world = None
            try:
                point_camera, point_world, depth_m = self.localizer.localize_world(
                    detection.center_uv, detection.bbox_xyxy, depth, intrinsics, world_from_camera_optical
                )
            except ValueError as exc:
                state.warnings.append(f"{key}[{index}]: {exc}")
            state.add(ObjectObservation(
                object_id=f"{key}_{index}", object_class=detection.object_class, color=detection.color,
                camera=camera, bbox_xyxy=detection.bbox_xyxy, center_uv=detection.center_uv,
                confidence=detection.confidence, depth_m=depth_m,
                position_camera=None if point_camera is None else tuple(float(v) for v in point_camera),
                position_world=None if point_world is None else tuple(float(v) for v in point_world),
            ))
        return state
