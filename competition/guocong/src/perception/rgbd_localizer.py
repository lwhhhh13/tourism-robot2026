from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, k: Sequence[float], width: int, height: int) -> "CameraIntrinsics":
        matrix = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]),
                   float(matrix[1, 2]), int(width), int(height))


class RGBDLocalizer:
    """对齐深度反投影；正式 ROS 深度为 mono16 毫米，光学系为 x右/y下/z前。"""

    def __init__(self, min_depth_m: float = 0.10, max_depth_m: float = 5.0,
                 patch_radius_px: int = 5, depth_scale: float = 0.001):
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.patch_radius_px = int(patch_radius_px)
        self.depth_scale = float(depth_scale)

    def robust_depth_m(self, depth: np.ndarray, center_uv: Sequence[int],
                       bbox_xyxy: Sequence[int] | None = None) -> Optional[float]:
        image = np.asarray(depth).squeeze()
        if image.ndim != 2:
            raise ValueError(f"Expected HxW depth, got {np.asarray(depth).shape}")
        u, v = (int(value) for value in center_uv)
        r = self.patch_radius_px
        x1, y1, x2, y2 = u - r, v - r, u + r, v + r
        if bbox_xyxy is not None:
            bx1, by1, bx2, by2 = (int(value) for value in bbox_xyxy)
            x1, y1, x2, y2 = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
        if x1 > x2 or y1 > y2:
            return None
        values_m = image[y1:y2 + 1, x1:x2 + 1].astype(np.float64) * self.depth_scale
        valid = values_m[np.isfinite(values_m) & (values_m >= self.min_depth_m) & (values_m <= self.max_depth_m)]
        return None if valid.size == 0 else float(np.median(valid))

    @staticmethod
    def pixel_to_camera(center_uv: Sequence[float], depth_m: float,
                        intrinsics: CameraIntrinsics) -> np.ndarray:
        u, v = (float(value) for value in center_uv)
        return np.array([(u - intrinsics.cx) * depth_m / intrinsics.fx,
                         (v - intrinsics.cy) * depth_m / intrinsics.fy,
                         depth_m], dtype=np.float64)

    @staticmethod
    def transform_point(point_xyz: Sequence[float], target_from_source: np.ndarray) -> np.ndarray:
        return (np.asarray(target_from_source, float) @ np.append(np.asarray(point_xyz, float), 1.0))[:3]

    def localize_world(self, center_uv: Sequence[int], bbox_xyxy: Sequence[int],
                       depth: np.ndarray, intrinsics: CameraIntrinsics,
                       world_from_camera_optical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        depth_m = self.robust_depth_m(depth, center_uv, bbox_xyxy)
        if depth_m is None:
            raise ValueError(f"No valid depth near pixel {tuple(center_uv)}")
        point_camera = self.pixel_to_camera(center_uv, depth_m, intrinsics)
        point_world = self.transform_point(point_camera, world_from_camera_optical)
        return point_camera, point_world, depth_m


def make_transform(rotation_3x3: np.ndarray, translation_xyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_3x3, float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_xyz, float)
    return transform
