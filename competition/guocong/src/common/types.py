from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ObjectObservation:
    """B 输出给控制/规划模块的单个观测，三维位置单位统一为米。"""

    object_id: str
    object_class: str
    color: Optional[str]
    camera: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[int, int]
    confidence: float
    depth_m: Optional[float] = None
    position_camera: Optional[Tuple[float, float, float]] = None
    position_world: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneState:
    """稳定的 B -> A/C/D 接口；正式比赛三维坐标系为 world。"""

    timestamp: float
    frame_id: str = "world"
    objects: Dict[str, List[ObjectObservation]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add(self, observation: ObjectObservation) -> None:
        key = f"{observation.color}_{observation.object_class}" if observation.color else observation.object_class
        self.objects.setdefault(key, []).append(observation)

    def best(self, key: str, require_3d: bool = False) -> Optional[ObjectObservation]:
        candidates = self.objects.get(key, [])
        if require_3d:
            candidates = [item for item in candidates if item.position_world is not None]
        return max(candidates, key=lambda item: item.confidence, default=None)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "objects": {key: [item.to_dict() for item in values] for key, values in self.objects.items()},
            "warnings": list(self.warnings),
        }
