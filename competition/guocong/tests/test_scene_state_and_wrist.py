from pathlib import Path

import cv2
import numpy as np

from src.common.types import ObjectObservation, SceneState
from src.perception.color_detector import ColorDetector
from src.perception.wrist_refiner import WristRefiner


CONFIG = Path(__file__).resolve().parents[1] / "config" / "perception.yaml"


def test_scene_state_best_can_require_3d():
    state = SceneState(timestamp=1.0)
    state.add(ObjectObservation("a", "box", "pink", "head", (0, 0, 1, 1), (0, 0), 0.9))
    state.add(ObjectObservation("b", "box", "pink", "head", (0, 0, 1, 1), (0, 0), 0.8,
                                position_world=(1.0, 2.0, 3.0)))
    assert state.best("pink_box").object_id == "a"
    assert state.best("pink_box", require_3d=True).object_id == "b"


def test_wrist_refiner_reports_image_center_error():
    image = np.zeros((200, 300, 3), np.uint8)
    cv2.rectangle(image, (180, 80), (240, 140), (180, 105, 255), -1)
    result = WristRefiner(ColorDetector.from_yaml(CONFIG), tolerance_px=10).refine(image, "pink")
    assert result is not None
    assert result.error_px[0] > 0
    assert result.aligned is False
