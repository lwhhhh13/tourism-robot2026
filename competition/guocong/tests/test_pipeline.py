from pathlib import Path

import cv2
import numpy as np
import yaml

from src.perception.color_detector import ColorDetector
from src.perception.perception_pipeline import PerceptionPipeline
from src.perception.reference_detector import ReferenceDetector
from src.perception.rgbd_localizer import CameraIntrinsics, RGBDLocalizer


CONFIG = Path(__file__).resolve().parents[1] / "config" / "perception.yaml"


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_pipeline_emits_box_scene_state_with_world_xyz():
    image = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(image, (60, 60), (140, 140), (0, 220, 240), -1)
    depth = np.full((200, 200), 1500, np.uint16)
    config = load_config()
    config["reference_detector"]["enabled"] = False
    pipeline = PerceptionPipeline(ColorDetector(config), RGBDLocalizer(), ReferenceDetector(config))
    state = pipeline.process(image, depth, CameraIntrinsics(100, 100, 100, 100, 200, 200), np.eye(4),
                             timestamp=123.0)
    target = state.best("yellow_box", require_3d=True)
    assert target is not None
    assert target.depth_m == 1.5
    np.testing.assert_allclose(target.position_world, [0.0, 0.0, 1.5], atol=0.02)


def test_reference_detector_labels_white_shapes():
    image = np.zeros((240, 400, 3), np.uint8)
    cv2.rectangle(image, (20, 60), (100, 140), (255, 255, 255), -1)
    cv2.rectangle(image, (180, 70), (360, 130), (255, 255, 255), -1)
    classes = {item.object_class for item in ReferenceDetector(load_config()).detect(image)}
    assert classes == {"cube", "shelf_obstacle"}
