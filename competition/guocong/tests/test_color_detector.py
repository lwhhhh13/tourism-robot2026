from pathlib import Path

import cv2
import numpy as np

from src.perception.color_detector import ColorDetector


CONFIG = Path(__file__).resolve().parents[1] / "config" / "perception.yaml"


def test_detects_all_three_colors_in_bgr():
    image = np.zeros((240, 360, 3), np.uint8)
    cv2.rectangle(image, (20, 30), (100, 150), (180, 105, 255), -1)
    cv2.rectangle(image, (130, 30), (210, 150), (0, 220, 240), -1)
    cv2.rectangle(image, (240, 30), (320, 150), (40, 70, 120), -1)
    detections = ColorDetector.from_yaml(CONFIG).detect(image)
    assert {item.color for item in detections} == {"pink", "yellow", "brown"}


def test_rejects_tiny_regions():
    image = np.zeros((100, 100, 3), np.uint8)
    image[2:5, 2:5] = (0, 220, 240)
    assert ColorDetector.from_yaml(CONFIG).detect(image) == []
