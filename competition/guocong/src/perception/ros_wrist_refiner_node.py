#!/usr/bin/env python3
"""左右手眼 RGB 精定位节点，发布目标相对图像中心的二维误差。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from src.perception.color_detector import ColorDetector
from src.perception.wrist_refiner import WristRefiner


def _find_target_color(payload) -> Optional[str]:
    if isinstance(payload, dict):
        color = payload.get("target_color")
        if color in {"pink", "yellow", "brown"}:
            return color
        for value in payload.values():
            found = _find_target_color(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_target_color(value)
            if found:
                return found
    return None


class WristRefinerNode(Node):
    def __init__(self, config_path: Path, target_color: Optional[str]):
        super().__init__("guocong_wrist_refiner")
        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        topics = self.config["topics"]
        tolerance = self.config["wrist_refinement"]["tolerance_px"]
        self.bridge = CvBridge()
        self.detector = ColorDetector(self.config)
        self.refiner = WristRefiner(self.detector, tolerance_px=tolerance)
        self.target_color = target_color
        self.output = self.create_publisher(String, topics["wrist_alignment"], 10)
        self.left_result = self.create_publisher(Image, topics["left_result_image"], 2)
        self.right_result = self.create_publisher(Image, topics["right_result_image"], 2)
        self.create_subscription(String, topics["instruction"], self.on_instruction, 10)
        self.create_subscription(Image, topics["left_rgb"], lambda msg: self.on_image("left", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, topics["right_rgb"], lambda msg: self.on_image("right", msg),
                                 qos_profile_sensor_data)

    def on_instruction(self, message: String) -> None:
        try:
            found = _find_target_color(json.loads(message.data))
        except json.JSONDecodeError:
            found = None
        if found:
            self.target_color = found
            self.get_logger().info(f"target color: {found}")

    def on_image(self, camera: str, message: Image) -> None:
        if self.target_color is None:
            return
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = self.refiner.refine(image, self.target_color)
        payload = {
            "camera": camera, "target_color": self.target_color,
            "timestamp": float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9,
            "visible": result is not None,
        }
        if result is not None:
            payload.update({
                "center_uv": list(result.center_uv), "error_px": list(result.error_px),
                "error_normalized": list(result.error_normalized), "confidence": result.confidence,
                "aligned": result.aligned,
            })
        output = String()
        output.data = json.dumps(payload, ensure_ascii=False)
        self.output.publish(output)
        detections = self.detector.detect(image, colors=[self.target_color])
        drawn = ColorDetector.draw_bgr(image, detections)
        result_image = self.bridge.cv2_to_imgmsg(drawn, encoding="bgr8")
        result_image.header = message.header
        (self.left_result if camera == "left" else self.right_result).publish(result_image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config" / "perception.yaml")
    parser.add_argument("--target-color", choices=["pink", "yellow", "brown"])
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = WristRefinerNode(args.config, args.target_color)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
