#!/usr/bin/env python3
"""保存并报告正式客户端可见的三路 RGB 与头部对齐深度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


TOPICS = {
    "head_rgb": "/head_camera/color/image_raw",
    "head_depth": "/head_camera/aligned_depth_to_color/image_raw",
    "left_rgb": "/left_camera/color/image_raw",
    "right_rgb": "/right_camera/color/image_raw",
}


class ObservationInspector(Node):
    def __init__(self, output_dir: Path, timeout_s: float):
        super().__init__("inspect_observation")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.frames: Dict[str, np.ndarray] = {}
        self.report: Dict[str, dict] = {}
        self.finished = False
        self.subscriptions = [
            self.create_subscription(Image, topic, lambda message, name=key: self.on_image(name, message),
                                     qos_profile_sensor_data)
            for key, topic in TOPICS.items()
        ]
        self.timer = self.create_timer(timeout_s, self.finish)

    def on_image(self, key: str, message: Image) -> None:
        if key in self.frames:
            return
        encoding = "passthrough" if key == "head_depth" else "bgr8"
        frame = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding=encoding))
        finite = frame[np.isfinite(frame)]
        self.frames[key] = frame.copy()
        self.report[key] = {
            "topic": TOPICS[key], "message_encoding": message.encoding,
            "shape": list(frame.shape), "dtype": str(frame.dtype),
            "min": None if finite.size == 0 else float(finite.min()),
            "max": None if finite.size == 0 else float(finite.max()),
        }
        if key == "head_depth":
            valid = frame[np.isfinite(frame) & (frame > 0)]
            self.report[key].update({
                "valid_min": None if valid.size == 0 else float(valid.min()),
                "valid_max": None if valid.size == 0 else float(valid.max()),
                "valid_min_m": None if valid.size == 0 else float(valid.min()) * 0.001,
                "valid_max_m": None if valid.size == 0 else float(valid.max()) * 0.001,
            })
        print(f"{key}: shape={frame.shape} dtype={frame.dtype} "
              f"min={self.report[key]['min']} max={self.report[key]['max']}")
        if len(self.frames) == len(TOPICS):
            self.finish()

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        print(f"obs.keys() = {sorted(self.frames)}")
        for key, frame in self.frames.items():
            if key == "head_depth":
                np.save(self.output_dir / "frame_depth.npy", frame)
                valid = frame[np.isfinite(frame) & (frame > 0)]
                if valid.size:
                    low, high = np.percentile(valid, [2, 98])
                    preview = np.clip((frame.astype(float) - low) / max(high - low, 1.0) * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(self.output_dir / "frame_depth_preview.png"), preview)
            else:
                filename = "frame_rgb.png" if key == "head_rgb" else f"{key}.png"
                cv2.imwrite(str(self.output_dir / filename), frame)
        self.report["missing"] = sorted(set(TOPICS) - set(self.frames))
        (self.output_dir / "observation_report.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"saved to {self.output_dir}; missing={self.report['missing']}")
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("recordings/observation_inspection"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = ObservationInspector(args.output, args.timeout)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
