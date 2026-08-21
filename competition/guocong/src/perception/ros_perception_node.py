#!/usr/bin/env python3
"""正式客户端视觉节点：头部 RGB-D -> /material/detections。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from src.perception.color_detector import ColorDetector
from src.perception.perception_pipeline import PerceptionPipeline
from src.perception.reference_detector import ReferenceDetector
from src.perception.rgbd_localizer import CameraIntrinsics, RGBDLocalizer, make_transform


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3)
    scale = 2.0 / norm
    return np.array([
        [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
        [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
        [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
    ], dtype=float)


class PerceptionNode(Node):
    def __init__(self, config_path: Path):
        super().__init__("guocong_perception")
        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        topics = self.config["topics"]
        depth_cfg = self.config["depth"]
        self.bridge = CvBridge()
        self.detector = ColorDetector(self.config)
        self.reference_detector = ReferenceDetector(self.config)
        self.pipeline = PerceptionPipeline(
            self.detector,
            RGBDLocalizer(depth_cfg["min_m"], depth_cfg["max_m"],
                          depth_cfg["patch_radius_px"], depth_cfg["scale"]),
            self.reference_detector,
        )
        self.max_delta = float(self.config["synchronization"]["max_rgb_depth_delta_s"])
        self.world_frame = str(self.config.get("world_frame", "world"))
        self.default_camera_frame = str(self.config.get("head_camera_optical_frame", "head_camera_color_optical_frame"))
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_stamp: Optional[float] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detection_pub = self.create_publisher(Detection3DArray, topics["detections"], 10)
        self.scene_state_pub = self.create_publisher(String, topics["scene_state"], 10)
        self.result_pub = self.create_publisher(Image, topics["result_image"], 2)
        self.create_subscription(CameraInfo, topics["head_camera_info"], self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, topics["head_depth"], self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, topics["head_rgb"], self.on_rgb, qos_profile_sensor_data)
        self.get_logger().info(f"loaded perception config: {config_path}")

    def on_camera_info(self, message: CameraInfo) -> None:
        self.intrinsics = CameraIntrinsics.from_camera_info(message.k, message.width, message.height)

    def on_depth(self, message: Image) -> None:
        self.latest_depth = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
        self.latest_depth_stamp = _stamp_seconds(message.header.stamp)

    def _world_from_camera(self, frame_id: str, stamp) -> Optional[np.ndarray]:
        source_frame = frame_id or self.default_camera_frame
        try:
            stamped = self.tf_buffer.lookup_transform(
                self.world_frame, source_frame, Time.from_msg(stamp), timeout=Duration(seconds=0.05)
            )
        except TransformException as exc:
            self.get_logger().warning(f"TF {self.world_frame} <- {source_frame} unavailable: {exc}")
            return None
        transform = stamped.transform
        q = transform.rotation
        t = transform.translation
        return make_transform(_quaternion_matrix(q.x, q.y, q.z, q.w), (t.x, t.y, t.z))

    def on_rgb(self, message: Image) -> None:
        rgb_stamp = _stamp_seconds(message.header.stamp)
        if self.latest_depth is None or self.latest_depth_stamp is None or self.intrinsics is None:
            return
        if abs(rgb_stamp - self.latest_depth_stamp) > self.max_delta:
            self.get_logger().warning("drop unsynchronized RGB/depth pair")
            return
        world_from_camera = self._world_from_camera(message.header.frame_id, message.header.stamp)
        if world_from_camera is None:
            return
        image_bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        state = self.pipeline.process(image_bgr, self.latest_depth, self.intrinsics, world_from_camera,
                                      timestamp=rgb_stamp, camera="head_camera")
        scene_message = String()
        scene_message.data = json.dumps(state.to_dict(), ensure_ascii=False)
        self.scene_state_pub.publish(scene_message)
        output = Detection3DArray()
        output.header = message.header
        output.header.frame_id = self.world_frame
        for observations in state.objects.values():
            for observation in observations:
                if observation.position_world is None:
                    continue
                detection = Detection3D()
                detection.header = output.header
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = (
                    observation.color if observation.object_class == "box" else f"{observation.color}_{observation.object_class}"
                )
                hypothesis.hypothesis.score = float(observation.confidence)
                px, py, pz = observation.position_world
                hypothesis.pose.pose.position.x = px
                hypothesis.pose.pose.position.y = py
                hypothesis.pose.pose.position.z = pz
                detection.results.append(hypothesis)
                detection.bbox.center.position.x = px
                detection.bbox.center.position.y = py
                detection.bbox.center.position.z = pz
                if observation.object_class == "box":
                    detection.bbox.size.x = 0.24
                    detection.bbox.size.y = 0.16
                    detection.bbox.size.z = 0.19
                output.detections.append(detection)
        self.detection_pub.publish(output)
        drawn = ColorDetector.draw_bgr(image_bgr, self.detector.detect(image_bgr))
        drawn = self.reference_detector.draw_bgr(drawn, self.reference_detector.detect(image_bgr))
        result = self.bridge.cv2_to_imgmsg(drawn, encoding="bgr8")
        result.header = message.header
        self.result_pub.publish(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config" / "perception.yaml")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = PerceptionNode(args.config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
