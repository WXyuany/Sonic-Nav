#!/usr/bin/env -S /usr/bin/python3
"""Measure the RGB-D/TF reconstruction error against a privileged simulation anchor."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.world_model import anchor_to_world, project_detection_with_depth, transform_point_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RGB-D projection/TF against a known ball or box anchor.")
    parser.add_argument("--reference-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--object-id", default="")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/depth/camera_info")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--pixel-radius", type=float, default=4.0)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=20, help="Stop writing after this many samples; 0 is unlimited.")
    parser.add_argument("--output", default="reports/perception/rgbd_calibration.jsonl")
    return parser.parse_args()


class RgbdCalibrationProbe(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_rgbd_calibration_probe")
        self.args = args
        self.depth: np.ndarray | None = None
        self.depth_frame = "camera_depth_optical_frame"
        self.camera_k: list[float] | None = None
        self.sample_count = 0
        self.output = _repo_path(args.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, 2)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 2)
        self.create_subscription(String, args.reference_topic, self._anchor_cb, 10)
        self.get_logger().info(f"RGB-D calibration: reference={args.reference_topic} output={self.output}")

    def _depth_cb(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            return
        self.depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(int(msg.height), int(msg.width)).copy()
        self.depth_frame = msg.header.frame_id or self.depth_frame

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_k = [float(value) for value in msg.k]

    def _anchor_cb(self, msg: String) -> None:
        if int(self.args.max_samples) > 0 and self.sample_count >= int(self.args.max_samples):
            return
        if self.depth is None or self.camera_k is None:
            return
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("reference anchor must be an object")
            world = anchor_to_world(payload)
            reference = _reference_object(world, self.args.object_id)
            pixel = reference.properties.get("pixel") if isinstance(reference.properties, dict) else None
            if not isinstance(pixel, dict) or not _finite(pixel.get("u")) or not _finite(pixel.get("v")):
                return
            radius = max(1.0, float(self.args.pixel_radius))
            u, v = float(pixel["u"]), float(pixel["v"])
            projected = project_detection_with_depth(
                {"object_id": reference.object_id, "category": reference.category, "bbox": [u - radius, v - radius, u + radius, v + radius]},
                self.depth,
                self.camera_k,
                frame_id=self.depth_frame,
                patch_radius=int(self.args.patch_radius),
            )
            projected_base = self._transform(projected["pose_camera"], self.args.base_frame)
            projected_map = self._transform(projected["pose_camera"], self.args.map_frame)
            record = {
                "schema": "sonic_rgbd_calibration_sample_v0",
                "sample_id": f"calibration_{self.sample_count + 1:06d}",
                "recorded_at": time.time(),
                "object_id": reference.object_id,
                "category": reference.category,
                "pixel": {"u": u, "v": v},
                "reference_pose_base": reference.pose_base.to_dict() if reference.pose_base else None,
                "reference_pose_map": reference.pose_map.to_dict() if reference.pose_map else None,
                "projected_pose_base": projected_base,
                "projected_pose_map": projected_map,
                "base_error_m": _pose_error(reference.pose_base.to_dict() if reference.pose_base else None, projected_base),
                "map_error_m": _pose_error(reference.pose_map.to_dict() if reference.pose_map else None, projected_map),
                "uncertainty": projected.get("uncertainty"),
            }
            _append_jsonl(self.output, record)
            self.sample_count += 1
            self.get_logger().info(
                f"calibration sample {self.sample_count}: base_error_m={record['base_error_m']} map_error_m={record['map_error_m']}"
            )
        except Exception as exc:
            self.get_logger().warn(f"calibration sample rejected: {exc}")

    def _transform(self, pose: dict[str, Any], target_frame: str) -> dict[str, Any]:
        transform = self.tf_buffer.lookup_transform(target_frame, str(pose.get("frame_id")), rclpy.time.Time())
        t, q = transform.transform.translation, transform.transform.rotation
        return transform_point_pose(
            pose,
            {"translation": [t.x, t.y, t.z], "rotation_xyzw": [q.x, q.y, q.z, q.w]},
            frame_id=target_frame,
        )


def _reference_object(world: Any, object_id: str) -> Any:
    if object_id:
        object_ = world.get_object(object_id)
        if object_ is None:
            raise ValueError(f"reference object {object_id!r} was not found")
        return object_
    object_ = world.primary_object()
    if object_ is None:
        raise ValueError("reference anchor has no primary object")
    return object_


def _pose_error(reference: dict[str, Any] | None, estimate: dict[str, Any] | None) -> float | None:
    if not isinstance(reference, dict) or not isinstance(estimate, dict):
        return None
    left, right = reference.get("position"), estimate.get("position")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) < 3 or len(right) < 3:
        return None
    try:
        return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RgbdCalibrationProbe(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
