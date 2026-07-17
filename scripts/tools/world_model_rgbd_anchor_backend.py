#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
from collections import deque
import json
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
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.world_model import (
    VisualRecoveryBudget,
    apply_translation_offset,
    detections_to_anchor,
    project_detection_with_depth,
    transform_point_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse VLM 2D detections with RGB-D and TF into generic object anchors.")
    parser.add_argument("--detections-topic", default="/sonic_world/vlm_detections_2d")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/depth/camera_info")
    parser.add_argument("--anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--status-topic", default="/sonic_world/rgbd_anchor_status")
    parser.add_argument("--reobserve-topic", default="/sonic_world/perception_reobserve_cmd")
    parser.add_argument("--recovery-request-topic", default="/sonic_world/recovery_request")
    parser.add_argument("--expected-object-id", action="append", default=[], help="Object id required in a fused anchor; repeat for each required object.")
    parser.add_argument("--auto-reobserve-on-missing", action="store_true", help="Request bounded perception_reobserve recovery when an expected object is absent.")
    parser.add_argument("--reobserve-max-attempts", type=int, default=2)
    parser.add_argument("--reobserve-cooldown-s", type=float, default=1.0)
    parser.add_argument("--visual-recovery-escalate-navigation", action="store_true", help="After re-observation exhaustion, request one bounded navigation micro-adjust and runtime replan.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument(
        "--base-pose-from-current-map",
        action="store_true",
        default=True,
        help="Derive base_link pose from the calibrated map pose at the latest TF time (default).",
    )
    parser.add_argument(
        "--base-pose-from-source-stamp",
        dest="base_pose_from_current_map",
        action="store_false",
        help="Keep the legacy source-image timestamp transform for pose_base.",
    )
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--depth-cache-size", type=int, default=24, help="Number of RGB-D frames retained for delayed VLM results.")
    parser.add_argument("--max-depth-skew-s", type=float, default=0.20, help="Maximum source RGB/depth timestamp skew for fusion.")
    parser.add_argument("--map-z-min", type=float, help="Optional lower physical support-height bound for fused map poses.")
    parser.add_argument("--map-z-max", type=float, help="Optional upper physical support-height bound for fused map poses.")
    parser.add_argument(
        "--calibration-file",
        help="Optional shadow-trained translation calibration JSON. It is applied to RGB-D base/map poses before publishing.",
    )
    return parser.parse_args()


class RgbdAnchorBackend(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_rgbd_anchor_backend")
        self.args = args
        self.calibration = _load_calibration(args.calibration_file)
        self.depth_samples: deque[tuple[dict[str, int], np.ndarray, str]] = deque(maxlen=max(1, int(args.depth_cache_size)))
        self.camera_k: list[float] | None = None
        self.latest_detections: dict[str, Any] | None = None
        self.recovery_budget = VisualRecoveryBudget(
            list(args.expected_object_id),
            max_attempts=args.reobserve_max_attempts,
            cooldown_s=args.reobserve_cooldown_s,
        )
        self.navigation_escalated: set[str] = set()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.anchor_pub = self.create_publisher(String, args.anchor_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.recovery_request_pub = self.create_publisher(String, args.recovery_request_topic, 10)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, 2)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 2)
        self.create_subscription(String, args.detections_topic, self._detections_cb, 10)
        self.create_subscription(String, args.reobserve_topic, self._reobserve_cb, 10)

    def _depth_cb(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self._status("depth_rejected", "failed", error=f"unsupported encoding {msg.encoding}")
            return
        self.depth_samples.append(
            (_stamp_dict(msg.header.stamp), np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(int(msg.height), int(msg.width)).copy(), msg.header.frame_id or "camera_depth_optical_frame")
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_k = [float(value) for value in msg.k]

    def _detections_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("detection payload must be an object")
            self.latest_detections = payload
            objects = self._fuse_and_publish(payload, recovery_request=None)
            self._recover_missing(objects, payload, reason="expected_object_missing_after_rgbd_fusion")
        except Exception as exc:
            self._status("rgbd_anchor_failed", "failed", error=str(exc))
            self._recover_missing([], _json_object(msg.data), reason=f"rgbd_anchor_failure:{exc}")

    def _reobserve_cb(self, msg: String) -> None:
        request = _json_object(msg.data)
        if self.latest_detections is None:
            self._status("reobserve_blocked", "blocked", error="no VLM detections are available", request=request)
            return
        try:
            self._fuse_and_publish(self.latest_detections, recovery_request=request)
        except Exception as exc:
            self._status("reobserve_failed", "failed", error=str(exc), request=request)

    def _fuse_and_publish(self, payload: dict[str, Any], *, recovery_request: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not self.depth_samples or self.camera_k is None:
            raise RuntimeError("depth image and camera info are required")
        source_stamp = _normalize_stamp(payload.get("sensor_stamp"))
        depth_stamp, depth, depth_frame = self._select_depth(source_stamp)
        raw = payload.get("detections") or payload.get("objects") or []
        if not isinstance(raw, list) or not raw:
            raise ValueError("VLM payload contains no detections")
        objects = []
        for detection in raw:
            if not isinstance(detection, dict):
                continue
            record = project_detection_with_depth(
                detection,
                depth,
                self.camera_k,
                frame_id=depth_frame,
                patch_radius=self.args.patch_radius,
            )
            source_base = self._transform(record["pose_camera"], self.args.base_frame, source_stamp)
            try:
                record["pose_map"] = self._transform(record["pose_camera"], self.args.map_frame, source_stamp)
                if self.calibration["map_offset_m"] is not None:
                    record["pose_map"] = apply_translation_offset(record["pose_map"], self.calibration["map_offset_m"])
            except TransformException:
                pass
            if self.args.base_pose_from_current_map and isinstance(record.get("pose_map"), dict):
                # Qwen can answer seconds after the camera frame. Map coordinates stay meaningful, whereas
                # the camera-to-base transform at the old image timestamp does not represent the current robot pose.
                record["pose_base"] = self._transform(record["pose_map"], self.args.base_frame, None)
                record.setdefault("properties", {})["base_pose_source"] = "calibrated_map_current_tf"
            else:
                record["pose_base"] = source_base
                if self.calibration["base_offset_m"] is not None:
                    record["pose_base"] = apply_translation_offset(record["pose_base"], self.calibration["base_offset_m"])
                record.setdefault("properties", {})["base_pose_source"] = "source_image_tf"
            if not _map_pose_allowed(record.get("pose_map"), self.args.map_z_min, self.args.map_z_max):
                continue
            objects.append(record)
        if not objects:
            raise ValueError("no RGB-D detections remained after physical pose filtering")
        anchor = detections_to_anchor(
            objects,
            scene=payload.get("scene"),
            source="rgbd_vlm_anchor_backend",
            metadata={
                "recovery_request": recovery_request,
                "rgbd_fused": True,
                "sensor_stamp": source_stamp,
                "depth_sensor_stamp": depth_stamp,
                "translation_calibration": self.calibration["metadata"],
            },
        )
        out = String()
        out.data = json.dumps(anchor, separators=(",", ":"))
        self.anchor_pub.publish(out)
        self._status("rgbd_anchor_published", "completed", object_count=len(objects), request=recovery_request)
        return objects

    def _recover_missing(self, objects: list[dict[str, Any]], payload: dict[str, Any], *, reason: str) -> None:
        observed = {str(item.get("object_id") or "") for item in objects}
        missing = self.recovery_budget.observe(observed)
        self.navigation_escalated.difference_update(observed)
        if not missing:
            return
        self._status("rgbd_anchor_partial", "degraded", missing_object_ids=sorted(missing), reason=reason)
        if not self.args.auto_reobserve_on_missing:
            return
        for object_id in sorted(missing):
            attempt = self.recovery_budget.request(object_id)
            if attempt is None:
                if self.recovery_budget.attempts(object_id) >= self.recovery_budget.max_attempts:
                    self._status(
                        "visual_reobserve_exhausted",
                        "blocked",
                        target_id=object_id,
                        attempts=self.recovery_budget.attempts(object_id),
                        reason=reason,
                    )
                    self._escalate_navigation(object_id, payload, reason=reason)
                continue
            request = {
                "event": "recovery_request",
                "source": "rgbd_anchor_backend",
                "task_id": payload.get("task_id"),
                "action_id": f"visual-reobserve:{object_id}",
                "target_id": object_id,
                "handler": "perception_reobserve",
                "reason": reason,
                "attempt": attempt,
                "command": {
                    "type": "reobserve_from_current_view",
                    "expected_object_id": object_id,
                    "sensor_stamp": payload.get("sensor_stamp"),
                },
                "stamp": time.time(),
            }
            message = String()
            message.data = json.dumps(request, separators=(",", ":"))
            self.recovery_request_pub.publish(message)
            self._status(
                "visual_reobserve_requested",
                "issued",
                target_id=object_id,
                attempt=attempt,
                reason=reason,
                recovery_request_topic=self.args.recovery_request_topic,
            )

    def _escalate_navigation(self, object_id: str, payload: dict[str, Any], *, reason: str) -> None:
        if not self.args.visual_recovery_escalate_navigation or object_id in self.navigation_escalated:
            return
        self.navigation_escalated.add(object_id)
        request = {
            "event": "recovery_request",
            "source": "rgbd_anchor_backend",
            "task_id": payload.get("task_id"),
            "action_id": f"visual-micro-adjust:{object_id}",
            "target_id": object_id,
            "handler": "navigation_micro_adjust",
            "reason": f"visual_reobserve_exhausted:{reason}",
            "command": {"type": "micro_adjust_base_for_observation", "max_step_m": 0.08, "speed_mps": 0.05, "direction": 1.0},
            "stamp": time.time(),
        }
        message = String()
        message.data = json.dumps(request, separators=(",", ":"))
        self.recovery_request_pub.publish(message)
        self._status(
            "visual_navigation_micro_adjust_requested",
            "issued",
            target_id=object_id,
            reason=reason,
            recovery_request_topic=self.args.recovery_request_topic,
        )

    def _select_depth(self, source_stamp: dict[str, int] | None) -> tuple[dict[str, int], np.ndarray, str]:
        if not self.depth_samples:
            raise RuntimeError("depth image is required")
        if source_stamp is None:
            return self.depth_samples[-1]
        source_seconds = _stamp_seconds(source_stamp)
        selected = min(self.depth_samples, key=lambda item: abs(_stamp_seconds(item[0]) - source_seconds))
        skew = abs(_stamp_seconds(selected[0]) - source_seconds)
        if skew > max(0.0, float(self.args.max_depth_skew_s)):
            raise RuntimeError(f"no depth frame within {self.args.max_depth_skew_s:.3f}s of detection image (closest={skew:.3f}s)")
        return selected

    def _transform(self, pose: dict[str, Any], target_frame: str, sensor_stamp: dict[str, int] | None) -> dict[str, Any]:
        lookup_time = _rclpy_time(sensor_stamp)
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, str(pose.get("frame_id")), lookup_time)
        except TransformException:
            if sensor_stamp is None:
                raise
            # A delayed detector can start before a full TF history is available. Fall back visibly rather than dropping a valid anchor.
            transform = self.tf_buffer.lookup_transform(target_frame, str(pose.get("frame_id")), Time())
        t, q = transform.transform.translation, transform.transform.rotation
        return transform_point_pose(
            pose,
            {"translation": [t.x, t.y, t.z], "rotation_xyzw": [q.x, q.y, q.z, q.w]},
            frame_id=target_frame,
        )

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps({"event": "rgbd_anchor_status", "action": action, "status": status, **extra}, separators=(",", ":"))
        self.status_pub.publish(msg)
        detail = str(extra.get("error") or "")
        if status in {"failed", "blocked"}:
            self.get_logger().warn(f"{action}: {detail or status}")
        elif action == "rgbd_anchor_published":
            self.get_logger().info(f"{action}: objects={int(extra.get('object_count') or 0)}")


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _stamp_dict(stamp: Any) -> dict[str, int]:
    return {"sec": int(getattr(stamp, "sec", 0)), "nanosec": int(getattr(stamp, "nanosec", 0))}


def _normalize_stamp(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        sec, nanosec = int(value.get("sec", 0)), int(value.get("nanosec", 0))
    except (TypeError, ValueError):
        return None
    return {"sec": sec, "nanosec": nanosec} if sec or nanosec else None


def _stamp_seconds(stamp: dict[str, int]) -> float:
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


def _rclpy_time(stamp: dict[str, int] | None) -> Time:
    if stamp is None:
        return Time()
    return Time(seconds=int(stamp["sec"]), nanoseconds=int(stamp["nanosec"]))


def _map_pose_allowed(pose: Any, lower: float | None, upper: float | None) -> bool:
    if lower is None and upper is None:
        return True
    if not isinstance(pose, dict):
        return True
    position = pose.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return True
    try:
        z = float(position[2])
    except (TypeError, ValueError):
        return False
    return (lower is None or z >= float(lower)) and (upper is None or z <= float(upper))


def _load_calibration(value: str | None) -> dict[str, Any]:
    empty = {"base_offset_m": None, "map_offset_m": None, "metadata": {"applied": False}}
    if not value:
        return empty
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"calibration file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "sonic_rgbd_anchor_translation_calibration_v0":
        raise ValueError(f"unsupported RGB-D calibration schema: {path}")
    base_offset = _calibration_vector(payload.get("base_offset_m"), "base_offset_m")
    map_raw = payload.get("map_offset_m")
    map_offset = _calibration_vector(map_raw, "map_offset_m") if map_raw is not None else None
    return {
        "base_offset_m": base_offset,
        "map_offset_m": map_offset,
        "metadata": {
            "applied": True,
            "schema": str(payload["schema"]),
            "file": str(path),
            "base_sample_count": int(payload.get("base_sample_count") or 0),
        },
    }


def _calibration_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"calibration {label} must be a three-dimensional vector")
    try:
        return [float(value[index]) for index in range(3)]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration {label} must contain numbers") from exc


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RgbdAnchorBackend(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
