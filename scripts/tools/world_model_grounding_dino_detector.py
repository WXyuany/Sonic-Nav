#!/usr/bin/env -S /usr/bin/python3
"""ROS client for a local Grounding DINO bbox service."""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use local Grounding DINO to publish task-conditioned 2D detections.")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--request-topic", default="/sonic_world/perception_reobserve_cmd")
    parser.add_argument("--detections-topic", default="/sonic_world/vlm_detections_2d")
    parser.add_argument("--status-topic", default="/sonic_world/grounding_dino_status")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/detections")
    parser.add_argument("--classes-json", required=True, help="JSON list with object_id, category, label, support, shape.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--period", type=float, default=0.0)
    return parser.parse_args()


class GroundingDinoDetector(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_grounding_dino_detector")
        self.args = args
        self.classes = _load_classes(args.classes_json)
        self.latest_image: np.ndarray | None = None
        self.latest_frame = "camera_depth_optical_frame"
        self.latest_sensor_stamp: dict[str, int] | None = None
        self.busy = False
        self.detections_pub = self.create_publisher(String, args.detections_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 2)
        self.create_subscription(String, args.request_topic, self._request_cb, 10)
        if args.period > 0.0:
            self.create_timer(max(0.2, float(args.period)), self._detect)

    def _image_cb(self, msg: Image) -> None:
        if msg.encoding not in {"rgb8", "bgr8"}:
            return
        image = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(int(msg.height), int(msg.width), 3)
        self.latest_image = image.copy() if msg.encoding == "rgb8" else image[:, :, ::-1].copy()
        self.latest_frame = msg.header.frame_id or self.latest_frame
        self.latest_sensor_stamp = _stamp_dict(msg.header.stamp)

    def _request_cb(self, _msg: String) -> None:
        self._detect()

    def _detect(self) -> None:
        if self.busy or self.latest_image is None:
            return
        self.busy = True
        threading.Thread(
            target=self._detect_sync,
            args=(self.latest_image.copy(), self.latest_frame, dict(self.latest_sensor_stamp or {})),
            daemon=True,
        ).start()

    def _detect_sync(self, image: np.ndarray, frame_id: str, sensor_stamp: dict[str, int]) -> None:
        started = time.monotonic()
        try:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("failed to encode image")
            response = requests.post(
                self.args.endpoint,
                json={
                    "image_url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "labels": [item["label"] for item in self.classes],
                    "box_threshold": float(self.args.box_threshold),
                    "text_threshold": float(self.args.text_threshold),
                },
                timeout=max(1.0, float(self.args.timeout)),
            )
            response.raise_for_status()
            raw = response.json().get("detections")
            if not isinstance(raw, list):
                raise ValueError("Grounding DINO response has no detections list")
            detections = _normalize(raw, self.classes)
            msg = String()
            msg.data = json.dumps(
                {
                    "schema": "sonic_vlm_detections_2d_v0",
                    "source": "grounding_dino",
                    "frame_id": frame_id,
                    "sensor_stamp": sensor_stamp,
                    "detections": detections,
                },
                separators=(",", ":"),
            )
            self.detections_pub.publish(msg)
            self._status("grounding_dino_detections", "completed", object_count=len(detections), elapsed_s=time.monotonic() - started)
            self.get_logger().info(f"grounding_dino_detections: objects={len(detections)}")
        except Exception as exc:
            self._status("grounding_dino_failed", "failed", error=str(exc), elapsed_s=time.monotonic() - started)
            self.get_logger().warn(f"grounding_dino_failed: {exc}")
        finally:
            self.busy = False

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps({"event": "grounding_dino_status", "action": action, "status": status, **extra}, separators=(",", ":"))
        self.status_pub.publish(msg)


def _load_classes(value: str) -> list[dict[str, Any]]:
    raw = json.loads(value)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--classes-json must be a non-empty list")
    classes = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("class entries must be objects")
        label = str(item.get("label") or item.get("category") or "object").strip()
        if not label:
            raise ValueError("class label is required")
        classes.append({**item, "label": label, "object_id": str(item.get("object_id") or label), "category": str(item.get("category") or label)})
    return classes


def _stamp_dict(stamp: Any) -> dict[str, int]:
    return {"sec": int(getattr(stamp, "sec", 0)), "nanosec": int(getattr(stamp, "nanosec", 0))}


def _normalize(raw: list[Any], classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {item["label"].lower(): item for item in classes}
    selected: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        spec = by_label.get(label)
        bbox = item.get("bbox")
        if spec is None or not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox[:4])
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        area = (x1 - x0) * (y1 - y0)
        try:
            min_area = max(0.0, float(spec.get("min_area") or 0.0))
            max_area = float(spec.get("max_area") or float("inf"))
        except (TypeError, ValueError):
            continue
        if area < min_area or area > max_area:
            continue
        record = {**spec, "bbox": [x0, y0, x1, y1], "confidence": confidence, "tracking_id": spec["object_id"]}
        current = selected.get(spec["object_id"])
        if current is None or confidence > float(current["confidence"]):
            selected[spec["object_id"]] = record
    return list(selected.values())


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GroundingDinoDetector(args)
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
