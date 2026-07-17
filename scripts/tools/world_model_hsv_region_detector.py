#!/usr/bin/env -S /usr/bin/python3
"""Deterministic HSV region detector for colored simulation benchmark objects."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish generic 2D detections from configured HSV color regions.")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--request-topic", default="/sonic_world/perception_reobserve_cmd")
    parser.add_argument("--detections-topic", default="/sonic_world/vlm_detections_2d")
    parser.add_argument("--status-topic", default="/sonic_world/hsv_detector_status")
    parser.add_argument("--period", type=float, default=0.0)
    parser.add_argument("--classes-json", required=True, help="JSON list: object_id, category, hsv_lower, hsv_upper, min_area.")
    return parser.parse_args()


class HsvRegionDetector(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_hsv_region_detector")
        self.args = args
        self.classes = _load_classes(args.classes_json)
        self.latest_image: np.ndarray | None = None
        self.latest_frame = "camera_depth_optical_frame"
        self.detections_pub = self.create_publisher(String, args.detections_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 2)
        self.create_subscription(String, args.request_topic, self._request_cb, 10)
        if args.period > 0.0:
            self.create_timer(max(0.2, float(args.period)), self._detect)
        self.get_logger().info(f"HSV detector classes={len(self.classes)} period={args.period}")

    def _image_cb(self, msg: Image) -> None:
        if msg.encoding not in {"rgb8", "bgr8"}:
            return
        image = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(int(msg.height), int(msg.width), 3)
        self.latest_image = image.copy() if msg.encoding == "rgb8" else image[:, :, ::-1].copy()
        self.latest_frame = msg.header.frame_id or self.latest_frame

    def _request_cb(self, _msg: String) -> None:
        self._detect()

    def _detect(self) -> None:
        if self.latest_image is None:
            self._status("hsv_no_image", "blocked")
            return
        started = time.monotonic()
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_RGB2HSV)
        detections: list[dict[str, Any]] = []
        for spec in self.classes:
            lower = np.asarray(spec["hsv_lower"], dtype=np.uint8)
            upper = np.asarray(spec["hsv_upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            candidates = [
                contour
                for contour in contours
                if float(spec["min_area"]) <= float(cv2.contourArea(contour)) <= float(spec["max_area"])
            ]
            if not candidates:
                continue
            contour = max(candidates, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(
                {
                    "object_id": spec["object_id"],
                    "category": spec["category"],
                    "bbox": [int(x), int(y), int(x + width), int(y + height)],
                    "confidence": 1.0,
                    "tracking_id": spec["object_id"],
                    "support": spec.get("support"),
                    "shape": spec.get("shape", "unknown"),
                    "affordances": spec.get("affordances", []),
                }
            )
        out = String()
        out.data = json.dumps(
            {
                "schema": "sonic_vlm_detections_2d_v0",
                "source": "hsv_region_detector",
                "frame_id": self.latest_frame,
                "detections": detections,
            },
            separators=(",", ":"),
        )
        self.detections_pub.publish(out)
        self._status("hsv_detections", "completed", object_count=len(detections), elapsed_s=time.monotonic() - started)

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps({"event": "hsv_detector_status", "action": action, "status": status, **extra}, separators=(",", ":"))
        self.status_pub.publish(msg)
        if action == "hsv_detections":
            self.get_logger().info(f"{action}: objects={extra.get('object_count', 0)}")
        elif status in {"failed", "blocked"}:
            self.get_logger().warn(f"{action}: {extra.get('error') or status}")


def _load_classes(value: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--classes-json is invalid: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("--classes-json must be a non-empty list")
    classes = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("HSV class entries must be objects")
        lower, upper = item.get("hsv_lower"), item.get("hsv_upper")
        if not isinstance(lower, list) or not isinstance(upper, list) or len(lower) != 3 or len(upper) != 3:
            raise ValueError("HSV class entries require hsv_lower/hsv_upper RGB-HSV triplets")
        max_area = float(item.get("max_area") or float("inf"))
        if max_area <= 0.0:
            raise ValueError("HSV class max_area must be positive")
        classes.append(
            {
                "object_id": str(item.get("object_id") or item.get("category") or "object"),
                "category": str(item.get("category") or "object"),
                "hsv_lower": [max(0, min(255, int(number))) for number in lower],
                "hsv_upper": [max(0, min(255, int(number))) for number in upper],
                "min_area": max(1.0, float(item.get("min_area") or 8.0)),
                "max_area": max_area,
                "support": item.get("support"),
                "shape": item.get("shape", "unknown"),
                "affordances": item.get("affordances") if isinstance(item.get("affordances"), list) else [],
            }
        )
    return classes


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = HsvRegionDetector(args)
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
