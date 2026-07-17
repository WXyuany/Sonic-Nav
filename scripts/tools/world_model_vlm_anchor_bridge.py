#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.world_model import detection_payload_to_anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize VLM/RGB-D detection JSON into /sonic_world/object_anchor messages."
    )
    parser.add_argument("--detections-topic", default="/sonic_world/vlm_detections")
    parser.add_argument("--anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--status-topic", default="/sonic_world/vlm_anchor_status")
    return parser.parse_args()


class VlmAnchorBridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_vlm_anchor_bridge")
        self.args = args
        self.anchor_pub = self.create_publisher(String, args.anchor_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(String, args.detections_topic, self._detections_cb, 10)
        self.get_logger().info(
            f"VLM anchor bridge listening detections={args.detections_topic} anchor={args.anchor_topic}"
        )

    def _detections_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("detection payload must be a JSON object")
            anchor = detection_payload_to_anchor(payload)
        except Exception as exc:
            self._publish_status({"event": "vlm_anchor_error", "status": "error", "error": str(exc)})
            return
        out = String()
        out.data = json.dumps(anchor, separators=(",", ":"))
        self.anchor_pub.publish(out)
        self._publish_status(
            {
                "event": "vlm_anchor_published",
                "status": "success",
                "object_count": len(anchor.get("objects") or []),
                "scene": anchor.get("scene"),
                "source": anchor.get("source"),
            }
        )

    def _publish_status(self, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.status_pub.publish(msg)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = VlmAnchorBridge(args)
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
