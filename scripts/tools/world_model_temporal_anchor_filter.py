#!/usr/bin/env -S /usr/bin/python3
"""ROS stream wrapper for temporal generic-anchor stabilization."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.world_model.temporal_anchor import TemporalAnchorFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish temporally stabilized generic object anchors.")
    parser.add_argument("--input-topic", default="/sonic_world/qwen_rgbd_anchor")
    parser.add_argument("--output-topic", default="/sonic_world/qwen_rgbd_anchor_temporal")
    parser.add_argument("--status-topic", default="/sonic_world/temporal_anchor_status")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--min-observations", type=int, default=3)
    return parser.parse_args()


class TemporalAnchorFilterNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_temporal_anchor_filter")
        self.filter = TemporalAnchorFilter(window_size=args.window_size, min_observations=args.min_observations)
        self.output_pub = self.create_publisher(String, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(String, args.input_topic, self._anchor_cb, 10)
        self.get_logger().info(
            f"Temporal anchor filter: input={args.input_topic} output={args.output_topic} "
            f"window={args.window_size} min_observations={args.min_observations}"
        )

    def _anchor_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("anchor must be an object")
            stable = self.filter.update(payload)
            out = String()
            out.data = json.dumps(stable, separators=(",", ":"))
            self.output_pub.publish(out)
            self._status("temporal_anchor_published", "completed", object_count=len(stable.get("objects") or []))
        except Exception as exc:
            self._status("temporal_anchor_failed", "failed", error=str(exc))
            self.get_logger().warn(f"temporal anchor failed: {exc}")

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps({"event": "temporal_anchor_status", "action": action, "status": status, **extra}, separators=(",", ":"))
        self.status_pub.publish(msg)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TemporalAnchorFilterNode(args)
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
