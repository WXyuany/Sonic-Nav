#!/usr/bin/env -S /usr/bin/python3
"""Promote a shadow VLM anchor stream only after a recorded evaluation passes."""
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
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.world_model import anchor_to_world
from sonic_world.world_model.vlm_gate import load_passing_vlm_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward Qwen/RGB-D anchors only when a VLM evaluation gate passed.")
    parser.add_argument("--gate-report", required=True, help="Passing output from world_model_vlm_anchor_eval.py.")
    parser.add_argument("--input-topic", default="/sonic_world/qwen_rgbd_anchor")
    parser.add_argument("--anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--status-topic", default="/sonic_world/vlm_anchor_gate_status")
    return parser.parse_args()


class VlmAnchorGate(Node):
    def __init__(self, args: argparse.Namespace, report: dict[str, Any]):
        super().__init__("sonic_world_vlm_anchor_gate")
        self.args = args
        self.report = report
        self.forwarded_count = 0
        self.anchor_pub = self.create_publisher(String, args.anchor_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(String, args.input_topic, self._anchor_cb, 10)
        self._status("vlm_anchor_gate_ready", "ready", gate_report=str(args.gate_report))
        self.get_logger().info(
            f"VLM anchor gate passed: input={args.input_topic} output={args.anchor_topic} report={args.gate_report}"
        )

    def _anchor_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("anchor payload must be an object")
            # Validate the generic contract before letting a visual estimate reach world memory.
            anchor_to_world(payload)
        except Exception as exc:
            self._status("vlm_anchor_gate_rejected", "rejected", error=str(exc))
            return
        self.anchor_pub.publish(msg)
        self.forwarded_count += 1
        if self.forwarded_count == 1:
            self.get_logger().info(f"forwarded first validated visual anchor to {self.args.anchor_topic}")
        self._status("vlm_anchor_gate_forwarded", "completed")

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps(
            {"event": "vlm_anchor_gate_status", "action": action, "status": status, **extra},
            separators=(",", ":"),
        )
        self.status_pub.publish(msg)


def main() -> None:
    args = parse_args()
    report = load_passing_vlm_gate(_repo_path(args.gate_report))
    rclpy.init()
    node = VlmAnchorGate(args, report)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


if __name__ == "__main__":
    main()
