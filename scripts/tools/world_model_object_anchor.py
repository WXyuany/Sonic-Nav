#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.world_model import load_anchor_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a Sonic world-model object anchor JSON.")
    parser.add_argument("--topic", default="/sonic_world/object_anchor")
    parser.add_argument("--json", help="Raw object-anchor JSON.")
    parser.add_argument("--file", help="Path to object-anchor JSON.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--period", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.json and not args.file:
        raise SystemExit("--json or --file is required")
    anchor = load_anchor_payload(args.file or args.json)
    payload = json.dumps(anchor, separators=(",", ":"))

    rclpy.init()
    node = rclpy.create_node("sonic_world_object_anchor")
    pub = node.create_publisher(String, args.topic, 10)
    try:
        for idx in range(max(1, int(args.repeat))):
            msg = String()
            msg.data = payload
            pub.publish(msg)
            node.get_logger().info(f"published object anchor {idx + 1}: {payload[:240]}")
            rclpy.spin_once(node, timeout_sec=0.1)
            if idx + 1 < int(args.repeat):
                time.sleep(max(0.0, float(args.period)))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
