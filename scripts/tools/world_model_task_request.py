#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
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

from sonic_world.planners import TaskRequest, task_request_from_json, task_request_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a Sonic world-model task request JSON.")
    parser.add_argument("--topic", default="/sonic_world/task_request")
    parser.add_argument("--json", help="Raw task request JSON.")
    parser.add_argument("--file", help="Path to task request JSON.")
    parser.add_argument("--task", "--verb", dest="verb", help="Task verb, e.g. navigate, pick, move.")
    parser.add_argument("--object", "--object-id", dest="object_id")
    parser.add_argument("--object-category")
    parser.add_argument("--target", "--target-id", dest="target_id")
    parser.add_argument("--request-id")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--period", type=float, default=0.5)
    return parser.parse_args()


def _build_request(args: argparse.Namespace) -> TaskRequest:
    if args.file:
        return task_request_from_json(open(args.file, "r", encoding="utf-8").read())
    if args.json:
        return task_request_from_json(args.json)
    if not args.verb:
        raise SystemExit("--task/--verb or --json is required")
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --metadata-json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must decode to an object")
    if args.request_id:
        metadata["request_id"] = args.request_id
    return TaskRequest(
        verb=args.verb,
        object_id=args.object_id,
        object_category=args.object_category,
        target_id=args.target_id,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    request = _build_request(args)
    payload = task_request_to_json(request)

    rclpy.init()
    node = rclpy.create_node("sonic_world_task_request")
    pub = node.create_publisher(String, args.topic, 10)
    try:
        for idx in range(max(1, int(args.repeat))):
            msg = String()
            msg.data = payload
            pub.publish(msg)
            node.get_logger().info(f"published task request {idx + 1}: {payload}")
            rclpy.spin_once(node, timeout_sec=0.1)
            if idx + 1 < int(args.repeat):
                time.sleep(max(0.0, float(args.period)))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
