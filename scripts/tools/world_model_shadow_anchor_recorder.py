#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record paired privileged and Qwen/RGB-D anchors for shadow evaluation.")
    parser.add_argument("--reference-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--prediction-topic", default="/sonic_world/qwen_rgbd_anchor")
    parser.add_argument("--reference-output", default="reports/perception/privileged_anchors.jsonl")
    parser.add_argument("--prediction-output", default="reports/perception/qwen_rgbd_anchors.jsonl")
    parser.add_argument("--max-pairs", type=int, default=0, help="Stop after this many pairs; 0 records indefinitely.")
    parser.add_argument("--max-skew-s", type=float, default=0.75)
    return parser.parse_args()


class ShadowAnchorRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_shadow_anchor_recorder")
        self.args = args
        self.reference: tuple[float, dict[str, Any]] | None = None
        self.prediction: tuple[float, dict[str, Any]] | None = None
        self.pair_count = 0
        self.reference_output = _repo_path(args.reference_output)
        self.prediction_output = _repo_path(args.prediction_output)
        self.reference_output.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        self.create_subscription(String, args.reference_topic, self._reference_cb, 10)
        self.create_subscription(String, args.prediction_topic, self._prediction_cb, 10)
        self.get_logger().info(
            f"Shadow recorder: reference={args.reference_topic} prediction={args.prediction_topic} "
            f"max_skew={args.max_skew_s:.2f}s"
        )

    def _reference_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None:
            return
        try:
            self.reference = (time.monotonic(), _normalize_anchor(payload))
            self._write_pair_if_ready()
        except Exception as exc:
            self.get_logger().warn(f"reference anchor rejected: {exc}")

    def _prediction_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None:
            return
        try:
            self.prediction = (time.monotonic(), _normalize_anchor(payload))
            self._write_pair_if_ready()
        except Exception as exc:
            self.get_logger().warn(f"prediction anchor rejected: {exc}")

    def _write_pair_if_ready(self) -> None:
        if self.reference is None or self.prediction is None:
            return
        reference_time, reference = self.reference
        prediction_time, prediction = self.prediction
        skew = abs(reference_time - prediction_time)
        if skew > max(0.0, float(self.args.max_skew_s)):
            if reference_time < prediction_time:
                self.reference = None
            else:
                self.prediction = None
            return
        self.pair_count += 1
        sample_id = f"shadow_{self.pair_count:06d}"
        metadata = {"sample_id": sample_id, "recorded_at": time.time(), "pair_skew_s": round(skew, 5)}
        _append_jsonl(self.reference_output, {**reference, **metadata, "source": "privileged_shadow_reference"})
        _append_jsonl(self.prediction_output, {**prediction, **metadata, "source": "qwen_rgbd_shadow_prediction"})
        self.get_logger().info(f"recorded shadow pair {sample_id} skew={skew:.3f}s")
        self.reference = None
        self.prediction = None
        if int(self.args.max_pairs) > 0 and self.pair_count >= int(self.args.max_pairs):
            self.get_logger().info("shadow pair limit reached; shutting down")
            rclpy.shutdown()


def _normalize_anchor(payload: dict[str, Any]) -> dict[str, Any]:
    world = anchor_to_world(payload)
    return {
        "scene": world.properties.get("scene"),
        "frame_id": world.frame_id,
        "objects": [object_.to_dict() for object_ in world.objects.values()],
        "relations": [relation.to_dict() for relation in world.relations],
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ShadowAnchorRecorder(args)
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
