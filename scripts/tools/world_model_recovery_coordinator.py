#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


PERCEPTION_HANDLERS = {
    "object_anchor_update",
    "perception_reobserve",
    "world_memory_update",
    "place_target_recovery",
    "affordance_repair",
    "support_surface_inference",
}
NAVIGATION_HANDLERS = {"navigation_micro_adjust", "navigation_replan"}
RUNTIME_HANDLERS = {"runtime_replan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route /sonic_world/recovery_request messages to perception/navigation/runtime backends."
    )
    parser.add_argument("--recovery-request-topic", default="/sonic_world/recovery_request")
    parser.add_argument("--status-topic", default="/sonic_world/recovery_status")
    parser.add_argument("--perception-topic", default="/sonic_world/perception_recovery_request")
    parser.add_argument("--navigation-topic", default="/sonic_world/navigation_recovery_request")
    parser.add_argument("--runtime-topic", default="/sonic_world/runtime_recovery_request")
    parser.add_argument("--manual-topic", default="/sonic_world/manual_recovery_request")
    parser.add_argument(
        "--dedupe-window-s",
        type=float,
        default=1.0,
        help="Suppress identical recovery requests seen within this many seconds.",
    )
    return parser.parse_args()


class WorldModelRecoveryCoordinator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_recovery_coordinator")
        self.args = args
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.route_pubs = {
            "perception": self.create_publisher(String, args.perception_topic, 10),
            "navigation": self.create_publisher(String, args.navigation_topic, 10),
            "runtime": self.create_publisher(String, args.runtime_topic, 10),
            "manual": self.create_publisher(String, args.manual_topic, 10),
        }
        self._last_seen: dict[tuple[str, str, str, str, str, str], float] = {}
        self.create_subscription(String, args.recovery_request_topic, self._request_cb, 10)
        self.get_logger().info(
            "Recovery coordinator listening on "
            f"{args.recovery_request_topic}; routes: "
            f"perception={args.perception_topic} navigation={args.navigation_topic} "
            f"runtime={args.runtime_topic} manual={args.manual_topic}"
        )

    def _request_cb(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
            if not isinstance(request, dict):
                raise ValueError("recovery request payload must be an object")
            self._handle_request(request)
        except Exception as exc:
            self._publish_status(
                {
                    "event": "recovery_route_error",
                    "status": "error",
                    "error": str(exc),
                    "raw": msg.data[:500],
                }
            )

    def _handle_request(self, request: dict[str, Any]) -> None:
        handler = str(request.get("handler") or "")
        route = _route_for_handler(handler)
        key = _dedupe_key(request)
        now = self.get_clock().now().nanoseconds / 1e9
        previous = self._last_seen.get(key)
        if previous is not None and now - previous < float(self.args.dedupe_window_s):
            self._publish_status(_status_payload(request, route, "duplicate_suppressed"))
            return
        self._last_seen[key] = now

        routed = _routed_payload(request, route)
        self._publish_json(self.route_pubs[route], routed)
        self._publish_status(_status_payload(request, route, "routed"))

    def _publish_status(self, payload: dict[str, Any]) -> None:
        self._publish_json(self.status_pub, payload)

    def _publish_json(self, publisher, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        publisher.publish(msg)


def _route_for_handler(handler: str) -> str:
    if handler in PERCEPTION_HANDLERS:
        return "perception"
    if handler in NAVIGATION_HANDLERS:
        return "navigation"
    if handler in RUNTIME_HANDLERS:
        return "runtime"
    return "manual"


def _dedupe_key(request: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    command = request.get("command") if isinstance(request.get("command"), dict) else {}
    return (
        str(request.get("source") or ""),
        str(request.get("action_id") or ""),
        str(request.get("handler") or ""),
        str(request.get("target_id") or ""),
        str(command.get("type") or ""),
        # A retry deliberately reuses the failed dispatch action id.  It is a
        # distinct repair request, so suppress only duplicate publications of
        # the same attempt rather than suppressing the recovery loop itself.
        str(command.get("attempt") or 0),
    )


def _routed_payload(request: dict[str, Any], route: str) -> dict[str, Any]:
    return {
        "event": "recovery_routed",
        "route": route,
        "request": request,
        "handler": request.get("handler"),
        "target_id": request.get("target_id"),
        "command": request.get("command") or {},
    }


def _status_payload(request: dict[str, Any], route: str, status: str) -> dict[str, Any]:
    return {
        "event": "recovery_status",
        "status": status,
        "route": route,
        "task_id": request.get("task_id"),
        "action_id": request.get("action_id"),
        "handler": request.get("handler"),
        "target_id": request.get("target_id"),
        "source": request.get("source"),
        "reason": request.get("reason"),
    }


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WorldModelRecoveryCoordinator(args)
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
