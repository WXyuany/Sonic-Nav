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
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.planners import TaskRequest, task_request_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish one world-model task request and wait for primitive execution status."
    )
    parser.add_argument("--task-request-topic", default="/sonic_world/task_request")
    parser.add_argument("--decision-topic", default="/sonic_world/decision_plan")
    parser.add_argument("--primitive-status-topic", default="/sonic_world/primitive_status")
    parser.add_argument("--executor-event-topic", default="/sonic_world/executor_event")
    parser.add_argument("--recovery-status-topic", default="/sonic_world/recovery_status")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--task", "--verb", dest="verb", required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--object-category")
    parser.add_argument("--target-id")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--output-jsonl", help="Optional JSONL report path.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--publish-period", type=float, default=0.25)
    parser.add_argument("--settle-after-success", type=float, default=0.2)
    parser.add_argument("--allow-recovery", action="store_true", default=True)
    parser.add_argument("--no-allow-recovery", dest="allow_recovery", action="store_false")
    return parser.parse_args()


def _live_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=16,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class AutonomousTaskNode(Node):
    def __init__(self, args: argparse.Namespace, request: TaskRequest, output_path: Path | None):
        super().__init__("sonic_world_autonomous_task")
        self.args = args
        self.request = request
        self.request_json = task_request_to_json(request)
        self.output_path = output_path
        self.request_pub = self.create_publisher(String, args.task_request_topic, _live_qos())
        self.create_subscription(String, args.decision_topic, self._decision_cb, _latched_qos())
        self.create_subscription(String, args.primitive_status_topic, self._primitive_status_cb, _live_qos())
        self.create_subscription(String, args.executor_event_topic, self._executor_event_cb, _live_qos())
        self.create_subscription(String, args.recovery_status_topic, self._recovery_status_cb, _live_qos())
        self.expected_actions: dict[str, dict[str, Any]] = {}
        self.primitive_status: dict[str, dict[str, Any]] = {}
        self.recovery_events: list[dict[str, Any]] = []
        self.executor_events: list[dict[str, Any]] = []
        self.final_status: str | None = None
        self.final_reason = ""
        self._last_publish = 0.0
        self._decision_seen = False
        self._log(
            "task_start",
            {
                "run_id": args.run_id,
                "task_id": args.task_id,
                "request": request.to_dict(),
            },
        )

    def tick(self) -> None:
        now = time.monotonic()
        if self.final_status is not None:
            return
        if now - self._last_publish >= max(0.05, float(self.args.publish_period)) and not self._decision_seen:
            msg = String()
            msg.data = self.request_json
            self.request_pub.publish(msg)
            self._last_publish = now
            self._log("task_request_published", {"request": self.request.to_dict()})

    def _decision_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None:
            return
        source = str(payload.get("source") or "")
        if source == "anchor_replan":
            return
        plan = payload.get("decision_plan") if isinstance(payload.get("decision_plan"), dict) else payload
        if not isinstance(plan, dict):
            return
        plan_task_id = str(plan.get("task_id") or "")
        if self.args.task_id and plan_task_id != str(self.args.task_id):
            return
        actions = [item for item in plan.get("actions", []) if isinstance(item, dict)]
        dispatch = [item for item in actions if item.get("kind") == "dispatch"]
        recovery = [item for item in actions if item.get("kind") == "recovery"]
        self._decision_seen = bool(dispatch or recovery)
        self.expected_actions.clear()
        for action in dispatch:
            action_id = str(action.get("action_id") or "")
            if not action_id:
                continue
            self.expected_actions[action_id] = action
        self._log(
            "decision_plan",
            {
                "status": plan.get("status"),
                "dispatch_action_count": len(dispatch),
                "recovery_action_count": len(recovery),
                "expected_action_ids": sorted(self.expected_actions),
            },
        )
        if plan.get("status") == "needs_recovery" and not self.args.allow_recovery:
            self.final_status = "failed"
            self.final_reason = "decision_needs_recovery"
        self._maybe_complete()

    def _primitive_status_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None or payload.get("event") != "primitive_status":
            return
        action_id = str(payload.get("action_id") or "")
        if action_id:
            self.primitive_status[action_id] = payload
        self._log("primitive_status", payload)
        status = str(payload.get("status") or "")
        if status == "skipped":
            self.final_status = "failed"
            self.final_reason = "primitive_skipped_without_actuation"
        elif status in {"failed", "error", "timeout"} and not self.args.allow_recovery:
            self.final_status = "failed"
            self.final_reason = f"primitive_{status}"
        self._maybe_complete()

    def _executor_event_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None:
            return
        self.executor_events.append(payload)
        if payload.get("event") in {
            "decision_action",
            "primitive_status",
            "recovery_request",
            "execution_transition",
            "plan_terminal",
        }:
            self._log("executor_event", payload)
        if payload.get("event") == "plan_terminal":
            terminal_status = str(payload.get("status") or "")
            if terminal_status == "succeeded":
                self.final_status = "success"
                self.final_reason = "executor_plan_succeeded"
            elif terminal_status in {"failed", "error", "cancelled", "timeout"}:
                self.final_status = "failed"
                self.final_reason = str(payload.get("reason") or f"executor_plan_{terminal_status}")
        if (
            payload.get("event") == "execution_transition"
            and payload.get("kind") == "action_failed"
            and not self.args.allow_recovery
        ):
            self.final_status = "failed"
            self.final_reason = str(payload.get("reason") or "executor_action_failed")

    def _recovery_status_cb(self, msg: String) -> None:
        payload = _json_object(msg.data)
        if payload is None:
            return
        self.recovery_events.append(payload)
        self._log("recovery_status", payload)

    def _maybe_complete(self) -> None:
        if self.final_status is not None:
            return
        if not self.expected_actions:
            return
        # Completion is owned by the executor after effect verification. Raw
        # primitive success is intentionally insufficient here.

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "event": event,
            "stamp": time.time(),
            "run_id": self.args.run_id,
            "task_id": self.args.task_id,
            **payload,
        }
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def finish(self, status: str, reason: str) -> None:
        self.final_status = status
        self.final_reason = reason
        self._log(
            "task_end",
            {
                "status": status,
                "reason": reason,
                "expected_action_count": len(self.expected_actions),
                "primitive_status_count": len(self.primitive_status),
                "recovery_event_count": len(self.recovery_events),
            },
        )


def main() -> int:
    args = parse_args()
    request = _request_from_args(args)
    output_path = Path(args.output_jsonl).expanduser() if args.output_jsonl else None
    if output_path is not None and output_path.exists():
        output_path.unlink()
    rclpy.init()
    node = AutonomousTaskNode(args, request, output_path)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + max(0.1, float(args.timeout))
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            node.tick()
            executor.spin_once(timeout_sec=0.05)
            if node.final_status is not None:
                if node.final_status == "success":
                    settle_deadline = time.monotonic() + max(0.0, float(args.settle_after_success))
                    while time.monotonic() < settle_deadline:
                        executor.spin_once(timeout_sec=0.05)
                break
        if node.final_status is None:
            node.finish("failed", "timeout")
        else:
            node.finish(node.final_status, node.final_reason)
        return 0 if node.final_status == "success" else 1
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _request_from_args(args: argparse.Namespace) -> TaskRequest:
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --metadata-json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must decode to an object")
    request_id = args.task_id or args.run_id
    if request_id:
        metadata.setdefault("request_id", request_id)
    if args.run_id:
        metadata.setdefault("run_id", args.run_id)
    return TaskRequest(
        verb=str(args.verb),
        object_id=args.object_id,
        object_category=args.object_category,
        target_id=args.target_id,
        metadata=metadata,
    )


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
