#!/usr/bin/env python3
"""Serial ROS client for an already-running carry-state world-model episode."""
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
    parser = argparse.ArgumentParser(description="Execute episode-manifest stages without resetting world-model or MuJoCo state.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-request-topic", default="/sonic_world/task_request")
    parser.add_argument("--world-topic", default="/sonic_world/model")
    parser.add_argument("--decision-topic", default="/sonic_world/decision_plan")
    parser.add_argument("--policy-topic", default="/sonic_world/policy_action")
    parser.add_argument("--primitive-status-topic", default="/sonic_world/primitive_status")
    parser.add_argument("--executor-event-topic", default="/sonic_world/executor_event")
    parser.add_argument("--recovery-status-topic", default="/sonic_world/recovery_status")
    parser.add_argument("--output-jsonl", default="reports/episodes/latest.jsonl")
    parser.add_argument("--timeout-per-stage", type=float, default=120.0)
    parser.add_argument("--publish-period", type=float, default=0.25)
    parser.add_argument("--world-ready-timeout", type=float, default=45.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--stage-start", type=int, default=1, help="First manifest stage_index to execute (inclusive).")
    parser.add_argument("--stage-stop", type=int, default=0, help="Last manifest stage_index to execute (inclusive); 0 executes through the end.")
    return parser.parse_args()


def _live_qos() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=16, reliability=QoSReliabilityPolicy.RELIABLE)


def _latched_qos() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class EpisodeNode(Node):
    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any], output: Path):
        super().__init__("sonic_world_autonomous_episode")
        stages = manifest.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("episode manifest requires non-empty stages")
        self.args, self.manifest, self.stages, self.output = args, manifest, stages, output
        self.episode_scope = "full_sequence" if len(stages) == int(manifest.get("stage_count") or len(stages)) else "curriculum_stage"
        self.index = 0
        self.stage_started = 0.0
        self.world_wait_started = 0.0
        self.world_ready = False
        self.bootstrap_published = False
        self.last_publish = 0.0
        self.finished = False
        self.failed_stages: list[str] = []
        self.request_pub = self.create_publisher(String, args.task_request_topic, _live_qos())
        self.create_subscription(String, args.world_topic, self._world_cb, _latched_qos())
        self.create_subscription(String, args.decision_topic, self._decision_cb, _latched_qos())
        self.create_subscription(String, args.policy_topic, self._policy_cb, _latched_qos())
        self.create_subscription(String, args.primitive_status_topic, self._primitive_cb, _live_qos())
        self.create_subscription(String, args.executor_event_topic, self._executor_event_cb, _live_qos())
        self.create_subscription(String, args.recovery_status_topic, self._recovery_cb, _live_qos())
        self._start_stage()

    @property
    def stage(self) -> dict[str, Any]:
        return self.stages[self.index]

    @property
    def task_id(self) -> str:
        return str(self.stage.get("task_id") or "")

    @property
    def stage_index(self) -> int:
        """Preserve the manifest identity when running a curriculum slice."""
        return int(self.stage.get("stage_index") or self.index + 1)

    def tick(self) -> None:
        if self.finished:
            return
        now = time.monotonic()
        if not self.world_ready:
            if not self.bootstrap_published:
                self._publish_request()
                self.bootstrap_published = True
                self._log("world_anchor_bootstrap_request", {"stage_index": self.stage_index, "task_id": self.task_id})
            if now - self.world_wait_started > max(0.1, float(self.args.world_ready_timeout)):
                self._terminal("failed", "world_anchor_not_ready")
            return
        if now - self.last_publish >= max(0.05, float(self.args.publish_period)):
            self._publish_request()
            self.last_publish = now
        if now - self.stage_started > max(0.1, float(self.args.timeout_per_stage)):
            self._terminal("failed", "stage_timeout")

    def _start_stage(self) -> None:
        self.stage_started = time.monotonic()
        self.world_wait_started = self.stage_started
        self.world_ready = False
        self.bootstrap_published = False
        self.last_publish = 0.0
        self._log("stage_start", {"sequence_id": self.manifest.get("sequence_id"), "stage_index": self.stage_index, "task_id": self.task_id, "scene": self.manifest.get("scene")})

    def _world_cb(self, msg: String) -> None:
        if self.finished or self.world_ready:
            return
        payload = _object(msg.data)
        world = payload.get("world") if isinstance(payload, dict) and isinstance(payload.get("world"), dict) else None
        objects = world.get("objects") if isinstance(world, dict) and isinstance(world.get("objects"), dict) else {}
        request = _task_request(self.stage)
        required = [value for value in (request.object_id, request.target_id) if value]
        if required and not all(str(value) in objects for value in required):
            return
        self.world_ready = True
        self.stage_started = time.monotonic()
        self._log(
            "world_ready",
            {"stage_index": self.stage_index, "task_id": self.task_id, "required_object_ids": required, "world_object_count": len(objects)},
        )

    def _publish_request(self) -> None:
        msg = String()
        msg.data = task_request_to_json(_task_request(self.stage))
        self.request_pub.publish(msg)

    def _decision_cb(self, msg: String) -> None:
        payload = _object(msg.data)
        plan = payload.get("decision_plan") if isinstance(payload, dict) and isinstance(payload.get("decision_plan"), dict) else payload
        if not isinstance(plan, dict) or str(plan.get("task_id") or "") != self.task_id:
            return
        self._log("decision_plan", {"stage_index": self.stage_index, "task_id": self.task_id, "status": plan.get("status"), "actions": len(plan.get("actions") or [])})

    def _executor_event_cb(self, msg: String) -> None:
        payload = _object(msg.data)
        if not isinstance(payload, dict) or str(payload.get("task_id") or "") != self.task_id:
            return
        if payload.get("event") == "plan_terminal":
            status = str(payload.get("status") or "failed")
            self._log("stage_terminal", {"stage_index": self.stage_index, "task_id": self.task_id, "status": status, "reason": payload.get("reason")})
            self._terminal(status, str(payload.get("reason") or ""))

    def _policy_cb(self, msg: String) -> None:
        payload = _object(msg.data)
        action = payload.get("policy_action") if isinstance(payload, dict) else None
        if not isinstance(action, dict):
            return
        task_id = str(action.get("task_id") or "")
        if task_id and task_id != self.task_id:
            return
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        backend = metadata.get("policy_backend") if isinstance(metadata.get("policy_backend"), dict) else {}
        self._log(
            "policy_action",
            {
                "stage_index": self.stage_index,
                "task_id": self.task_id,
                "policy_id": backend.get("policy_id") or metadata.get("policy_id"),
                "policy_type": backend.get("type"),
                "base_goal": action.get("base_goal"),
                "grasp_offsets": action.get("grasp_offsets"),
                "recovery": action.get("recovery_decision"),
                "policy_metadata": backend,
            },
        )

    def _primitive_cb(self, msg: String) -> None:
        payload = _object(msg.data)
        if not isinstance(payload, dict) or str(payload.get("task_id") or "") != self.task_id:
            return
        self._log(
            "primitive_status",
            {
                "stage_index": self.stage_index,
                "task_id": self.task_id,
                "status": payload.get("status"),
                "skill_name": payload.get("skill_name"),
                "detail": payload.get("detail"),
                "metrics": payload.get("metrics"),
                "effect_evidence": payload.get("effect_evidence"),
            },
        )

    def _recovery_cb(self, msg: String) -> None:
        payload = _object(msg.data)
        if isinstance(payload, dict) and str(payload.get("task_id") or "") == self.task_id:
            self._log("recovery_status", {"stage_index": self.stage_index, "task_id": self.task_id, "payload": payload})

    def _terminal(self, status: str, reason: str) -> None:
        success = status == "succeeded"
        if not success:
            self.failed_stages.append(self.task_id)
            if not self.args.continue_on_failure:
                self.finished = True
                self._log("episode_terminal", {"status": "failed", "reason": reason, "completed_stage_count": self.index, "failed_stages": self.failed_stages})
                return
        self.index += 1
        if self.index >= len(self.stages):
            self.finished = True
            self._log("episode_terminal", {"status": "succeeded" if not self.failed_stages else "partial", "completed_stage_count": len(self.stages) - len(self.failed_stages), "failed_stages": self.failed_stages})
            return
        self._start_stage()

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if event == "episode_terminal":
            payload = {
                **payload,
                "episode_scope": self.episode_scope,
                "manifest_stage_count": int(self.manifest.get("stage_count") or len(self.stages)),
                "selected_stage_count": len(self.stages),
            }
        row = {"schema": "sonic_world_model_episode_event_v0", "event": event, "stamp": time.time(), **payload}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), list) else []
    start = max(1, int(args.stage_start))
    stop = int(args.stage_stop) if int(args.stage_stop) > 0 else None
    selected = [stage for stage in stages if isinstance(stage, dict) and int(stage.get("stage_index") or 0) >= start and (stop is None or int(stage.get("stage_index") or 0) <= stop)]
    if not selected:
        raise SystemExit(f"no manifest stages selected by --stage-start={start} --stage-stop={args.stage_stop}")
    manifest = {**manifest, "stages": selected, "selected_stage_range": {"start": start, "stop": stop or max(int(stage.get("stage_index") or 0) for stage in selected)}}
    output = Path(args.output_jsonl).expanduser()
    if output.exists():
        output.unlink()
    rclpy.init()
    node = EpisodeNode(args, manifest, output)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.finished:
            node.tick()
            executor.spin_once(timeout_sec=0.05)
        if not node.finished:
            # A parent stack shutdown used to make this client return success
            # without any terminal evidence, poisoning curriculum statistics.
            node._terminal("failed", "episode_client_shutdown_before_terminal")
            return 2
        return 0 if not node.failed_stages else 1
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _task_request(stage: dict[str, Any]) -> TaskRequest:
    payload = stage.get("request") if isinstance(stage.get("request"), dict) else {}
    request = TaskRequest.from_dict(payload)
    return TaskRequest(verb=request.verb, object_id=request.object_id, object_category=request.object_category, target_id=request.target_id, metadata={**request.metadata, "request_id": str(stage.get("task_id") or "")})


def _object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
