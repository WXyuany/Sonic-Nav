#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.planners import PlanningResult, TaskRequest, WorldModelPipeline, task_request_from_json
from sonic_world.policies import load_policy_backend
from sonic_world.skills import SkillExecutionMonitor
from sonic_world.world_model import WorldMemory


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def _live_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=4,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Sonic demo anchors into a normalized world model and skill graph."
    )
    parser.add_argument("--box-topic", default="/sonic_demo/box_anchor")
    parser.add_argument("--ball-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--object-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--goal-topic", default="/goal_pose")
    parser.add_argument("--world-topic", default="/sonic_world/model")
    parser.add_argument("--skill-topic", default="/sonic_world/skill_graph")
    parser.add_argument("--runtime-topic", default="/sonic_world/runtime_plan")
    parser.add_argument("--dispatch-topic", default="/sonic_world/dispatch_plan")
    parser.add_argument("--recovery-topic", default="/sonic_world/recovery_plan")
    parser.add_argument("--decision-topic", default="/sonic_world/decision_plan")
    parser.add_argument("--policy-topic", default="/sonic_world/policy_action")
    parser.add_argument("--execution-topic", default="/sonic_world/execution_state")
    parser.add_argument("--task-request-topic", default="/sonic_world/task_request")
    parser.add_argument("--active-task-topic", default="/sonic_world/active_task")
    parser.add_argument("--runtime-replan-topic", default="/sonic_world/runtime_replan_request")
    parser.add_argument("--recovery-backend-status-topic", default="/sonic_world/recovery_backend_status")
    parser.add_argument("--phase-topic", default="/sonic_demo/phase")
    parser.add_argument("--dwa-status-topic", default="/sonic_nav/dwa/status")
    parser.add_argument("--mppi-status-topic", default="/sonic_nav/mppi/status")
    parser.add_argument("--nav-metrics-topic", default="/sonic_nav/metrics_summary")
    parser.add_argument("--box-verb", default="pick")
    parser.add_argument("--ball-verb", default="pick_place")
    parser.add_argument("--memory-stale-s", type=float, default=15.0)
    parser.add_argument(
        "--policy-backend",
        choices=["heuristic", "memory", "learned"],
        default="heuristic",
        help="High-level task/skill policy backend used for /sonic_world/policy_action.",
    )
    parser.add_argument(
        "--policy-model",
        help="Path to a trained policy model JSON/JSONL. Required for --policy-backend memory or learned.",
    )
    parser.add_argument("--runtime-override-file", default="", help="Optional JSON mapping of skill names to bounded primitive parameters.")
    return parser.parse_args()


class WorldModelNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_model")
        self.args = args
        self.pipeline = WorldModelPipeline(
            memory=WorldMemory(stale_after_s=float(args.memory_stale_s)),
            box_verb=str(args.box_verb),
            ball_verb=str(args.ball_verb),
        )
        self.policy = load_policy_backend(args.policy_backend, model_path=args.policy_model)
        self.static_runtime_overrides = _load_runtime_overrides(args.runtime_override_file)
        self.active_request: TaskRequest | None = None
        self.active_request_source: str | None = None
        self.execution_monitor = SkillExecutionMonitor()
        self.world_pub = self.create_publisher(String, args.world_topic, _latched_qos())
        self.skill_pub = self.create_publisher(String, args.skill_topic, _latched_qos())
        self.runtime_pub = self.create_publisher(String, args.runtime_topic, _latched_qos())
        self.dispatch_pub = self.create_publisher(String, args.dispatch_topic, _latched_qos())
        self.recovery_pub = self.create_publisher(String, args.recovery_topic, _latched_qos())
        self.decision_pub = self.create_publisher(String, args.decision_topic, _latched_qos())
        self.policy_pub = self.create_publisher(String, args.policy_topic, _latched_qos())
        self.execution_pub = self.create_publisher(String, args.execution_topic, _latched_qos())
        self.active_task_pub = self.create_publisher(String, args.active_task_topic, _latched_qos())
        self.recovery_backend_status_pub = self.create_publisher(String, args.recovery_backend_status_topic, 10)
        self.create_subscription(String, args.box_topic, self._anchor_cb, _live_qos())
        self.create_subscription(String, args.ball_topic, self._anchor_cb, _live_qos())
        self.create_subscription(String, args.object_topic, self._anchor_cb, _live_qos())
        self.create_subscription(String, args.task_request_topic, self._task_request_cb, _live_qos())
        self.create_subscription(String, args.runtime_replan_topic, self._runtime_replan_cb, _live_qos())
        self.create_subscription(String, args.phase_topic, self._phase_cb, _live_qos())
        self.create_subscription(String, args.dwa_status_topic, self._nav_status_cb, _live_qos())
        self.create_subscription(String, args.mppi_status_topic, self._nav_status_cb, _live_qos())
        self.create_subscription(String, args.nav_metrics_topic, self._nav_status_cb, _live_qos())
        self.create_subscription(PoseStamped, args.goal_topic, self._goal_cb, _live_qos())
        self.get_logger().info(
            "World model node listening: "
            f"box={args.box_topic} ball={args.ball_topic} "
            f"object={args.object_topic} goal={args.goal_topic} "
            f"task={args.task_request_topic} phase={args.phase_topic} policy={args.policy_topic} "
            f"policy_backend={getattr(self.policy, 'policy_id', args.policy_backend)}"
        )

    def _anchor_cb(self, msg: String) -> None:
        try:
            anchor = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse anchor JSON: {exc}")
            return
        self._publish_anchor(anchor)

    def _goal_cb(self, msg: PoseStamped) -> None:
        anchor = {
            "goal_name": "rviz_goal",
            "source": "goal_pose",
            "frame_id": msg.header.frame_id or "map",
            "goal_center_map": [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            "goal_yaw": _yaw_from_pose(msg),
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        }
        request = TaskRequest(verb="navigate", object_id="rviz_goal")
        try:
            result = self.pipeline.observe_anchor(
                anchor,
                source="goal_pose",
                request=request,
                kind="task_request",
            )
            self.active_request = request
            self.active_request_source = "goal_pose"
            self._publish_result(result)
        except Exception as exc:
            self.get_logger().warn(f"failed to handle goal pose: {exc}")

    def _task_request_cb(self, msg: String) -> None:
        try:
            request = task_request_from_json(msg.data)
            result = self.pipeline.plan_current(request, kind="task_request", source="task_request")
            self.active_request = request
            self.active_request_source = "task_request"
            self._publish_result(result)
        except Exception as exc:
            self.get_logger().warn(f"failed to handle task request: {exc}")

    def _phase_cb(self, msg: String) -> None:
        state = self.execution_monitor.update_phase(msg.data, event="demo_phase")
        self._publish_execution_state(state)

    def _runtime_replan_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("runtime replan request must be an object")
            if self.active_request is None:
                raise RuntimeError("no active task is available for runtime replanning")
            command = payload.get("command") if isinstance(payload.get("command"), dict) else {}
            failed = command.get("failed_status") if isinstance(command.get("failed_status"), dict) else {}
            recovery_context = {
                "handler": payload.get("handler") or command.get("handler"),
                "reason": payload.get("reason") or command.get("reason") or command.get("type"),
                "failed_skill": failed.get("skill_name") or command.get("failed_skill"),
                "attempt": command.get("attempt", 1),
                "runtime_overrides": command.get("runtime_overrides") if isinstance(command.get("runtime_overrides"), dict) else {},
            }
            self.active_request = TaskRequest(
                verb=self.active_request.verb,
                object_id=self.active_request.object_id,
                object_category=self.active_request.object_category,
                target_id=self.active_request.target_id,
                metadata={**self.active_request.metadata, "runtime_recovery": recovery_context},
            )
            result = self.pipeline.plan_current(
                self.active_request,
                kind="task_request",
                source="runtime_recovery",
            )
            self._publish_result(result)
            self._publish_json(
                self.recovery_backend_status_pub,
                {
                    "event": "recovery_backend_status",
                    "route": "runtime",
                    "status": "completed",
                    "action": "runtime_replan_completed",
                    "task_id": result.skill_graph.task_id,
                    "action_id": payload.get("action_id"),
                    "target_id": payload.get("target_id"),
                },
            )
        except Exception as exc:
            self._publish_json(
                self.recovery_backend_status_pub,
                {
                    "event": "recovery_backend_status",
                    "route": "runtime",
                    "status": "failed",
                    "action": "runtime_replan_failed",
                    "error": str(exc),
                },
            )

    def _nav_status_cb(self, msg: String) -> None:
        state = self.execution_monitor.update_status_text(msg.data, event="nav_status")
        self._publish_execution_state(state)

    def _publish_anchor(self, anchor: dict[str, Any]) -> None:
        try:
            active = self.active_request
            if active is not None and _anchor_matches_request(anchor, active):
                result = self.pipeline.observe_anchor(
                    anchor,
                    source="anchor_replan",
                    request=active,
                    kind="task_request",
                )
            else:
                result = self.pipeline.observe_anchor(anchor, source="anchor")
        except Exception as exc:
            self.get_logger().warn(f"failed to build world model from anchor: {exc}")
            return
        self._publish_result(result)

    def _publish_result(self, result: PlanningResult) -> None:
        try:
            execution_state = self.execution_monitor.set_runtime(result.runtime_plan)
        except Exception as exc:
            self.get_logger().warn(f"failed to set execution runtime: {exc}")
            return
        policy_action = self.policy.act(result).to_dict()
        decision_payload = result.decision_payload()
        recovery_context = result.request.metadata.get("runtime_recovery") if isinstance(result.request.metadata, dict) else {}
        _apply_policy_action_to_decision(
            decision_payload, policy_action, recovery_context=recovery_context, static_overrides=self.static_runtime_overrides
        )
        self._publish_json(self.world_pub, result.world_payload())
        self._publish_json(self.active_task_pub, result.active_task_payload())
        self._publish_json(self.skill_pub, result.skill_payload())
        self._publish_json(self.runtime_pub, result.runtime_payload())
        self._publish_json(self.dispatch_pub, result.dispatch_payload())
        self._publish_json(self.recovery_pub, result.recovery_payload())
        self._publish_json(self.decision_pub, decision_payload)
        self._publish_json(
            self.policy_pub,
            {
                "kind": result.kind,
                "source": result.source,
                "policy_action": policy_action,
            },
        )
        self._publish_execution_state(execution_state, kind=result.kind, source=result.source)

    def _publish_json(self, publisher, payload: dict[str, Any]) -> None:
        out = String()
        out.data = json.dumps(payload, separators=(",", ":"))
        publisher.publish(out)

    def _publish_execution_state(
        self,
        state,
        *,
        kind: str | None = None,
        source: str | None = None,
    ) -> None:
        payload = state.to_dict()
        metadata = dict(payload.get("metadata") or {})
        if kind is not None:
            metadata["kind"] = kind
        if source is not None:
            metadata["source"] = source
        payload["metadata"] = metadata
        self._publish_json(self.execution_pub, payload)


def _apply_policy_action_to_decision(
    decision_payload: dict[str, Any], policy_action: dict[str, Any], *, recovery_context: dict[str, Any] | None = None,
    static_overrides: dict[str, dict[str, float]] | None = None,
) -> None:
    """Apply bounded learned task-space residuals to executable dispatch commands."""
    metadata = policy_action.get("metadata") if isinstance(policy_action.get("metadata"), dict) else {}
    backend = metadata.get("policy_backend") if isinstance(metadata.get("policy_backend"), dict) else {}
    if backend.get("type") not in {"hybrid_ppo", "linear_learned"}:
        return
    base = policy_action.get("base_goal") if isinstance(policy_action.get("base_goal"), dict) else {}
    close = policy_action.get("grasp_close_ratio") if isinstance(policy_action.get("grasp_close_ratio"), dict) else {}
    runtime = backend.get("skill_runtime_overrides") if isinstance(backend.get("skill_runtime_overrides"), dict) else {}
    runtime_mode = str(backend.get("skill_runtime_override_mode") or "absolute")
    recovery = recovery_context if isinstance(recovery_context, dict) else {}
    recovery_runtime = recovery.get("runtime_overrides") if isinstance(recovery.get("runtime_overrides"), dict) else {}
    static = static_overrides if isinstance(static_overrides, dict) else {}
    plan = decision_payload.get("decision_plan", decision_payload)
    for action in plan.get("actions", []):
        if not isinstance(action, dict):
            continue
        command = action.get("command") if isinstance(action.get("command"), dict) else {}
        if str(action.get("handler") or "") == "demo_locomotion_phase_runtime" and isinstance(base.get("position"), list):
            pose = command.get("approach_pose") if isinstance(command.get("approach_pose"), dict) else {}
            pose["frame_id"] = base.get("frame_id") or pose.get("frame_id") or "map"
            pose["position"] = list(base["position"][:3])
            if base.get("yaw") is not None:
                pose["yaw"] = base["yaw"]
            command["approach_pose"] = pose
        skill_name = str(action.get("skill_name") or action.get("source_id") or "")
        if skill_name.startswith("manip."):
            params = command.get("params") if isinstance(command.get("params"), dict) else {}
            if isinstance(close.get("close_ratio"), (int, float)):
                params["close_ratio"] = max(0.18, min(0.92, float(close["close_ratio"])))
            skill_params = dict(static.get(skill_name) or {}) if isinstance(static.get(skill_name), dict) else {}
            learned_params = dict(runtime.get(skill_name) or {}) if isinstance(runtime.get(skill_name), dict) else {}
            if runtime_mode == "residual_additive":
                for key, value in learned_params.items():
                    if not isinstance(value, (int, float)):
                        continue
                    baseline = skill_params.get(str(key), params.get(str(key), 0.0))
                    if isinstance(baseline, (int, float)):
                        skill_params[str(key)] = float(baseline) + float(value)
            else:
                skill_params.update(learned_params)
            if isinstance(recovery_runtime.get(skill_name), dict):
                skill_params.update(recovery_runtime[skill_name])
            for key, value in skill_params.items():
                if isinstance(value, (int, float)):
                    params[str(key)] = float(value)
            command["params"] = params
        action["command"] = command
    next_action = plan.get("next_action") if isinstance(plan.get("next_action"), dict) else None
    if next_action:
        for action in plan.get("actions", []):
            if isinstance(action, dict) and action.get("action_id") == next_action.get("action_id"):
                plan["next_action"] = action
                break
    plan.setdefault("metadata", {})["policy_backend"] = backend


def _load_runtime_overrides(raw: str) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ValueError(f"runtime override file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime override file must be a JSON object")
    result: dict[str, dict[str, float]] = {}
    for skill, params in value.items():
        if not isinstance(params, dict):
            continue
        numeric = {str(key): float(item) for key, item in params.items() if isinstance(item, (int, float))}
        if numeric:
            result[str(skill)] = numeric
    return result


def _yaw_from_pose(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z)),
    )


def _anchor_matches_request(anchor: dict[str, Any], request: TaskRequest) -> bool:
    ids, categories = _anchor_identity_sets(anchor)
    if request.object_id and request.object_id in ids:
        return True
    if request.target_id and request.target_id in ids:
        return True
    if request.object_category and request.object_category in categories:
        return True
    return False


def _anchor_identity_sets(anchor: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    categories: set[str] = set()
    if isinstance(anchor.get("objects"), list):
        records = anchor.get("objects")
    else:
        records = [anchor]
    for record in records:
        if not isinstance(record, dict):
            continue
        object_id = record.get("object_id") or record.get("id") or record.get("name")
        if object_id:
            ids.add(str(object_id))
        category = record.get("category") or record.get("object_category") or record.get("class")
        if category:
            categories.add(str(category))
    if anchor.get("box_name"):
        ids.add(str(anchor.get("box_name")))
        categories.add("box")
    if anchor.get("ball_name"):
        ids.add(str(anchor.get("ball_name")))
        categories.add("ball")
    if anchor.get("goal_name"):
        ids.add(str(anchor.get("goal_name")))
        categories.add("navigation_goal")
    if "place_center_map" in anchor or "place_point_base" in anchor:
        ids.add("place_target")
        categories.add("place_target")
    return ids, categories


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WorldModelNode(args)
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
