#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from typing import Any, Callable

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", os.environ.get("SONIC_WORLD_SMOKE_DOMAIN", "93"))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from world_model_node import WorldModelNode
from world_model_executor import WorldModelExecutor
from world_model_primitive_runner import WorldModelPrimitiveRunner
from world_model_recovery_backends import WorldModelRecoveryBackends
from world_model_recovery_coordinator import WorldModelRecoveryCoordinator
from world_model_preview import SAMPLE_BALL_ANCHOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-simulator ROS smoke test for the Sonic world-model node."
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--topic-prefix", help="Unique ROS topic prefix. Defaults to a random smoke-test namespace.")
    return parser.parse_args()


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
        depth=8,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


class SmokeProbe(Node):
    def __init__(self, node_args: argparse.Namespace):
        super().__init__("sonic_world_smoke_probe")
        self.messages: dict[str, list[dict[str, Any]]] = {
            "world": [],
            "active_task": [],
            "skill": [],
            "runtime": [],
            "dispatch": [],
            "recovery": [],
            "decision": [],
            "execution": [],
            "executor_event": [],
            "primitive_command": [],
            "primitive_status": [],
            "recovery_request": [],
            "recovery_status": [],
            "perception_recovery": [],
            "perception_reobserve": [],
            "recovery_backend_status": [],
            "runtime_replan": [],
        }
        self.object_pub = self.create_publisher(String, node_args.object_topic, _live_qos())
        self.ball_pub = self.create_publisher(String, node_args.ball_topic, _live_qos())
        self.task_pub = self.create_publisher(String, node_args.task_request_topic, _live_qos())
        self.visual_recovery_pub = self.create_publisher(String, node_args.recovery_request_topic, _live_qos())
        self.subscriptions_hold = []
        for key, topic in _output_topics(node_args).items():
            sub = self.create_subscription(String, topic, self._capture(key), _latched_qos())
            self.subscriptions_hold.append(sub)
        self.subscriptions_hold.append(
            self.create_subscription(String, node_args.event_topic, self._capture("executor_event"), _live_qos())
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.runtime_replan_topic,
                self._capture("runtime_replan"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.primitive_command_topic,
                self._capture("primitive_command"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.primitive_status_topic,
                self._capture("primitive_status"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.recovery_request_topic,
                self._capture("recovery_request"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.recovery_status_topic,
                self._capture("recovery_status"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.perception_recovery_topic,
                self._capture("perception_recovery"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.perception_reobserve_command_topic,
                self._capture("perception_reobserve"),
                _live_qos(),
            )
        )
        self.subscriptions_hold.append(
            self.create_subscription(
                String,
                node_args.recovery_backend_status_topic,
                self._capture("recovery_backend_status"),
                _live_qos(),
            )
        )

    def _capture(self, key: str) -> Callable[[String], None]:
        def callback(msg: String) -> None:
            try:
                payload = json.loads(msg.data)
            except Exception as exc:
                payload = {"_parse_error": str(exc), "_raw": msg.data}
            self.messages[key].append(payload)

        return callback


def main() -> None:
    args = parse_args()
    prefix = _topic_prefix(args.topic_prefix)
    node_args = _node_args(prefix)
    rclpy.init(args=None)
    world_node = WorldModelNode(node_args)
    executor_node = WorldModelExecutor(_executor_args(node_args))
    primitive_node = WorldModelPrimitiveRunner(_primitive_args(node_args))
    recovery_node = WorldModelRecoveryCoordinator(_recovery_args(node_args))
    probe = SmokeProbe(node_args)
    recovery_backends_node: WorldModelRecoveryBackends | None = None
    executor = SingleThreadedExecutor()
    executor.add_node(world_node)
    executor.add_node(executor_node)
    executor.add_node(primitive_node)
    executor.add_node(recovery_node)
    executor.add_node(probe)

    try:
        anchor = copy.deepcopy(SAMPLE_BALL_ANCHOR)
        anchor["source"] = "ros_smoke_test"
        anchor["scene"] = "ros_smoke_test"

        _publish_until(
            executor,
            probe.ball_pub,
            anchor,
            lambda: _has_ball_anchor_plan(probe.messages),
            timeout=args.timeout,
            label="ball anchor plan",
        )

        task_request = {
            "task": "move",
            "object": "demo_ball",
            "target": "place_target",
            "id": "ros-smoke-move-ball",
        }
        _publish_until(
            executor,
            probe.task_pub,
            task_request,
            lambda: _has_task_request_plan(probe.messages),
            timeout=args.timeout,
            label="task request plan",
        )
        _publish_until(
            executor,
            probe.task_pub,
            task_request,
            lambda: _latest_executor_decision_action(
                probe.messages,
                source="task_request",
                kind="dispatch",
                action="published_primitive_command",
            ),
            timeout=args.timeout,
            label="task request executor decision",
        )
        _publish_until(
            executor,
            probe.task_pub,
            task_request,
            lambda: _latest_primitive_success(
                probe.messages,
                skill_name="manip.single_hand_pinch",
            ),
            timeout=args.timeout,
            label="task request primitive status",
            debug=lambda: _primitive_debug(probe.messages),
        )
        _assert_payloads(probe.messages)

        _publish_until(
            executor,
            probe.object_pub,
            _generic_anchor(),
            lambda: _latest_world_has_object(probe.messages, "red_fruit")
            and _latest_skill_has_step(probe.messages, "manip.single_hand_pinch", source="anchor"),
            timeout=args.timeout,
            label="generic object anchor plan",
        )

        _publish_until(
            executor,
            probe.object_pub,
            _generic_missing_base_anchor(),
            lambda: _latest_recovery_has_suggestion(
                probe.messages,
                "publish_object_anchor_with_pose_base",
                source="anchor",
            )
            and _latest_decision_is_recovery(
                probe.messages,
                handler="object_anchor_update",
                source="anchor",
            )
            and _latest_executor_decision_action(
                probe.messages,
                source="anchor",
                kind="recovery",
                action="request_recovery",
                handler="object_anchor_update",
            )
            and _latest_recovery_request(
                probe.messages,
                source="anchor",
                handler="object_anchor_update",
                target_id="far_fruit",
                command_type="publish_object_anchor_with_pose_base",
            )
            and _latest_recovery_routed(
                probe.messages,
                route="perception",
                handler="object_anchor_update",
                target_id="far_fruit",
            ),
            timeout=args.timeout,
            label="generic recovery plan",
        )

        far_request = {
            "task": "pick",
            "object": "far_fruit",
            "id": "ros-smoke-pick-far-fruit",
        }
        _publish_until(
            executor,
            probe.task_pub,
            far_request,
            lambda: _latest_decision_is_recovery(
                probe.messages,
                handler="object_anchor_update",
                source="task_request",
            ),
            timeout=args.timeout,
            label="active task recovery decision",
        )
        _publish_until(
            executor,
            probe.object_pub,
            _generic_recovered_base_anchor(),
            lambda: _latest_decision_ready_for_request(
                probe.messages,
                request_id="ros-smoke-pick-far-fruit",
                source="anchor_replan",
            ),
            timeout=args.timeout,
            label="active task anchor replan",
        )

        recovery_backends_node = WorldModelRecoveryBackends(_recovery_backend_args(node_args))
        executor.add_node(recovery_backends_node)
        visual_recovery = {
            "event": "recovery_request",
            "source": "rgbd_anchor_backend",
            "action_id": "visual-reobserve:demo_ball",
            "target_id": "demo_ball",
            "handler": "perception_reobserve",
            "reason": "expected_object_missing_after_rgbd_fusion",
            "command": {"type": "reobserve_from_current_view", "expected_object_id": "demo_ball"},
        }
        _publish_until(
            executor,
            probe.visual_recovery_pub,
            visual_recovery,
            lambda: _latest_reobserve_command(probe.messages, target_id="demo_ball"),
            timeout=args.timeout,
            label="visual perception recovery backend",
        )
        visual_navigation_recovery = {
            "event": "recovery_request",
            "source": "rgbd_anchor_backend",
            "action_id": "visual-micro-adjust:demo_ball",
            "target_id": "demo_ball",
            "handler": "navigation_micro_adjust",
            "reason": "visual_reobserve_exhausted",
            "command": {"type": "micro_adjust_base_for_observation", "max_step_m": 0.02, "speed_mps": 0.10},
        }
        _publish_until(
            executor,
            probe.visual_recovery_pub,
            visual_navigation_recovery,
            lambda: _latest_runtime_replan(probe.messages, target_id="demo_ball"),
            timeout=args.timeout,
            label="visual navigation recovery backend",
        )

        print(f"world_model_ros_smoke_test: ok prefix={prefix}")
    finally:
        executor.remove_node(probe)
        executor.remove_node(recovery_node)
        if recovery_backends_node is not None:
            executor.remove_node(recovery_backends_node)
        executor.remove_node(primitive_node)
        executor.remove_node(executor_node)
        executor.remove_node(world_node)
        probe.destroy_node()
        recovery_node.destroy_node()
        if recovery_backends_node is not None:
            recovery_backends_node.destroy_node()
        primitive_node.close()
        primitive_node.destroy_node()
        executor_node.destroy_node()
        world_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _node_args(prefix: str) -> argparse.Namespace:
    return argparse.Namespace(
        box_topic=f"{prefix}/box_anchor",
        ball_topic=f"{prefix}/ball_anchor",
        object_topic=f"{prefix}/object_anchor",
        goal_topic=f"{prefix}/goal_pose",
        world_topic=f"{prefix}/model",
        skill_topic=f"{prefix}/skill_graph",
        runtime_topic=f"{prefix}/runtime_plan",
        dispatch_topic=f"{prefix}/dispatch_plan",
        recovery_topic=f"{prefix}/recovery_plan",
        decision_topic=f"{prefix}/decision_plan",
        policy_topic=f"{prefix}/policy_action",
        execution_topic=f"{prefix}/execution_state",
        event_topic=f"{prefix}/executor_event",
        primitive_command_topic=f"{prefix}/primitive_command",
        primitive_status_topic=f"{prefix}/primitive_status",
        recovery_request_topic=f"{prefix}/recovery_request",
        recovery_status_topic=f"{prefix}/recovery_status",
        perception_recovery_topic=f"{prefix}/perception_recovery_request",
        perception_reobserve_command_topic=f"{prefix}/perception_reobserve_cmd",
        navigation_recovery_topic=f"{prefix}/navigation_recovery_request",
        runtime_recovery_topic=f"{prefix}/runtime_recovery_request",
        manual_recovery_topic=f"{prefix}/manual_recovery_request",
        task_request_topic=f"{prefix}/task_request",
        active_task_topic=f"{prefix}/active_task",
        runtime_replan_topic=f"{prefix}/runtime_replan_request",
        recovery_backend_status_topic=f"{prefix}/recovery_backend_status",
        phase_topic=f"{prefix}/phase",
        dwa_status_topic=f"{prefix}/dwa_status",
        mppi_status_topic=f"{prefix}/mppi_status",
        nav_metrics_topic=f"{prefix}/nav_metrics",
        box_verb="pick",
        ball_verb="pick_place",
        memory_stale_s=0.0,
        policy_backend="heuristic",
        policy_model=None,
    )


def _executor_args(node_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        decision_topic=node_args.decision_topic,
        dispatch_topic="",
        event_topic=node_args.event_topic,
        recovery_request_topic=node_args.recovery_request_topic,
        primitive_command_topic=node_args.primitive_command_topic,
        primitive_status_topic=node_args.primitive_status_topic,
        navigation_status_topic=f"{node_args.goal_topic}_status",
        primitive_timeout_s=45.0,
        max_recovery_attempts=2,
        require_effect_evidence=True,
        goal_topic=node_args.goal_topic,
        execute_navigation=False,
        execute_anchor_plans=False,
    )


def _primitive_args(node_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        primitive_command_topic=node_args.primitive_command_topic,
        primitive_status_topic=node_args.primitive_status_topic,
        ball_anchor_topic=node_args.ball_topic,
        box_anchor_topic=node_args.box_topic,
        object_anchor_topic=node_args.object_topic,
        backend="contract_test",
        demo_kind="auto",
        scene=None,
        qpos_path="/tmp/sonic_qpos.npy",
        qpos_meta_path="/tmp/sonic_qpos_meta.json",
        effect_observer="none",
        rate=30.0,
        warmup=0.0,
        zmq_bind_host="*",
        zmq_port=5556,
        start_bursts=1,
        max_upper_body_velocity=1.8,
        apply_anchor=False,
        demo_arg=[],
    )


def _recovery_args(node_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        recovery_request_topic=node_args.recovery_request_topic,
        status_topic=node_args.recovery_status_topic,
        perception_topic=node_args.perception_recovery_topic,
        navigation_topic=node_args.navigation_recovery_topic,
        runtime_topic=node_args.runtime_recovery_topic,
        manual_topic=node_args.manual_recovery_topic,
        dedupe_window_s=1.0,
    )


def _recovery_backend_args(node_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        world_topic=node_args.world_topic,
        object_anchor_topic=node_args.object_topic,
        perception_reobserve_command_topic=node_args.perception_reobserve_command_topic,
        status_topic=node_args.recovery_backend_status_topic,
        perception_topic=node_args.perception_recovery_topic,
        navigation_topic=node_args.navigation_recovery_topic,
        runtime_topic=node_args.runtime_recovery_topic,
        manual_topic=node_args.manual_recovery_topic,
        navigation_command_topic=f"{node_args.goal_topic}_recovery_cmd",
        cmd_vel_topic=f"{node_args.goal_topic}_cmd_vel",
        runtime_replan_topic=node_args.runtime_replan_topic,
        default_place_offset_y=0.28,
    )


def _output_topics(node_args: argparse.Namespace) -> dict[str, str]:
    return {
        "world": node_args.world_topic,
        "active_task": node_args.active_task_topic,
        "skill": node_args.skill_topic,
        "runtime": node_args.runtime_topic,
        "dispatch": node_args.dispatch_topic,
        "recovery": node_args.recovery_topic,
        "decision": node_args.decision_topic,
        "execution": node_args.execution_topic,
    }


def _generic_anchor() -> dict[str, Any]:
    return {
        "scene": "ros_smoke_test",
        "source": "ros_smoke_test",
        "frame_id": "map",
        "objects": [
            {
                "object_id": "red_fruit",
                "category": "fruit",
                "shape": {"kind": "sphere", "radius": 0.04},
                "pose_map": {"frame_id": "map", "position": [1.5, -0.22, 0.84]},
                "pose_base": {"frame_id": "base_link", "position": [0.5, -0.22, 0.03]},
                "support": "table",
                "grasp": {
                    "approach_target_x": 0.54,
                    "target_y": -0.22,
                    "reach_x": 0.5,
                    "reach_z": 0.03,
                    "base_target_map": [1.0, -0.16, 0.0],
                },
            },
            {
                "object_id": "right_tray",
                "category": "place_target",
                "shape": "target",
                "pose_map": {"frame_id": "map", "position": [1.5, 0.1, 0.84]},
                "pose_base": {"frame_id": "base_link", "position": [0.5, 0.08, 0.03]},
                "support": "table",
            },
        ],
    }


def _generic_missing_base_anchor() -> dict[str, Any]:
    return {
        "scene": "ros_smoke_test",
        "source": "ros_smoke_test",
        "frame_id": "map",
        "objects": [
            {
                "object_id": "far_fruit",
                "category": "fruit",
                "shape": {"kind": "sphere", "radius": 0.04},
                "pose_map": {"frame_id": "map", "position": [2.2, 0.2, 0.84]},
                "support": "table",
                "grasp": {
                    "approach_target_x": 0.55,
                    "target_y": 0.2,
                    "reach_x": 0.5,
                    "reach_z": 0.03,
                    "base_target_map": [1.65, 0.16, 0.0],
                },
            },
        ],
    }


def _generic_recovered_base_anchor() -> dict[str, Any]:
    anchor = _generic_missing_base_anchor()
    item = anchor["objects"][0]
    item["pose_base"] = {"frame_id": "base_link", "position": [0.50, 0.18, 0.03]}
    item["grasp"]["target_y"] = 0.18
    return anchor


def _publish_until(
    executor: SingleThreadedExecutor,
    publisher,
    payload: dict[str, Any],
    predicate: Callable[[], bool],
    *,
    timeout: float,
    label: str,
    debug: Callable[[], str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    next_publish = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_publish:
            msg = String()
            msg.data = json.dumps(payload, separators=(",", ":"))
            publisher.publish(msg)
            next_publish = now + 0.2
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return
    suffix = f": {debug()}" if debug is not None else ""
    raise RuntimeError(f"timed out waiting for {label}{suffix}")


def _has_ball_anchor_plan(messages: dict[str, list[dict[str, Any]]]) -> bool:
    return (
        _latest_world_has_object(messages, "demo_ball")
        and _latest_skill_has_step(messages, "manip.single_hand_pinch")
        and _latest_runtime_ready(messages, source="anchor")
        and _latest_dispatch_has_handler(messages, "contact_grasp_primitive")
    )


def _has_task_request_plan(messages: dict[str, list[dict[str, Any]]]) -> bool:
    active = _latest_by_source(messages["active_task"], "task_request")
    return (
        active is not None
        and active.get("request", {}).get("verb") == "move"
        and _latest_skill_has_step(messages, "manip.single_hand_pinch", source="task_request")
        and _latest_dispatch_has_handler(messages, "contact_grasp_primitive", source="task_request")
        and _latest_execution_planned(messages, source="task_request")
    )


def _assert_payloads(messages: dict[str, list[dict[str, Any]]]) -> None:
    task_skill = _latest_by_source(messages["skill"], "task_request")
    task_runtime = _latest_by_source(messages["runtime"], "task_request")
    task_dispatch = _latest_by_source(messages["dispatch"], "task_request")
    task_decision = _latest_by_source(messages["decision"], "task_request")
    task_execution = _latest_execution(messages, source="task_request")
    if task_skill is None:
        raise AssertionError("missing task_request skill graph")
    steps = [step.get("name") for step in task_skill.get("skill_graph", {}).get("steps", [])]
    expected = [
        "navigate.approach_object",
        "manip.align_workspace",
        "manip.single_hand_pinch",
        "manip.lift_object",
        "manip.transport_object",
        "manip.place_object",
        "manip.release",
    ]
    if steps != expected:
        raise AssertionError(f"unexpected task skill order: {steps}")
    if task_runtime is None or task_runtime.get("runtime_plan", {}).get("metadata", {}).get("missing_skills") != []:
        raise AssertionError("runtime plan has missing skills")
    unready = task_dispatch.get("dispatch_plan", {}).get("metadata", {}).get("unready_count") if task_dispatch else None
    if unready != 0:
        raise AssertionError(f"dispatch plan not ready: unready_count={unready}")
    if task_decision is None:
        raise AssertionError("missing task_request decision plan")
    task_decision_plan = task_decision.get("decision_plan", {})
    if task_decision_plan.get("status") != "ready_to_execute":
        raise AssertionError(f"decision plan is not ready: {task_decision_plan}")
    next_action = task_decision_plan.get("next_action") or {}
    if next_action.get("kind") != "dispatch":
        raise AssertionError(f"unexpected task next action: {next_action}")
    if task_execution is None or task_execution.get("status") != "planned":
        raise AssertionError(f"execution state is not planned: {task_execution}")


def _latest_world_has_object(messages: dict[str, list[dict[str, Any]]], object_id: str) -> bool:
    for payload in reversed(messages["world"]):
        objects = payload.get("world", {}).get("objects", {})
        if object_id in objects:
            return True
    return False


def _latest_skill_has_step(
    messages: dict[str, list[dict[str, Any]]],
    step_name: str,
    *,
    source: str | None = None,
) -> bool:
    payload = _latest_by_source(messages["skill"], source) if source else _latest(messages["skill"])
    if payload is None:
        return False
    steps = payload.get("skill_graph", {}).get("steps", [])
    return any(step.get("name") == step_name for step in steps)


def _latest_runtime_ready(messages: dict[str, list[dict[str, Any]]], *, source: str | None = None) -> bool:
    payload = _latest_by_source(messages["runtime"], source) if source else _latest(messages["runtime"])
    if payload is None:
        return False
    metadata = payload.get("runtime_plan", {}).get("metadata", {})
    return metadata.get("missing_skills") == []


def _latest_dispatch_has_handler(
    messages: dict[str, list[dict[str, Any]]],
    handler: str,
    *,
    source: str | None = None,
) -> bool:
    payload = _latest_by_source(messages["dispatch"], source) if source else _latest(messages["dispatch"])
    if payload is None:
        return False
    steps = payload.get("dispatch_plan", {}).get("steps", [])
    return any(step.get("handler") == handler for step in steps)


def _latest_execution_planned(messages: dict[str, list[dict[str, Any]]], *, source: str | None = None) -> bool:
    payload = _latest_execution(messages, source=source)
    return bool(payload and payload.get("status") == "planned")


def _latest_recovery_has_suggestion(
    messages: dict[str, list[dict[str, Any]]],
    suggestion: str,
    *,
    source: str | None = None,
) -> bool:
    payload = _latest_by_source(messages["recovery"], source) if source else _latest(messages["recovery"])
    if payload is None:
        return False
    plan = payload.get("recovery_plan", {})
    suggestions = plan.get("metadata", {}).get("suggestions", [])
    return plan.get("status") == "needs_recovery" and suggestion in suggestions


def _latest_decision_is_recovery(
    messages: dict[str, list[dict[str, Any]]],
    *,
    handler: str,
    source: str | None = None,
) -> bool:
    payload = _latest_by_source(messages["decision"], source) if source else _latest(messages["decision"])
    if payload is None:
        return False
    plan = payload.get("decision_plan", {})
    action = plan.get("next_action") or {}
    return (
        plan.get("status") == "needs_recovery"
        and action.get("kind") == "recovery"
        and action.get("handler") == handler
    )


def _latest_decision_ready_for_request(
    messages: dict[str, list[dict[str, Any]]],
    *,
    request_id: str,
    source: str,
) -> bool:
    payload = _latest_by_source(messages["decision"], source)
    if payload is None:
        return False
    request = _latest_by_source(messages["active_task"], source)
    if request is None:
        return False
    metadata = request.get("request", {}).get("metadata", {})
    plan = payload.get("decision_plan", {})
    action = plan.get("next_action") or {}
    return (
        metadata.get("request_id") == request_id
        and plan.get("status") == "ready_to_execute"
        and action.get("kind") == "dispatch"
    )


def _latest_executor_decision_action(
    messages: dict[str, list[dict[str, Any]]],
    *,
    source: str,
    kind: str,
    action: str,
    handler: str | None = None,
) -> bool:
    for payload in reversed(messages["executor_event"]):
        if payload.get("event") != "decision_action":
            continue
        if payload.get("source") != source:
            continue
        if payload.get("kind") != kind or payload.get("action") != action:
            continue
        if handler is not None and payload.get("handler") != handler:
            continue
        return True
    return False


def _latest_primitive_success(
    messages: dict[str, list[dict[str, Any]]],
    *,
    skill_name: str,
) -> bool:
    for payload in reversed(messages["primitive_status"]):
        if payload.get("event") != "primitive_status":
            continue
        if payload.get("status") != "success":
            continue
        if payload.get("skill_name") == skill_name:
            return True
    return False


def _primitive_debug(messages: dict[str, list[dict[str, Any]]]) -> str:
    commands = messages.get("primitive_command", [])
    statuses = messages.get("primitive_status", [])
    tail = {
        "command_count": len(commands),
        "status_count": len(statuses),
        "last_command": commands[-1] if commands else None,
        "last_status": statuses[-1] if statuses else None,
    }
    return json.dumps(tail, sort_keys=True)


def _latest_recovery_request(
    messages: dict[str, list[dict[str, Any]]],
    *,
    source: str,
    handler: str,
    target_id: str,
    command_type: str,
) -> bool:
    for payload in reversed(messages["recovery_request"]):
        if payload.get("event") != "recovery_request":
            continue
        if payload.get("source") != source:
            continue
        if payload.get("handler") != handler or payload.get("target_id") != target_id:
            continue
        command = payload.get("command") if isinstance(payload.get("command"), dict) else {}
        if command.get("type") != command_type:
            continue
        return True
    return False


def _latest_recovery_routed(
    messages: dict[str, list[dict[str, Any]]],
    *,
    route: str,
    handler: str,
    target_id: str,
) -> bool:
    status_ok = False
    for payload in reversed(messages["recovery_status"]):
        if payload.get("event") != "recovery_status":
            continue
        if payload.get("status") != "routed" or payload.get("route") != route:
            continue
        if payload.get("handler") == handler and payload.get("target_id") == target_id:
            status_ok = True
            break
    if not status_ok:
        return False
    route_key = f"{route}_recovery"
    for payload in reversed(messages[route_key]):
        if payload.get("event") != "recovery_routed":
            continue
        if payload.get("route") != route:
            continue
        if payload.get("handler") == handler and payload.get("target_id") == target_id:
            return True
    return False


def _latest_reobserve_command(messages: dict[str, list[dict[str, Any]]], *, target_id: str) -> bool:
    for payload in reversed(messages["perception_reobserve"]):
        if payload.get("schema") != "sonic_perception_reobserve_command_v0":
            continue
        if payload.get("target_id") == target_id:
            return True
    return False


def _latest_runtime_replan(messages: dict[str, list[dict[str, Any]]], *, target_id: str) -> bool:
    for payload in reversed(messages["runtime_replan"]):
        if payload.get("event") != "runtime_replan_request":
            continue
        if payload.get("target_id") == target_id and payload.get("reason") == "navigation_micro_adjust_completed":
            return True
    return False


def _latest_execution(
    messages: dict[str, list[dict[str, Any]]],
    *,
    source: str | None = None,
) -> dict[str, Any] | None:
    if source is None:
        return _latest(messages["execution"])
    for payload in reversed(messages["execution"]):
        if payload.get("metadata", {}).get("source") == source:
            return payload
    return None


def _latest_by_source(messages: list[dict[str, Any]], source: str | None) -> dict[str, Any] | None:
    if source is None:
        return _latest(messages)
    for payload in reversed(messages):
        if payload.get("source") == source:
            return payload
    return None


def _latest(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    return messages[-1] if messages else None


def _topic_prefix(raw: str | None) -> str:
    if raw:
        stripped = raw.strip()
    else:
        stripped = f"sonic_world_smoke_{uuid.uuid4().hex[:8]}"
    return "/" + stripped.strip("/")


if __name__ == "__main__":
    main()
