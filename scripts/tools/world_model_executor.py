#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.skills import DispatchExecutionSession, SkillRuntimeExecutor, verify_backend_effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume /sonic_world/decision_plan and publish executor events."
    )
    parser.add_argument("--decision-topic", default="/sonic_world/decision_plan")
    parser.add_argument(
        "--dispatch-topic",
        default="",
        help="Optional legacy dispatch-plan topic to observe every dispatch step.",
    )
    parser.add_argument("--event-topic", default="/sonic_world/executor_event")
    parser.add_argument("--recovery-request-topic", default="/sonic_world/recovery_request")
    parser.add_argument("--primitive-command-topic", default="/sonic_world/primitive_command")
    parser.add_argument("--primitive-status-topic", default="/sonic_world/primitive_status")
    parser.add_argument("--navigation-status-topic", default="/sonic_world/navigation_status")
    parser.add_argument("--primitive-timeout-s", type=float, default=45.0)
    parser.add_argument("--max-recovery-attempts", type=int, default=3)
    parser.add_argument(
        "--require-effect-evidence",
        action="store_true",
        help="Reject terminal primitive success unless the backend supplies effect_evidence.",
    )
    parser.add_argument("--goal-topic", default="/goal_pose")
    parser.add_argument(
        "--execute-anchor-plans",
        action="store_true",
        help="Allow autonomous execution of plans created only by passive anchor updates.",
    )
    parser.add_argument(
        "--execute-navigation",
        action="store_true",
        help="Actually publish ros2_goal_pose commands from task_request-sourced decision actions.",
    )
    return parser.parse_args()


class WorldModelExecutor(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_executor")
        self.args = args
        self.event_pub = self.create_publisher(String, args.event_topic, 10)
        self.recovery_request_pub = self.create_publisher(String, args.recovery_request_topic, 10)
        self.primitive_command_pub = self.create_publisher(String, args.primitive_command_topic, 10)
        self.goal_pub = self.create_publisher(PoseStamped, args.goal_topic, 10)
        self.pending_primitives: dict[str, dict[str, Any]] = {}
        self.recovery_attempts: dict[tuple[str, str], int] = {}
        self.active_plan: dict[str, Any] | None = None
        self.active_source = ""
        self.execution = DispatchExecutionSession(
            timeout_s=float(args.primitive_timeout_s),
            require_effect_evidence=bool(args.require_effect_evidence),
            verifier=verify_backend_effects if args.require_effect_evidence else None,
        )
        self.runtime_executor = SkillRuntimeExecutor(
            publish_goal=self._publish_goal,
            publish_primitive=self._publish_primitive_command,
            execute_navigation=bool(args.execute_navigation),
        )
        self.create_subscription(String, args.decision_topic, self._decision_cb, 10)
        self.create_subscription(String, args.primitive_status_topic, self._primitive_status_cb, 10)
        self.create_subscription(String, args.navigation_status_topic, self._navigation_status_cb, 10)
        if args.dispatch_topic:
            self.create_subscription(String, args.dispatch_topic, self._dispatch_cb, 10)
        self.create_timer(0.5, self._check_primitive_timeouts)
        mode = "navigation execution enabled" if args.execute_navigation else "dry-run"
        legacy = f", legacy_dispatch={args.dispatch_topic}" if args.dispatch_topic else ""
        self.get_logger().info(
            f"World-model executor listening on decision={args.decision_topic}{legacy} "
            f"recovery_request={args.recovery_request_topic} "
            f"primitive_command={args.primitive_command_topic} "
            f"primitive_status={args.primitive_status_topic} ({mode})"
        )

    def _decision_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            plan = payload.get("decision_plan") or payload
            if not isinstance(plan, dict):
                raise ValueError("decision_plan payload must be an object")
            source = str(payload.get("source", plan.get("metadata", {}).get("source", "")))
            self._handle_decision(plan, source)
        except Exception as exc:
            self._publish_event("error", {"error": str(exc), "raw": msg.data[:500]})

    def _handle_decision(self, plan: dict[str, Any], source: str) -> None:
        action = plan.get("next_action")
        if not isinstance(action, dict):
            self._publish_event(
                "decision_action",
                {
                    "task_id": plan.get("task_id"),
                    "objective": plan.get("objective"),
                    "status": plan.get("status"),
                    "source": source,
                    "action": "blocked",
                    "reason": plan.get("metadata", {}).get("reason", "no next action"),
                },
            )
            return

        allowed_sources = {"task_request", "runtime_recovery", "goal_pose"}
        passive_safe_recovery = action.get("kind") == "recovery" and action.get("handler") in {
            "object_anchor_update",
            "perception_reobserve",
            "world_memory_update",
            "affordance_repair",
            "support_surface_inference",
            "place_target_recovery",
        }
        if source not in allowed_sources and not passive_safe_recovery and not self.args.execute_anchor_plans:
            if source == "anchor_replan" and action.get("kind") == "dispatch":
                return
            self._publish_event(
                "decision_action",
                {
                    "task_id": plan.get("task_id"),
                    "status": plan.get("status"),
                    "source": source,
                    "action": "ignored_non_authoritative_plan",
                    "reason": "passive anchor plans require --execute-anchor-plans",
                },
            )
            return

        event, metadata = self._decision_event(plan, action, source)

        if action.get("kind") == "recovery":
            event["action"] = "request_recovery"
            event["command"] = action.get("command") or {}
            event["request_topic"] = self.args.recovery_request_topic
            self._publish_recovery_request(plan, action, source, metadata)
            self._publish_event("decision_action", event)
            return

        if action.get("kind") != "dispatch":
            event["action"] = "skip_unknown_decision_kind"
            self._publish_event("decision_action", event)
            return

        if self.active_plan is not None:
            active_task_id = str(self.active_plan.get("task_id") or "")
            if active_task_id == str(plan.get("task_id") or ""):
                if source == "runtime_recovery":
                    if self.execution.status != "failed":
                        event["action"] = "ignored_nonterminal_runtime_recovery"
                        event["reason"] = (
                            "runtime recovery replacement is accepted only after the active dispatch action failed"
                        )
                        self._publish_event("decision_action", event)
                        return
                    failed_action_id = self.execution.failed_action_id
                    transition = self.execution.release_for_replan(
                        "runtime recovery supplied a replacement dispatch plan"
                    )
                    self._publish_event("execution_transition", transition.to_dict())
                    self._release_active_plan(active_task_id)
                    if failed_action_id:
                        plan = _resume_plan_from_action(plan, failed_action_id)
                else:
                    event["action"] = "ignored_active_plan_replan"
                    event["reason"] = "active dispatch plan owns execution until terminal status"
                    self._publish_event("decision_action", event)
                    return

        transition = self.execution.load(plan)
        self._publish_event("execution_transition", transition.to_dict())
        if transition.kind == "duplicate_plan":
            return
        if transition.kind in {"busy", "invalid_plan"}:
            return
        self.active_plan = plan
        self.active_source = source
        self._dispatch_next_action()

    def _dispatch_next_action(self) -> None:
        plan = self.active_plan
        action = self.execution.next_action()
        if plan is None or action is None:
            return
        dispatch_event, _metadata = self._decision_event(plan, action, self.active_source)
        result = self.runtime_executor.execute_decision_action(plan, action, source=self.active_source)
        dispatch_event["action"] = result.action
        dispatch_event["runtime_status"] = result.status
        dispatch_event["runtime_metrics"] = result.metrics
        dispatch_event["command"] = result.command
        dispatch_event["phase_names"] = result.metadata.get("phase_names", [])
        dispatch_event["sequence_index"] = self.execution.cursor + 1
        dispatch_event["sequence_count"] = len(self.execution.actions)
        self._publish_event("decision_action", dispatch_event)
        action_id = str(action.get("action_id") or "")
        if result.status == "queued":
            transition = self.execution.mark_dispatched(action_id)
            self._publish_event("execution_transition", transition.to_dict())
            return
        if result.status == "success":
            transition = self.execution.mark_dispatched(action_id)
            self._publish_event("execution_transition", transition.to_dict())
            self._handle_runtime_status(
                {
                    "action_id": action_id,
                    "status": "success",
                    "metrics": result.metrics,
                    "effect_evidence": {"passed": True, "reason": "synchronous handler completed"},
                }
            )
            return
        transition = self.execution.mark_dispatched(action_id)
        self._publish_event("execution_transition", transition.to_dict())
        self._handle_runtime_status(
            {
                "action_id": action_id,
                "status": "failed",
                "detail": result.reason or result.action,
                "metrics": result.metrics,
            }
        )

    def _decision_event(
        self,
        plan: dict[str, Any],
        action: dict[str, Any],
        source: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        event = {
            "task_id": plan.get("task_id"),
            "objective": plan.get("objective"),
            "status": plan.get("status"),
            "action_id": action.get("action_id"),
            "kind": action.get("kind"),
            "handler": action.get("handler"),
            "target_id": action.get("target_id"),
            "source_id": action.get("source_id"),
            "source": source,
            "reason": action.get("reason"),
            "action": "observed",
        }
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        if metadata:
            event["metadata"] = metadata
        return event, metadata

    def _dispatch_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            plan = payload.get("dispatch_plan") or payload
            if not isinstance(plan, dict):
                raise ValueError("dispatch_plan payload must be an object")
            source = str(payload.get("source", plan.get("metadata", {}).get("source", "")))
            for idx, step in enumerate(plan.get("steps", [])):
                self._handle_step(plan, step, idx, source)
        except Exception as exc:
            self._publish_event("error", {"error": str(exc), "raw": msg.data[:500]})

    def _handle_step(self, plan: dict[str, Any], step: dict[str, Any], index: int, source: str) -> None:
        event = {
            "task_id": plan.get("task_id"),
            "objective": plan.get("objective"),
            "index": index,
            "skill_name": step.get("skill_name"),
            "handler": step.get("handler"),
            "capability": step.get("capability"),
            "readiness": step.get("readiness"),
            "source": source,
            "action": "observed",
        }
        contract = step.get("contract") if isinstance(step.get("contract"), dict) else {}
        if contract:
            event["contract_ready"] = contract.get("ready")
            event["contract_failed_errors"] = contract.get("failed_errors", [])
            event["contract_failed_warnings"] = contract.get("failed_warnings", [])
            event["recovery_suggestions"] = contract.get("recovery_suggestions", [])
        if step.get("readiness") != "ready":
            event["action"] = "skip_unready"
            event["notes"] = step.get("notes", [])
            self._publish_event("dispatch_step", event)
            return

        result = self.runtime_executor.execute_dispatch_step(plan, step, index=index, source=source)
        event["action"] = result.action
        event["runtime_status"] = result.status
        event["runtime_metrics"] = result.metrics
        event["command"] = result.command
        event["phase_names"] = result.metadata.get("phase_names", [])
        self._publish_event("dispatch_step", event)

    def _publish_goal(self, pose_payload: dict[str, Any] | None) -> bool:
        if not isinstance(pose_payload, dict):
            return False
        position = pose_payload.get("position")
        if not isinstance(position, list) or len(position) < 2:
            return False
        msg = PoseStamped()
        msg.header.frame_id = str(pose_payload.get("frame_id", "map"))
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2]) if len(position) > 2 else 0.0
        yaw = pose_payload.get("yaw")
        if yaw is None:
            msg.pose.orientation.w = 1.0
        else:
            y = float(yaw)
            msg.pose.orientation.w = math.cos(y * 0.5)
            msg.pose.orientation.z = math.sin(y * 0.5)
        self.goal_pub.publish(msg)
        return True

    def _publish_event(self, event_type: str, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps({"event": event_type, **payload}, separators=(",", ":"))
        self.event_pub.publish(msg)

    def _publish_primitive_command(self, payload: dict[str, Any]) -> bool:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.primitive_command_pub.publish(msg)
        action_id = str(payload.get("action_id") or "")
        if action_id:
            self.pending_primitives[action_id] = {
                "command": payload,
                "sent_monotonic": time.monotonic(),
                "recovery_requested": False,
            }
        return True

    def _primitive_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("primitive status payload must be an object")
            if payload.get("event") != "primitive_status":
                return
            self._handle_primitive_status(payload)
        except Exception as exc:
            self._publish_event("error", {"error": str(exc), "raw": msg.data[:500]})

    def _handle_primitive_status(self, status: dict[str, Any]) -> None:
        action_id = str(status.get("action_id") or "")
        pending = self.pending_primitives.get(action_id) if action_id else None
        elapsed = None
        if pending is not None:
            elapsed = time.monotonic() - float(pending.get("sent_monotonic") or time.monotonic())
        state = str(status.get("status") or "")
        event = {
            "task_id": status.get("task_id"),
            "action_id": action_id,
            "skill_name": status.get("skill_name"),
            "target_id": status.get("target_id"),
            "demo_kind": status.get("demo_kind"),
            "handler": status.get("handler"),
            "capability": status.get("capability"),
            "runtime_status": state,
            "backend": status.get("backend"),
            "phases": status.get("phases") or [],
            "metrics": status.get("metrics") or {},
            "detail": status.get("detail") or "",
            "elapsed_since_command_s": round(float(elapsed), 4) if elapsed is not None else None,
            "action": "primitive_status_observed",
        }
        self._publish_event("primitive_status", event)
        self._handle_runtime_status(status, pending=pending)

    def _navigation_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("navigation status payload must be an object")
            self._handle_runtime_status(payload)
        except Exception as exc:
            self._publish_event("error", {"error": str(exc), "raw": msg.data[:500]})

    def _handle_runtime_status(
        self,
        status: dict[str, Any],
        *,
        pending: dict[str, Any] | None = None,
    ) -> None:
        transition = self.execution.observe_status(status)
        self._publish_event("execution_transition", transition.to_dict())
        if transition.kind in {"action_feedback", "stale_status", "duplicate_status", "invalid_status"}:
            return
        action_id = str(status.get("action_id") or "")
        if action_id:
            self.pending_primitives.pop(action_id, None)
        if transition.kind == "action_succeeded":
            self._dispatch_next_action()
            return
        if transition.kind == "plan_succeeded":
            task_id = str(self.execution.task_id or "")
            self.recovery_attempts = {
                key: value for key, value in self.recovery_attempts.items() if key[0] != task_id
            }
            self._publish_event("plan_terminal", self.execution.to_dict())
            self._release_active_plan(task_id)
            return
        if transition.kind == "action_failed":
            self._request_primitive_recovery(
                status,
                pending=pending,
                reason=transition.reason or f"primitive_{transition.status}",
            )

    def _check_primitive_timeouts(self) -> None:
        transition = self.execution.check_timeout()
        if transition is None:
            return
        action_id = str(transition.action_id or "")
        pending = self.pending_primitives.get(action_id)
        command = pending.get("command") if isinstance(pending, dict) and isinstance(pending.get("command"), dict) else {}
        status = {
            "task_id": command.get("task_id"),
            "action_id": action_id,
            "skill_name": command.get("skill_name"),
            "target_id": command.get("target_id"),
            "demo_kind": command.get("demo_kind"),
            "handler": command.get("handler"),
            "capability": command.get("capability"),
            "status": "timeout",
            "backend": "executor_timeout",
            "detail": transition.reason,
            "metrics": transition.metrics,
        }
        self._publish_event("execution_transition", transition.to_dict())
        self._publish_event("primitive_status", {**status, "runtime_status": "timeout", "action": "primitive_timeout"})
        self._request_primitive_recovery(status, pending=pending, reason="primitive_timeout")
        self.pending_primitives.pop(action_id, None)

    def _request_primitive_recovery(
        self,
        status: dict[str, Any],
        *,
        pending: dict[str, Any] | None,
        reason: str,
    ) -> None:
        command = pending.get("command") if isinstance(pending, dict) and isinstance(pending.get("command"), dict) else {}
        skill_name = str(status.get("skill_name") or command.get("skill_name") or "")
        task_id = str(status.get("task_id") or command.get("task_id") or "")
        attempt_key = (task_id, skill_name)
        attempt = self.recovery_attempts.get(attempt_key, 0) + 1
        self.recovery_attempts[attempt_key] = attempt
        handler, recovery_type = _failure_recovery_route(status, command, reason)
        if attempt > max(0, int(self.args.max_recovery_attempts)):
            handler, recovery_type = "manual_review", "recovery_attempts_exhausted"
        payload = {
            "event": "recovery_request",
            "task_id": task_id,
            "objective": command.get("objective"),
            "status": "needs_recovery",
            "source": "primitive_status",
            "action_id": status.get("action_id") or command.get("action_id"),
            "kind": "recovery",
            "handler": handler,
            "target_id": status.get("target_id") or command.get("target_id"),
            "source_id": skill_name,
            "reason": reason,
            "command": {
                "type": recovery_type,
                "failed_status": status,
                "failed_command": command,
                "attempt": attempt,
                "runtime_overrides": _primitive_recovery_overrides(skill_name, reason, attempt),
            },
            "metadata": {
                "affected_skills": [status.get("skill_name") or command.get("skill_name")],
                "failed_checks": [reason],
                "suggestion": recovery_type,
                "attempt": attempt,
                "max_attempts": int(self.args.max_recovery_attempts),
            },
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.recovery_request_pub.publish(msg)
        self._publish_event(
            "recovery_request",
            {
                "action": "request_recovery_from_primitive_status",
                "task_id": payload["task_id"],
                "action_id": payload["action_id"],
                "handler": payload["handler"],
                "target_id": payload["target_id"],
                "reason": reason,
            },
        )
        if recovery_type == "recovery_attempts_exhausted":
            terminal = self.execution.to_dict()
            terminal.update(
                {
                    "status": "failed",
                    "reason": f"recovery attempts exhausted for {skill_name}: {reason}",
                    "recovery_attempt": attempt,
                    "max_recovery_attempts": int(self.args.max_recovery_attempts),
                }
            )
            self._publish_event("plan_terminal", terminal)
            self._release_active_plan(task_id)

    def _release_active_plan(self, task_id: str) -> None:
        """Forget stale primitive state once a plan is terminal or replaced."""
        self.pending_primitives = {
            action_id: pending
            for action_id, pending in self.pending_primitives.items()
            if str(
                (pending.get("command") if isinstance(pending, dict) else {}).get("task_id") or ""
            )
            != task_id
        }
        self.active_plan = None
        self.active_source = ""

    def _publish_recovery_request(
        self,
        plan: dict[str, Any],
        action: dict[str, Any],
        source: str,
        metadata: dict[str, Any],
    ) -> None:
        payload = {
            "event": "recovery_request",
            "task_id": plan.get("task_id"),
            "objective": plan.get("objective"),
            "status": plan.get("status"),
            "source": source,
            "action_id": action.get("action_id"),
            "kind": action.get("kind"),
            "handler": action.get("handler"),
            "target_id": action.get("target_id"),
            "source_id": action.get("source_id"),
            "reason": action.get("reason"),
            "command": action.get("command") or {},
            "metadata": metadata,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.recovery_request_pub.publish(msg)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WorldModelExecutor(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _failure_recovery_route(
    status: dict[str, Any],
    command: dict[str, Any],
    reason: str,
) -> tuple[str, str]:
    skill = str(status.get("skill_name") or command.get("skill_name") or "")
    text = f"{reason} {status.get('detail') or ''}".lower()
    if "anchor" in text or "pose" in text or "target" in text and "missing" in text:
        return "perception_reobserve", "reobserve_from_current_view"
    if skill.startswith("navigate.") or skill == "manip.align_workspace":
        return "navigation_micro_adjust", "micro_adjust_base_for_observation"
    if skill in {
        "manip.single_hand_pinch",
        "manip.side_grasp",
        "manip.top_grasp",
        "manip.bimanual_clamp",
    }:
        return "runtime_replan", "repair_grasp_contact"
    return "runtime_replan", "regenerate_runtime_plan"


def _primitive_recovery_overrides(skill_name: str, reason: str, attempt: int) -> dict[str, dict[str, float]]:
    """Return a bounded physical repair profile carried with runtime replan."""
    level = min(3, max(1, int(attempt)))
    text = str(reason).lower()
    if skill_name == "manip.side_grasp" and "object_contact_ready" in text:
        # Regrasp locally before escalating to perception. Alternate a small
        # lateral target correction and raise closure with each retry.
        direction = -1.0 if level % 2 else 1.0
        return {
            "manip.side_grasp": {
                "close_ratio": round(min(0.82, 0.66 + 0.05 * level), 5),
                "contact_x_delta_m": round(direction * min(0.012, 0.004 * level), 5),
                "contact_z_delta_m": -0.006,
            }
        }
    if skill_name != "manip.lift_object" or ("object_in_hand" not in text and "low_hold_contact_lost" not in text):
        return {}
    return {
        "manip.side_grasp": {"close_ratio": round(min(0.82, 0.68 + 0.04 * level), 5)},
        "manip.lift_object": {
            "squeeze_close_ratio": round(min(0.90, 0.78 + 0.03 * level), 5),
            "hold_close_ratio": round(min(0.94, 0.86 + 0.02 * level), 5),
            "servo_lift_z_lead": round(max(0.025, 0.035 - 0.003 * level), 5),
            # Once contact is verified but the object has not cleared the
            # support plane, increase trajectory height and duration together.
            # This is bounded below the drop-prone 0.30m exploration limit.
            "lift_z": round(min(0.26, 0.19 + 0.02 * level), 5),
            "lift_duration": round(min(3.2, 1.8 + 0.4 * level), 5),
        },
    }


def _resume_plan_from_action(plan: dict[str, Any], failed_action_id: str) -> dict[str, Any]:
    """Keep verified prefix actions out of a recovery-generated replacement plan."""
    actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    start = next(
        (index for index, action in enumerate(actions) if isinstance(action, dict) and action.get("action_id") == failed_action_id),
        None,
    )
    if start is None:
        return plan
    failed = actions[start] if isinstance(actions[start], dict) else {}
    # ``object_in_hand`` requires rebuilding contact. Retrying lift alone can
    # never repair an already-open grasp, so include the preceding grasp step.
    if str(failed.get("source_id") or failed.get("skill_name") or "") == "manip.lift_object":
        grasp_skills = {"manip.side_grasp", "manip.top_grasp", "manip.single_hand_pinch", "manip.bimanual_clamp"}
        for index in range(start - 1, -1, -1):
            candidate = actions[index]
            if isinstance(candidate, dict) and str(candidate.get("source_id") or candidate.get("skill_name") or "") in grasp_skills:
                start = index
                break
    resumed = dict(plan)
    resumed_actions = [dict(action) for action in actions[start:] if isinstance(action, dict)]
    resumed["actions"] = resumed_actions
    if resumed_actions:
        resumed["next_action"] = resumed_actions[0]
    metadata = dict(resumed.get("metadata") or {})
    metadata["recovery_resume_from_action_id"] = failed_action_id
    metadata["recovery_skipped_verified_action_count"] = start
    resumed["metadata"] = metadata
    return resumed


if __name__ == "__main__":
    main()
