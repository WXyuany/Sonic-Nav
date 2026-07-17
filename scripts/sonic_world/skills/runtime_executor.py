from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable


GoalPublisher = Callable[[dict[str, Any] | None], bool]
PrimitivePublisher = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class SkillRuntimeResult:
    task_id: str | None
    objective: str | None
    action_id: str | None
    skill_name: str | None
    target_id: str | None
    handler: str
    capability: str | None
    status: str
    action: str
    reason: str = ""
    command: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "action_id": self.action_id,
            "skill_name": self.skill_name,
            "target_id": self.target_id,
            "handler": self.handler,
            "capability": self.capability,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "command": self.command,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "stamp": float(self.stamp),
        }


class SkillRuntimeExecutor:
    """Execute dispatch actions through a small, backend-neutral contract.

    The world model emits high-level dispatch actions. This executor does not
    know how to move a specific robot by itself; it normalizes those actions into
    either a navigation goal publish or a primitive command publish. Real
    primitive runners can subscribe to the command stream and report status with
    the same action/skill identifiers.
    """

    def __init__(
        self,
        *,
        publish_goal: GoalPublisher | None = None,
        publish_primitive: PrimitivePublisher | None = None,
        execute_navigation: bool = False,
    ) -> None:
        self.publish_goal = publish_goal
        self.publish_primitive = publish_primitive
        self.execute_navigation = bool(execute_navigation)

    def execute_decision_action(
        self,
        plan: dict[str, Any],
        action: dict[str, Any],
        *,
        source: str,
    ) -> SkillRuntimeResult:
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        plan_metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        if "demo_kind" not in metadata and plan_metadata.get("demo_kind") is not None:
            metadata = {**metadata, "demo_kind": plan_metadata.get("demo_kind")}
        command = action.get("command") if isinstance(action.get("command"), dict) else {}
        return self._execute(
            task_id=_string_or_none(plan.get("task_id")),
            objective=_string_or_none(plan.get("objective")),
            action_id=_string_or_none(action.get("action_id")),
            skill_name=_string_or_none(action.get("source_id")),
            target_id=_string_or_none(action.get("target_id")),
            handler=str(action.get("handler") or ""),
            capability=_string_or_none(metadata.get("capability")),
            command=command,
            metadata={**metadata, "source": source, "decision_kind": action.get("kind")},
            reason=str(action.get("reason") or ""),
        )

    def execute_dispatch_step(
        self,
        plan: dict[str, Any],
        step: dict[str, Any],
        *,
        index: int,
        source: str,
    ) -> SkillRuntimeResult:
        metadata = {
            "source": source,
            "demo_kind": (plan.get("metadata") or {}).get("demo_kind") if isinstance(plan.get("metadata"), dict) else None,
            "dispatch_index": index,
            "phase_names": step.get("phase_names", []),
            "monitor_events": step.get("monitor_events", []),
            "recovery_events": step.get("recovery_events", []),
            "preconditions": step.get("preconditions", []),
            "effects": step.get("effects", []),
            "readiness": step.get("readiness"),
            "contract": step.get("contract"),
        }
        command = step.get("command") if isinstance(step.get("command"), dict) else {}
        return self._execute(
            task_id=_string_or_none(plan.get("task_id")),
            objective=_string_or_none(plan.get("objective")),
            action_id=f"dispatch_step_{index:02d}",
            skill_name=_string_or_none(step.get("skill_name")),
            target_id=_string_or_none(step.get("target_id")),
            handler=str(step.get("handler") or ""),
            capability=_string_or_none(step.get("capability")),
            command=command,
            metadata=metadata,
            reason=f"execute {step.get('skill_name')}",
        )

    def _execute(
        self,
        *,
        task_id: str | None,
        objective: str | None,
        action_id: str | None,
        skill_name: str | None,
        target_id: str | None,
        handler: str,
        capability: str | None,
        command: dict[str, Any],
        metadata: dict[str, Any],
        reason: str,
    ) -> SkillRuntimeResult:
        if handler == "ros2_goal_pose":
            return self._execute_navigation(
                task_id=task_id,
                objective=objective,
                action_id=action_id,
                skill_name=skill_name,
                target_id=target_id,
                handler=handler,
                capability=capability,
                command=command,
                metadata=metadata,
                reason=reason,
            )
        if handler == "demo_locomotion_phase_runtime" or handler.endswith("_primitive"):
            return self._execute_primitive(
                task_id=task_id,
                objective=objective,
                action_id=action_id,
                skill_name=skill_name,
                target_id=target_id,
                handler=handler,
                capability=capability,
                command=command,
                metadata=metadata,
                reason=reason,
            )
        return SkillRuntimeResult(
            task_id=task_id,
            objective=objective,
            action_id=action_id,
            skill_name=skill_name,
            target_id=target_id,
            handler=handler,
            capability=capability,
            command=command,
            status="skipped",
            action="skip_unknown_handler",
            reason=reason or "unknown handler",
            metadata=metadata,
        )

    def _execute_navigation(
        self,
        *,
        task_id: str | None,
        objective: str | None,
        action_id: str | None,
        skill_name: str | None,
        target_id: str | None,
        handler: str,
        capability: str | None,
        command: dict[str, Any],
        metadata: dict[str, Any],
        reason: str,
    ) -> SkillRuntimeResult:
        if not self.execute_navigation:
            return SkillRuntimeResult(
                task_id=task_id,
                objective=objective,
                action_id=action_id,
                skill_name=skill_name,
                target_id=target_id,
                handler=handler,
                capability=capability,
                command=command,
                status="dry_run",
                action="dry_run_goal_pose",
                reason=reason,
                metadata=metadata,
            )
        if command.get("type") != "publish_pose_stamped":
            return SkillRuntimeResult(
                task_id=task_id,
                objective=objective,
                action_id=action_id,
                skill_name=skill_name,
                target_id=target_id,
                handler=handler,
                capability=capability,
                command=command,
                status="failed",
                action="failed_publish_goal_pose",
                reason="navigation command is not publish_pose_stamped",
                metadata=metadata,
            )
        ok = bool(self.publish_goal and self.publish_goal(command.get("pose")))
        return SkillRuntimeResult(
            task_id=task_id,
            objective=objective,
            action_id=action_id,
            skill_name=skill_name,
            target_id=target_id,
            handler=handler,
            capability=capability,
            command=command,
            status="queued" if ok else "failed",
            action="published_goal_pose" if ok else "failed_publish_goal_pose",
            reason=reason,
            metadata=metadata,
        )

    def _execute_primitive(
        self,
        *,
        task_id: str | None,
        objective: str | None,
        action_id: str | None,
        skill_name: str | None,
        target_id: str | None,
        handler: str,
        capability: str | None,
        command: dict[str, Any],
        metadata: dict[str, Any],
        reason: str,
    ) -> SkillRuntimeResult:
        primitive_payload = {
            "schema": "sonic_skill_primitive_command_v0",
            "task_id": task_id,
            "objective": objective,
            "action_id": action_id,
            "skill_name": skill_name,
            "target_id": target_id,
            "demo_kind": metadata.get("demo_kind"),
            "handler": handler,
            "capability": capability,
            "command": command,
            "phase_names": metadata.get("phase_names") or [],
            "monitor_events": metadata.get("monitor_events") or [],
            "recovery_events": metadata.get("recovery_events") or [],
            "contract": metadata.get("contract"),
            "effects": metadata.get("effects") or [],
            "reason": reason,
            "source": metadata.get("source"),
            "stamp": time.time(),
        }
        if self.publish_primitive is None:
            return SkillRuntimeResult(
                task_id=task_id,
                objective=objective,
                action_id=action_id,
                skill_name=skill_name,
                target_id=target_id,
                handler=handler,
                capability=capability,
                command=command,
                status="queued",
                action="wait_for_runtime_phase",
                reason=reason,
                metadata={**metadata, "primitive_command": primitive_payload},
            )
        ok = bool(self.publish_primitive(primitive_payload))
        return SkillRuntimeResult(
            task_id=task_id,
            objective=objective,
            action_id=action_id,
            skill_name=skill_name,
            target_id=target_id,
            handler=handler,
            capability=capability,
            command=command,
            status="queued" if ok else "failed",
            action="published_primitive_command" if ok else "failed_publish_primitive_command",
            reason=reason,
            metrics={"phase_count": len(primitive_payload["phase_names"])},
            metadata={**metadata, "primitive_command": primitive_payload},
        )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
