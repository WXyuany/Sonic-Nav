from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dispatch import DispatchPlan, DispatchStep
from .recovery import RecoveryAction, RecoveryPlan


@dataclass(frozen=True)
class DecisionAction:
    action_id: str
    kind: str
    handler: str
    target_id: str | None
    source_id: str
    command: dict[str, Any]
    priority: int = 100
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "handler": self.handler,
            "target_id": self.target_id,
            "source_id": self.source_id,
            "command": self.command,
            "priority": int(self.priority),
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DecisionPlan:
    task_id: str
    objective: str
    status: str
    next_action: DecisionAction | None
    actions: tuple[DecisionAction, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "next_action": self.next_action.to_dict() if self.next_action else None,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": self.metadata,
        }


def decision_plan_for_plans(
    dispatch: DispatchPlan,
    recovery: RecoveryPlan,
) -> DecisionPlan:
    if recovery.actions:
        actions = tuple(
            _decision_from_recovery_action(action, dispatch.task_id)
            for action in sorted(recovery.actions, key=lambda item: (item.priority, item.action_id))
        )
        return DecisionPlan(
            task_id=dispatch.task_id,
            objective=dispatch.objective,
            status="needs_recovery",
            next_action=actions[0],
            actions=actions,
            metadata={
                "source": "recovery_plan",
                "demo_kind": dispatch.metadata.get("demo_kind"),
                "dispatch_unready_count": dispatch.metadata.get("unready_count", 0),
                "contract_error_count": dispatch.metadata.get("contract_error_count", 0),
                "recovery_action_count": len(actions),
            },
        )

    ready_steps = tuple(step for step in dispatch.steps if step.readiness == "ready")
    unready_count = int(dispatch.metadata.get("unready_count") or 0)
    if unready_count == 0:
        actions = tuple(
            _decision_from_dispatch_step(step, index, dispatch.task_id)
            for index, step in enumerate(ready_steps, start=1)
        )
        return DecisionPlan(
            task_id=dispatch.task_id,
            objective=dispatch.objective,
            status="ready_to_execute",
            next_action=actions[0] if actions else None,
            actions=actions,
            metadata={
                "source": "dispatch_plan",
                "demo_kind": dispatch.metadata.get("demo_kind"),
                "dispatch_unready_count": 0,
                "contract_error_count": dispatch.metadata.get("contract_error_count", 0),
                "dispatch_action_count": len(actions),
            },
        )

    return DecisionPlan(
        task_id=dispatch.task_id,
        objective=dispatch.objective,
        status="blocked",
        next_action=None,
        actions=(),
        metadata={
            "source": "dispatch_plan",
            "demo_kind": dispatch.metadata.get("demo_kind"),
            "dispatch_unready_count": unready_count,
            "contract_error_count": dispatch.metadata.get("contract_error_count", 0),
            "reason": "dispatch has unready steps but no routable recovery action",
        },
    )


def _decision_from_recovery_action(action: RecoveryAction, task_id: str) -> DecisionAction:
    return DecisionAction(
        action_id=f"decision_{_safe_id(task_id)}_{action.action_id}",
        kind="recovery",
        handler=action.handler,
        target_id=action.target_id,
        source_id=action.action_id,
        command=action.command,
        priority=action.priority,
        reason=action.suggestion,
        metadata={
            "affected_skills": list(action.affected_skills),
            "failed_checks": list(action.failed_checks),
            "suggestion": action.suggestion,
        },
    )


def _decision_from_dispatch_step(step: DispatchStep, index: int, task_id: str) -> DecisionAction:
    metadata = {
        "capability": step.capability,
        "phase_names": list(step.phase_names),
        "monitor_events": list(step.monitor_events),
        "recovery_events": list(step.recovery_events),
        "preconditions": list(step.preconditions),
        "effects": list(step.effects),
        "readiness": step.readiness,
    }
    if step.contract is not None:
        metadata["contract"] = step.contract.to_dict()
    return DecisionAction(
        action_id=f"decision_{_safe_id(task_id)}_dispatch_{index:02d}_{step.skill_name.replace('.', '_')}",
        kind="dispatch",
        handler=step.handler,
        target_id=step.target_id,
        source_id=step.skill_name,
        command=step.command,
        priority=index,
        reason=f"execute {step.skill_name}",
        metadata=metadata,
    )


def _safe_id(value: str) -> str:
    text = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))
    return text[:64] or "task"
