from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dispatch import DispatchPlan, DispatchStep


@dataclass(frozen=True)
class RecoveryAction:
    action_id: str
    suggestion: str
    handler: str
    target_id: str | None
    affected_skills: tuple[str, ...]
    failed_checks: tuple[str, ...]
    command: dict[str, Any]
    priority: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "suggestion": self.suggestion,
            "handler": self.handler,
            "target_id": self.target_id,
            "affected_skills": list(self.affected_skills),
            "failed_checks": list(self.failed_checks),
            "command": self.command,
            "priority": int(self.priority),
        }


@dataclass(frozen=True)
class RecoveryPlan:
    task_id: str
    objective: str
    status: str
    actions: tuple[RecoveryAction, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": self.metadata,
        }


def recovery_plan_for_dispatch(dispatch: DispatchPlan) -> RecoveryPlan:
    buckets: dict[tuple[str, str | None], dict[str, Any]] = {}
    for step in dispatch.steps:
        if step.contract is None or step.contract.ready:
            continue
        failed = step.contract.failed_errors
        for suggestion in step.contract.recovery_suggestions:
            key = (suggestion, step.target_id)
            bucket = buckets.setdefault(
                key,
                {
                    "suggestion": suggestion,
                    "target_id": step.target_id,
                    "affected_skills": [],
                    "failed_checks": [],
                },
            )
            bucket["affected_skills"].append(step.skill_name)
            bucket["failed_checks"].extend(failed)

    actions = tuple(
        _action_from_bucket(index, bucket)
        for index, bucket in enumerate(
            sorted(buckets.values(), key=lambda item: (_priority(str(item["suggestion"])), str(item["suggestion"]))),
            start=1,
        )
    )
    return RecoveryPlan(
        task_id=dispatch.task_id,
        objective=dispatch.objective,
        status="needs_recovery" if actions else "not_needed",
        actions=actions,
        metadata={
            "action_count": len(actions),
            "source_unready_count": dispatch.metadata.get("unready_count", 0),
            "source_contract_error_count": dispatch.metadata.get("contract_error_count", 0),
            "suggestions": list(dict.fromkeys(action.suggestion for action in actions)),
        },
    )


def _action_from_bucket(index: int, bucket: dict[str, Any]) -> RecoveryAction:
    suggestion = str(bucket["suggestion"])
    target_id = bucket["target_id"]
    affected_skills = tuple(dict.fromkeys(str(item) for item in bucket["affected_skills"]))
    failed_checks = tuple(dict.fromkeys(str(item) for item in bucket["failed_checks"]))
    handler, command = _handler_and_command(suggestion, target_id, affected_skills, failed_checks)
    return RecoveryAction(
        action_id=f"recovery_{index:02d}_{suggestion}",
        suggestion=suggestion,
        handler=handler,
        target_id=target_id,
        affected_skills=affected_skills,
        failed_checks=failed_checks,
        command=command,
        priority=_priority(suggestion),
    )


def _handler_and_command(
    suggestion: str,
    target_id: str | None,
    affected_skills: tuple[str, ...],
    failed_checks: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    base = {
        "type": suggestion,
        "object_id": target_id,
        "affected_skills": list(affected_skills),
        "failed_checks": list(failed_checks),
    }
    if suggestion == "reobserve_from_current_view":
        return "perception_reobserve", {
            **base,
            "required_frames": ["base_link"],
            "preferred_topic": "/sonic_world/object_anchor",
        }
    if suggestion == "publish_object_anchor_with_pose_base":
        return "object_anchor_update", {
            **base,
            "required_fields": ["pose_base"],
            "required_frame": "base_link",
            "target_topic": "/sonic_world/object_anchor",
        }
    if suggestion == "micro_adjust_base_for_observation":
        return "navigation_micro_adjust", {
            **base,
            "max_step_m": 0.08,
            "purpose": "improve_object_base_pose",
        }
    if suggestion in {"request_object_anchor", "refresh_world_memory"}:
        return "world_memory_update", {
            **base,
            "target_topic": "/sonic_world/object_anchor",
        }
    if suggestion in {"request_place_target_anchor", "choose_nearby_place_region"}:
        return "place_target_recovery", {
            **base,
            "required_category": "place_target",
            "target_topic": "/sonic_world/object_anchor",
        }
    if suggestion in {"estimate_approach_pose", "request_map_pose", "replan_approach_pose"}:
        return "navigation_replan", {
            **base,
            "required_fields": ["pose_map"],
        }
    if suggestion in {"switch_runtime_template", "add_runtime_phase_binding", "regenerate_runtime_plan"}:
        return "runtime_replan", base
    if suggestion in {"infer_grasp_from_shape", "request_explicit_affordance", "repair_grasp_affordance"}:
        return "affordance_repair", {
            **base,
            "target_topic": "/sonic_world/object_anchor",
        }
    if suggestion == "infer_support_surface":
        return "support_surface_inference", base
    return "manual_review", base


def _priority(suggestion: str) -> int:
    order = {
        "publish_object_anchor_with_pose_base": 10,
        "reobserve_from_current_view": 20,
        "micro_adjust_base_for_observation": 30,
        "request_object_anchor": 40,
        "request_place_target_anchor": 40,
        "estimate_approach_pose": 50,
        "replan_approach_pose": 60,
        "switch_runtime_template": 70,
        "regenerate_runtime_plan": 75,
        "request_explicit_affordance": 80,
        "manual_review": 100,
    }
    return order.get(suggestion, 90)
