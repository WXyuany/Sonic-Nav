from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "task_skill_policy_v0"


@dataclass(frozen=True)
class PolicyObservation:
    task_id: str
    suite: str
    suite_version: str
    request: dict[str, Any]
    world_frame: str
    robot: dict[str, Any]
    objects: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    sensor_contract: list[str] = field(default_factory=list)
    available_skills: list[dict[str, Any]] = field(default_factory=list)
    dispatch_status: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "task_id": self.task_id,
            "suite": self.suite,
            "suite_version": self.suite_version,
            "request": self.request,
            "world_frame": self.world_frame,
            "robot": self.robot,
            "objects": self.objects,
            "relations": self.relations,
            "sensor_contract": self.sensor_contract,
            "available_skills": self.available_skills,
            "dispatch_status": self.dispatch_status,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PolicyAction:
    policy_id: str
    task_id: str
    status: str
    task_intent: dict[str, Any]
    object_target_anchors: list[dict[str, Any]]
    skill_selection: list[str]
    base_goal: dict[str, Any] | None = None
    hand_pose_target: dict[str, Any] | None = None
    wrist_target: dict[str, Any] | None = None
    grasp_close_ratio: dict[str, Any] | None = None
    grasp_offsets: dict[str, Any] | None = None
    lift_place_targets: dict[str, Any] | None = None
    recovery_decision: dict[str, Any] | None = None
    ordered_skill_commands: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "status": self.status,
            "task_intent": self.task_intent,
            "object_target_anchors": self.object_target_anchors,
            "skill_selection": self.skill_selection,
            "base_goal": self.base_goal,
            "hand_pose_target": self.hand_pose_target,
            "wrist_target": self.wrist_target,
            "grasp_close_ratio": self.grasp_close_ratio,
            "grasp_offsets": self.grasp_offsets,
            "lift_place_targets": self.lift_place_targets,
            "recovery_decision": self.recovery_decision,
            "ordered_skill_commands": self.ordered_skill_commands,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PolicySample:
    sample_id: str
    observation: PolicyObservation
    action: PolicyAction
    planning: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "metadata": self.metadata,
        }
        if self.planning is not None:
            payload["planning"] = self.planning
        return payload
