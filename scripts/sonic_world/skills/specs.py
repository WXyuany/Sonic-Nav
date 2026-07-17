from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillSpec:
    name: str
    target_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    timeout: float | None = None
    recovery: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_id": self.target_id,
            "params": self.params,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "timeout": self.timeout,
            "recovery": self.recovery,
        }


@dataclass(frozen=True)
class SkillGraph:
    task_id: str
    objective: str
    steps: list[SkillSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }


def approach_object(target_id: str, params: dict[str, Any]) -> SkillSpec:
    return SkillSpec(
        "navigate.approach_object",
        target_id=target_id,
        params=params,
        preconditions=["object_localized", "navigation_map_ready"],
        effects=["robot_near_object"],
        recovery=["relocalize_object", "replan_approach_pose"],
    )


def navigate_to(target_id: str, params: dict[str, Any]) -> SkillSpec:
    return SkillSpec(
        "navigate.goto",
        target_id=target_id,
        params=params,
        preconditions=["navigation_map_ready"],
        effects=["robot_at_goal"],
        recovery=["replan_global_path", "clear_local_costmap", "relax_goal_tolerance"],
    )


def align_workspace(target_id: str, params: dict[str, Any]) -> SkillSpec:
    return SkillSpec(
        "manip.align_workspace",
        target_id=target_id,
        params=params,
        preconditions=["robot_near_object"],
        effects=["object_in_hand_workspace"],
        recovery=["micro_step_base", "increase_standoff"],
    )


def grasp_object(target_id: str, affordance: str, params: dict[str, Any]) -> SkillSpec:
    return SkillSpec(
        f"manip.{affordance}",
        target_id=target_id,
        params=params,
        preconditions=["object_in_hand_workspace"],
        effects=["object_contact_ready"],
        recovery=["retry_pregrasp", "switch_affordance", "adjust_base_standoff"],
    )


def lift_object(target_id: str, params: dict[str, Any] | None = None) -> SkillSpec:
    return SkillSpec(
        "manip.lift_object",
        target_id=target_id,
        params={} if params is None else params,
        preconditions=["object_contact_ready"],
        effects=["object_in_hand"],
        recovery=["squeeze_more", "lower_and_regrasp", "abort_if_unstable"],
    )


def transport_object(target_id: str, destination_id: str, params: dict[str, Any] | None = None) -> SkillSpec:
    payload = {"destination_id": destination_id}
    if params:
        payload.update(params)
    return SkillSpec(
        "manip.transport_object",
        target_id=target_id,
        params=payload,
        preconditions=["object_in_hand"],
        effects=["object_near_destination"],
        recovery=["hold_pose", "replan_navigation", "lower_if_unstable"],
    )


def place_object(target_id: str, destination_id: str, params: dict[str, Any] | None = None) -> SkillSpec:
    payload = {"destination_id": destination_id}
    if params:
        payload.update(params)
    return SkillSpec(
        "manip.place_object",
        target_id=target_id,
        params=payload,
        preconditions=["object_near_destination"],
        effects=["object_on_destination"],
        recovery=["raise_and_retry_place", "choose_nearby_place_region"],
    )


def release_object(target_id: str) -> SkillSpec:
    return SkillSpec(
        "manip.release",
        target_id=target_id,
        preconditions=["object_on_destination"],
        effects=["hand_free"],
        recovery=["open_hand_again", "retreat_hand"],
    )
