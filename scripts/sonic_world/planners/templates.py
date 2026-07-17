from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..skills.specs import (
    SkillSpec,
    align_workspace,
    approach_object,
    grasp_object,
    lift_object,
    navigate_to,
    place_object,
    release_object,
    transport_object,
)
from ..world_model.entities import Affordance, Pose3, WorldObject, WorldState


@dataclass(frozen=True)
class PlanContext:
    world: WorldState
    request: Any
    task_object: WorldObject | None
    target: WorldObject | None
    grasp: Affordance | None
    approach_params: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    verbs: frozenset[str]
    builder: Callable[[PlanContext], list[SkillSpec]]
    requires_object: bool = True
    requires_grasp: bool = False
    prefers_target: bool = False

    def matches(self, verb: str) -> bool:
        return verb in self.verbs


def default_task_templates() -> tuple[TaskTemplate, ...]:
    return (
        TaskTemplate(
            name="navigation",
            verbs=frozenset({"navigate", "go_to", "goto"}),
            builder=_build_navigation,
            requires_grasp=False,
            prefers_target=True,
        ),
        TaskTemplate(
            name="approach",
            verbs=frozenset({"approach"}),
            builder=_build_approach,
            requires_grasp=False,
        ),
        TaskTemplate(
            name="pick",
            verbs=frozenset({"pick", "grasp"}),
            builder=_build_pick,
            requires_grasp=True,
        ),
        TaskTemplate(
            name="pick_place",
            verbs=frozenset({"pick_place", "move", "place"}),
            builder=_build_pick_place,
            requires_grasp=True,
            prefers_target=True,
        ),
    )


def _build_navigation(ctx: PlanContext) -> list[SkillSpec]:
    target = ctx.target or ctx.task_object
    if target is None:
        raise ValueError("navigation task has no target")
    params = _pose_params(target.pose_map or target.pose_base)
    params.update(_navigation_params(target))
    return [navigate_to(target.object_id, params)]


def _build_approach(ctx: PlanContext) -> list[SkillSpec]:
    obj = _require_object(ctx)
    return [approach_object(obj.object_id, dict(ctx.approach_params))]


def _build_pick(ctx: PlanContext) -> list[SkillSpec]:
    obj = _require_object(ctx)
    grasp = _require_grasp(ctx)
    return [
        approach_object(obj.object_id, dict(ctx.approach_params)),
        align_workspace(obj.object_id, _workspace_params(obj, grasp)),
        grasp_object(obj.object_id, grasp.name, grasp.params),
        lift_object(obj.object_id, {"lift_check": "object_z_delta"}),
    ]


def _build_pick_place(ctx: PlanContext) -> list[SkillSpec]:
    steps = _build_pick(ctx)
    obj = _require_object(ctx)
    target = ctx.target
    if target is None:
        return steps
    steps.extend(
        [
            transport_object(obj.object_id, target.object_id, _transport_params(target)),
            place_object(obj.object_id, target.object_id, _place_params(target)),
            release_object(obj.object_id),
        ]
    )
    return steps


def _workspace_params(obj: WorldObject, grasp: Affordance) -> dict[str, Any]:
    return {
        "preferred_frame": "base_link",
        "object_pose_base": obj.pose_base.to_dict() if obj.pose_base else None,
        "hand_workspace": "bimanual" if grasp.name == "bimanual_clamp" else "right",
        "support": obj.support,
    }


def _pose_params(pose: Pose3 | None) -> dict[str, Any]:
    return {"pose": pose.to_dict()} if pose is not None else {}


def _navigation_params(target: WorldObject) -> dict[str, Any]:
    params: dict[str, Any] = {}
    tolerance = target.properties.get("goal_tolerance")
    if tolerance is not None:
        params["goal_tolerance"] = tolerance
    return params


def _transport_params(target: WorldObject) -> dict[str, Any]:
    return {
        "destination_pose_map": target.pose_map.to_dict() if target.pose_map else None,
        "destination_pose_base": target.pose_base.to_dict() if target.pose_base else None,
    }


def _place_params(target: WorldObject) -> dict[str, Any]:
    return {
        "place_pose_map": target.pose_map.to_dict() if target.pose_map else None,
        "place_pose_base": target.pose_base.to_dict() if target.pose_base else None,
        "support": target.support,
    }


def _require_object(ctx: PlanContext) -> WorldObject:
    if ctx.task_object is None:
        raise ValueError("task has no object")
    return ctx.task_object


def _require_grasp(ctx: PlanContext) -> Affordance:
    if ctx.grasp is None:
        obj = ctx.task_object.object_id if ctx.task_object is not None else "<none>"
        raise ValueError(f"no grasp affordance available for {obj}")
    return ctx.grasp
