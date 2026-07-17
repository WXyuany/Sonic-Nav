from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .schema import PolicyAction, PolicyObservation, PolicySample
from ..planners import PlanningResult
from ..skills.decision import DecisionAction
from ..skills.specs import SkillSpec
from ..world_model.entities import Pose3, WorldObject


POLICY_ID = "heuristic_task_skill_policy_v0"


@dataclass(frozen=True)
class HeuristicSkillPolicy:
    policy_id: str = POLICY_ID

    def observation(
        self,
        result: PlanningResult,
        *,
        suite_name: str,
        suite_version: str,
        suite_metadata: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
        task_tags: tuple[str, ...] | list[str] = (),
    ) -> PolicyObservation:
        task_id = _task_id(result)
        objects = [_compact_object(obj) for obj in result.world.objects.values()]
        return PolicyObservation(
            task_id=task_id,
            suite=suite_name,
            suite_version=suite_version,
            request=result.request.to_dict(),
            world_frame=result.world.frame_id,
            robot=result.world.robot.to_dict(),
            objects=objects,
            relations=[rel.to_dict() for rel in result.world.relations],
            sensor_contract=list((suite_metadata or {}).get("sensor_contract") or ()),
            available_skills=[step.to_dict() for step in result.dispatch_plan.steps],
            dispatch_status={
                "decision_status": result.decision_plan.status,
                "unready_count": result.dispatch_plan.metadata.get("unready_count", 0),
                "contract_error_count": result.dispatch_plan.metadata.get("contract_error_count", 0),
                "contract_warning_count": result.dispatch_plan.metadata.get("contract_warning_count", 0),
                "recovery_suggestions": list(result.dispatch_plan.metadata.get("recovery_suggestions") or ()),
            },
            metadata={
                "task_tags": list(task_tags),
                "task_metadata": dict(task_metadata or {}),
                "planner": result.skill_graph.metadata.get("planner"),
                "task_template": result.skill_graph.metadata.get("task_template"),
                "demo_kind": result.runtime_plan.demo_kind,
            },
        )

    def act(self, result: PlanningResult) -> PolicyAction:
        task_id = _task_id(result)
        grasp_step = _first_grasp_step(result.skill_graph.steps)
        task_object = _object_for_step(result, grasp_step) or _primary_task_object(result)
        target = _target_object(result)
        base_goal = _base_goal(result)
        hand_pose = _hand_pose_target(grasp_step, task_object)
        wrist = _wrist_target(grasp_step, task_object)
        close_ratio = _grasp_close_ratio(grasp_step, task_object)
        offsets = _grasp_offsets(grasp_step, task_object, hand_pose)
        lift_place = _lift_place_targets(result, task_object, target, hand_pose)
        recovery = _recovery_decision(result)
        return PolicyAction(
            policy_id=self.policy_id,
            task_id=task_id,
            status=_action_status(result),
            task_intent={
                "verb": result.request.verb,
                "object_id": result.request.object_id,
                "object_category": result.request.object_category,
                "target_id": result.request.target_id,
                "demo_kind": result.runtime_plan.demo_kind,
            },
            object_target_anchors=_object_target_anchors(result, task_object, target),
            skill_selection=[step.name for step in result.skill_graph.steps],
            base_goal=base_goal,
            hand_pose_target=hand_pose,
            wrist_target=wrist,
            grasp_close_ratio=close_ratio,
            grasp_offsets=offsets,
            lift_place_targets=lift_place,
            recovery_decision=recovery,
            ordered_skill_commands=[_compact_decision_action(action) for action in result.decision_plan.actions],
            metadata={
                "controller_boundary": "frozen_sonic_low_level",
                "trainable_scope": "task_and_skill_policy_only",
                "grasp_affordance": result.skill_graph.metadata.get("grasp_affordance"),
                "object_category": result.skill_graph.metadata.get("object_category"),
                "target_category": result.skill_graph.metadata.get("target_category"),
                "plan_ready": result.decision_plan.status == "ready_to_execute",
                "dispatch_metadata": dict(result.dispatch_plan.metadata),
                "runtime_metadata": dict(result.runtime_plan.metadata),
            },
        )

    def sample(
        self,
        result: PlanningResult,
        *,
        suite_name: str,
        suite_version: str,
        suite_metadata: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
        task_tags: tuple[str, ...] | list[str] = (),
        include_planning: bool = False,
    ) -> PolicySample:
        obs = self.observation(
            result,
            suite_name=suite_name,
            suite_version=suite_version,
            suite_metadata=suite_metadata,
            task_metadata=task_metadata,
            task_tags=task_tags,
        )
        action = self.act(result)
        planning = result.to_dict() if include_planning else None
        return PolicySample(
            sample_id=f"{suite_name}:{suite_version}:{obs.task_id}",
            observation=obs,
            action=action,
            planning=planning,
            metadata={
                "source": "heuristic_teacher",
                "policy_id": self.policy_id,
                "controller_training": "frozen_sonic_no_low_level_training",
            },
        )


def _action_status(result: PlanningResult) -> str:
    if result.recovery_plan.actions:
        return "needs_recovery"
    if result.decision_plan.status == "ready_to_execute":
        return "ready"
    return result.decision_plan.status


def _task_id(result: PlanningResult) -> str:
    request_id = result.request.metadata.get("request_id")
    if request_id:
        return str(request_id)
    return str(result.skill_graph.task_id)


def _first_grasp_step(steps: list[SkillSpec]) -> SkillSpec | None:
    for step in steps:
        if step.name in {"manip.bimanual_clamp", "manip.single_hand_pinch", "manip.side_grasp", "manip.top_grasp"}:
            return step
    return None


def _object_for_step(result: PlanningResult, step: SkillSpec | None) -> WorldObject | None:
    if step is None or step.target_id is None:
        return None
    return result.world.get_object(step.target_id)


def _primary_task_object(result: PlanningResult) -> WorldObject | None:
    if result.request.object_id:
        obj = result.world.get_object(result.request.object_id)
        if obj is not None:
            return obj
    return result.world.primary_object()


def _target_object(result: PlanningResult) -> WorldObject | None:
    if result.request.target_id:
        obj = result.world.get_object(result.request.target_id)
        if obj is not None:
            return obj
    for obj in result.world.objects.values():
        if obj.category in {"place_target", "navigation_goal"}:
            return obj
    return None


def _object_target_anchors(
    result: PlanningResult,
    task_object: WorldObject | None,
    target: WorldObject | None,
) -> list[dict[str, Any]]:
    ids: list[str] = []
    for obj in (task_object, target):
        if obj is not None and obj.object_id not in ids:
            ids.append(obj.object_id)
    if not ids:
        ids = [
            obj.object_id
            for obj in result.world.objects.values()
            if obj.category not in {"table", "counter", "support_surface"}
        ]
    return [_compact_object(result.world.objects[obj_id]) for obj_id in ids if obj_id in result.world.objects]


def _base_goal(result: PlanningResult) -> dict[str, Any] | None:
    for step in result.dispatch_plan.steps:
        if step.capability != "navigation":
            continue
        pose = step.command.get("approach_pose") or step.command.get("pose")
        if isinstance(pose, dict):
            return {
                "frame_id": pose.get("frame_id") or "map",
                "position": _list3((pose.get("position") or [math.nan, math.nan, math.nan])),
                "yaw": _optional_float(pose.get("yaw")),
                "handler": step.handler,
                "skill_name": step.skill_name,
                "standoff": step.command.get("standoff"),
            }
    return None


def _hand_pose_target(step: SkillSpec | None, obj: WorldObject | None) -> dict[str, Any] | None:
    if step is None or obj is None:
        return None
    pose = obj.pose_base
    obj_pos = _pose_position(pose, default=(0.52, 0.0, 0.04))
    params = step.params
    reach_x = _finite(params.get("reach_x"), obj_pos[0])
    target_y = _finite(params.get("target_y"), obj_pos[1])
    reach_z = _finite(params.get("reach_z"), obj_pos[2])
    object_height = _object_height(obj)
    clearance_z = max(reach_z + 0.16, obj_pos[2] + object_height + 0.08, 0.16)

    if step.name == "manip.bimanual_clamp":
        half_y = max(_object_half_y(obj) + 0.015, _finite(params.get("clamp_y"), 0.12) * 0.5)
        open_half_y = max(half_y + 0.06, _finite(params.get("open_y"), 0.24) * 0.5)
        center_y = obj_pos[1]
        contact_z = max(obj_pos[2], reach_z)
        return {
            "mode": "bimanual_clamp",
            "frame_id": "base_link",
            "object_id": obj.object_id,
            "pregrasp": {
                "left_forearm": [reach_x, center_y + open_half_y, clearance_z],
                "right_forearm": [reach_x, center_y - open_half_y, clearance_z],
            },
            "contact": {
                "left_forearm": [reach_x, center_y + half_y, contact_z],
                "right_forearm": [reach_x, center_y - half_y, contact_z],
            },
            "hold": {
                "left_forearm": [reach_x - 0.03, center_y + half_y, contact_z + 0.12],
                "right_forearm": [reach_x - 0.03, center_y - half_y, contact_z + 0.12],
            },
        }

    radius = _object_radius(obj)
    hand = str(params.get("hand") or "right")
    if step.name == "manip.top_grasp":
        contact_z = obj_pos[2] + max(radius * 0.5, 0.02)
        approach_z = contact_z + max(0.12, radius * 2.0)
    elif step.name == "manip.side_grasp":
        contact_z = obj_pos[2] + max(min(object_height * 0.55, 0.08), radius * 0.45)
        approach_z = max(contact_z + 0.12, clearance_z)
    else:
        contact_z = max(reach_z, obj_pos[2] + min(radius * 0.25, 0.02))
        approach_z = max(contact_z + 0.13, clearance_z)
    return {
        "mode": step.name.removeprefix("manip."),
        "frame_id": "base_link",
        "hand": hand,
        "object_id": obj.object_id,
        "pregrasp": {
            "palm": [reach_x - max(0.05, radius), target_y, approach_z],
            "finger_aperture": _open_aperture(step, obj),
        },
        "contact": {
            "palm": [reach_x, target_y, contact_z],
            "finger_aperture": _contact_aperture(step, obj),
        },
        "hold": {
            "palm": [reach_x - 0.04, target_y, contact_z + max(0.08, radius * 1.5)],
            "finger_aperture": _secure_aperture(step, obj),
        },
    }


def _wrist_target(step: SkillSpec | None, obj: WorldObject | None) -> dict[str, Any] | None:
    if step is None or obj is None:
        return None
    hand = str(step.params.get("hand") or "right")
    if step.name == "manip.bimanual_clamp":
        return {
            "frame_id": "base_link",
            "mode": "forearm_clamp",
            "left": {"roll": 0.0, "pitch": -0.35, "yaw": 0.0},
            "right": {"roll": 0.0, "pitch": -0.35, "yaw": 0.0},
        }
    if step.name == "manip.top_grasp":
        pitch = -1.05
        roll = 0.0
    elif step.name == "manip.side_grasp":
        pitch = -0.48
        roll = -0.20 if hand == "right" else 0.20
    else:
        pitch = -0.58
        roll = -0.12 if hand == "right" else 0.12
    return {
        "frame_id": "base_link",
        "hand": hand,
        "roll": roll,
        "pitch": pitch,
        "yaw": 0.0,
        "orientation_hint": step.name.removeprefix("manip."),
    }


def _grasp_close_ratio(step: SkillSpec | None, obj: WorldObject | None) -> dict[str, Any] | None:
    if step is None or obj is None:
        return None
    if step.name == "manip.bimanual_clamp":
        open_y = _finite(step.params.get("open_y"), max(_object_half_y(obj) * 2.0 + 0.12, 0.24))
        clamp_y = _finite(step.params.get("clamp_y"), max(_object_half_y(obj) * 2.0 + 0.02, 0.10))
        return {
            "mode": "bimanual_clamp",
            "open_y": open_y,
            "secure_y": clamp_y,
            "close_ratio": _clamp(1.0 - clamp_y / max(open_y, 1e-6), 0.05, 0.95),
        }
    open_aperture = _open_aperture(step, obj)
    secure_aperture = _secure_aperture(step, obj)
    return {
        "mode": step.name.removeprefix("manip."),
        "open_aperture": open_aperture,
        "contact_aperture": _contact_aperture(step, obj),
        "secure_aperture": secure_aperture,
        "close_ratio": _clamp(1.0 - secure_aperture / max(open_aperture, 1e-6), 0.05, 0.95),
    }


def _grasp_offsets(
    step: SkillSpec | None,
    obj: WorldObject | None,
    hand_pose: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if step is None or obj is None or not hand_pose:
        return None
    obj_pos = _pose_position(obj.pose_base, default=(0.0, 0.0, 0.0))
    contact = hand_pose.get("contact") or {}
    if step.name == "manip.bimanual_clamp":
        left = _list3(contact.get("left_forearm") or obj_pos)
        right = _list3(contact.get("right_forearm") or obj_pos)
        center = [(left[i] + right[i]) * 0.5 for i in range(3)]
    else:
        center = _list3(contact.get("palm") or obj_pos)
    return {
        "frame_id": "base_link",
        "object_id": obj.object_id,
        "contact_offset": [center[0] - obj_pos[0], center[1] - obj_pos[1], center[2] - obj_pos[2]],
        "pregrasp_offset": [-0.08, 0.0, 0.12],
        "retreat_offset": [-0.06, 0.0, 0.10],
        "computed_from": "object_pose_base_and_shape",
    }


def _lift_place_targets(
    result: PlanningResult,
    obj: WorldObject | None,
    target: WorldObject | None,
    hand_pose: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if obj is None:
        return None
    obj_pos = _pose_position(obj.pose_base, default=(0.52, 0.0, 0.04))
    lift_delta = 0.14
    for step in result.skill_graph.steps:
        if step.name == "manip.lift_object":
            lift_delta = _finite(step.params.get("lift_z"), lift_delta)
    contact = (hand_pose or {}).get("contact") or {}
    if "palm" in contact:
        base_point = _list3(contact["palm"])
    elif "left_forearm" in contact and "right_forearm" in contact:
        left = _list3(contact["left_forearm"])
        right = _list3(contact["right_forearm"])
        base_point = [(left[i] + right[i]) * 0.5 for i in range(3)]
    else:
        base_point = list(obj_pos)
    payload: dict[str, Any] = {
        "frame_id": "base_link",
        "object_id": obj.object_id,
        "lift_pose_base": [base_point[0], base_point[1], base_point[2] + lift_delta],
        "carry_pose_base": [base_point[0] - 0.06, base_point[1], base_point[2] + lift_delta],
    }
    if target is not None:
        payload["target_id"] = target.object_id
        payload["place_pose_base"] = target.pose_base.to_dict() if target.pose_base else None
        payload["place_pose_map"] = target.pose_map.to_dict() if target.pose_map else None
        payload["place_support"] = target.support
    return payload


def _recovery_decision(result: PlanningResult) -> dict[str, Any] | None:
    if not result.recovery_plan.actions:
        return {"status": "none", "handler": None, "reason": "plan_ready"}
    action = sorted(result.recovery_plan.actions, key=lambda item: (item.priority, item.action_id))[0]
    return {
        "status": "needs_recovery",
        "action_id": action.action_id,
        "handler": action.handler,
        "target_id": action.target_id,
        "command": action.command,
        "reason": action.suggestion,
        "failed_checks": list(action.failed_checks),
    }


def _compact_decision_action(action: DecisionAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "handler": action.handler,
        "target_id": action.target_id,
        "source_id": action.source_id,
        "priority": action.priority,
        "command": action.command,
        "reason": action.reason,
    }


def _compact_object(obj: WorldObject) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "category": obj.category,
        "shape": obj.shape.to_dict(),
        "pose_map": obj.pose_map.to_dict() if obj.pose_map else None,
        "pose_base": obj.pose_base.to_dict() if obj.pose_base else None,
        "pose_camera": obj.pose_camera.to_dict() if obj.pose_camera else None,
        "support": obj.support,
        "source": obj.source,
        "affordances": [aff.to_dict() for aff in obj.affordances],
        "properties": {
            key: value
            for key, value in obj.properties.items()
            if key in {"scene", "raw_anchor_kind", "pixel", "grasp"}
        },
    }


def _pose_position(pose: Pose3 | None, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if pose is None:
        return default
    return pose.position


def _object_radius(obj: WorldObject) -> float:
    if obj.shape.radius is not None and math.isfinite(obj.shape.radius):
        return max(0.005, float(obj.shape.radius))
    if obj.shape.size is not None:
        sx, sy, sz = obj.shape.size
        return max(0.005, min(abs(float(sx)), abs(float(sy)), abs(float(sz))) * 0.5)
    return 0.045


def _object_half_y(obj: WorldObject) -> float:
    if obj.shape.size is not None:
        return max(0.005, abs(float(obj.shape.size[1])) * 0.5)
    return _object_radius(obj)


def _object_height(obj: WorldObject) -> float:
    if obj.shape.size is not None:
        return max(0.005, abs(float(obj.shape.size[2])))
    return _object_radius(obj) * 2.0


def _open_aperture(step: SkillSpec, obj: WorldObject) -> float:
    radius = _object_radius(obj)
    if step.name == "manip.top_grasp":
        return max(_finite(step.params.get("aperture"), radius * 2.4), radius * 2.2 + 0.012)
    return max(radius * 2.4 + 0.018, 0.07)


def _contact_aperture(step: SkillSpec, obj: WorldObject) -> float:
    radius = _object_radius(obj)
    if step.name == "manip.top_grasp":
        return max(radius * 2.0 + 0.006, 0.045)
    return max(radius * 2.0 + 0.008, 0.045)


def _secure_aperture(step: SkillSpec, obj: WorldObject) -> float:
    radius = _object_radius(obj)
    if step.name == "manip.top_grasp":
        return max(radius * 1.8, 0.035)
    return max(radius * 1.75, 0.035)


def _list3(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [_finite(value[0], math.nan), _finite(value[1], math.nan), _finite(value[2], math.nan)]
    return [math.nan, math.nan, math.nan]


def _optional_float(value: Any) -> float | None:
    number = _finite(value, math.nan)
    return number if math.isfinite(number) else None


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
