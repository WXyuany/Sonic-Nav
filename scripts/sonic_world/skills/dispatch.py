from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import CapabilityContract, capability_contract_for_step
from .runtime import RuntimePlan
from .specs import SkillGraph, SkillSpec
from ..world_model.entities import WorldObject, WorldState


@dataclass(frozen=True)
class DispatchStep:
    skill_name: str
    target_id: str | None
    handler: str
    capability: str
    command: dict[str, Any]
    phase_names: tuple[str, ...] = ()
    monitor_events: tuple[str, ...] = ()
    recovery_events: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    readiness: str = "ready"
    notes: tuple[str, ...] = ()
    contract: CapabilityContract | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "target_id": self.target_id,
            "handler": self.handler,
            "capability": self.capability,
            "command": self.command,
            "phase_names": list(self.phase_names),
            "monitor_events": list(self.monitor_events),
            "recovery_events": list(self.recovery_events),
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "readiness": self.readiness,
            "notes": list(self.notes),
            "contract": self.contract.to_dict() if self.contract else None,
        }


@dataclass(frozen=True)
class DispatchPlan:
    task_id: str
    objective: str
    steps: list[DispatchStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }


def dispatch_plan_for_graph(
    graph: SkillGraph,
    runtime: RuntimePlan,
    world: WorldState,
) -> DispatchPlan:
    bindings = {binding.skill_name: binding for binding in runtime.bindings}
    steps: list[DispatchStep] = []
    for step in graph.steps:
        binding = bindings.get(step.name)
        target = world.get_object(step.target_id) if step.target_id else None
        command = _command_for_step(step, target, runtime, world)
        handler, capability = _handler_for_step(step, runtime)
        contract = capability_contract_for_step(
            step,
            target=target,
            world=world,
            runtime=runtime,
            binding=binding,
            handler=handler,
            capability=capability,
            command=command,
        )
        readiness, notes = _readiness_for_step(step, target, runtime, contract)
        steps.append(
            DispatchStep(
                skill_name=step.name,
                target_id=step.target_id,
                handler=handler,
                capability=capability,
                command=command,
                phase_names=binding.phase_names if binding else (),
                monitor_events=binding.monitor_events if binding else (),
                recovery_events=binding.recovery_events if binding else tuple(step.recovery),
                preconditions=tuple(step.preconditions),
                effects=tuple(step.effects),
                readiness=readiness,
                notes=notes,
                contract=contract,
            )
        )
    return DispatchPlan(
        task_id=graph.task_id,
        objective=graph.objective,
        steps=steps,
        metadata={
            "demo_kind": runtime.demo_kind,
            "handler_count": len({step.handler for step in steps}),
            "unready_count": sum(1 for step in steps if step.readiness != "ready"),
            "contract_error_count": sum(len(step.contract.failed_errors) for step in steps if step.contract),
            "contract_warning_count": sum(len(step.contract.failed_warnings) for step in steps if step.contract),
            "recovery_suggestions": _recovery_suggestions(steps),
        },
    )


def _handler_for_step(step: SkillSpec, runtime: RuntimePlan) -> tuple[str, str]:
    if step.name == "navigate.goto":
        return "ros2_goal_pose", "navigation"
    if step.name == "navigate.approach_object":
        if runtime.demo_kind in {"box", "ball"}:
            return "demo_locomotion_phase_runtime", "navigation"
        return "ros2_goal_pose", "navigation"
    if step.name == "manip.align_workspace":
        return "workspace_alignment_primitive", "workspace_alignment"
    if step.name in {"manip.bimanual_clamp", "manip.single_hand_pinch", "manip.side_grasp", "manip.top_grasp"}:
        return "contact_grasp_primitive", "contact_grasp"
    if step.name == "manip.lift_object":
        return "lift_stability_primitive", "object_lift"
    if step.name == "manip.transport_object":
        return "carry_transfer_primitive", "object_transport"
    if step.name == "manip.place_object":
        return "place_primitive", "object_place"
    if step.name == "manip.release":
        return "release_primitive", "hand_release"
    return "unassigned", "unknown"


def _command_for_step(
    step: SkillSpec,
    target: WorldObject | None,
    runtime: RuntimePlan,
    world: WorldState,
) -> dict[str, Any]:
    if step.name == "navigate.goto":
        return {
            "type": "publish_pose_stamped",
            "topic": "/goal_pose",
            "pose": _pose_for_step(step, target),
            "target_ref": _entity_ref(target),
        }
    if step.name == "navigate.approach_object":
        command_type = "runtime_phase_sequence" if runtime.demo_kind in {"box", "ball"} else "publish_pose_stamped"
        payload: dict[str, Any] = {
            "type": command_type,
            "object_id": step.target_id,
            "approach_pose": _pose_for_step(step, target),
            "target_ref": _entity_ref(target),
        }
        if "standoff" in step.params:
            payload["standoff"] = step.params["standoff"]
        return payload
    if step.name.startswith("manip."):
        params = dict(step.params)
        destination_id = params.get("destination_id")
        destination = world.get_object(str(destination_id)) if destination_id else None
        return {
            "type": "runtime_phase_sequence",
            "object_id": step.target_id,
            "params": params,
            "target_ref": _entity_ref(target),
            "destination_ref": _entity_ref(destination),
        }
    return {"type": "manual_review", "params": step.params}


def _entity_ref(obj: WorldObject | None) -> dict[str, Any] | None:
    if obj is None:
        return None
    properties = obj.properties if isinstance(obj.properties, dict) else {}
    object_id = str(obj.object_id)
    return {
        "object_id": object_id,
        "category": str(obj.category),
        "anchor_id": str(properties.get("anchor_id") or object_id),
        "policy_object_id": str(properties.get("policy_object_id") or object_id),
        "body_name": str(properties.get("body_name") or object_id),
        "joint_name": str(properties.get("joint_name") or f"{object_id}_freejoint"),
        "geom_name": str(properties.get("geom_name") or f"{object_id}_geom"),
        "site_name": str(properties.get("site_name") or f"{object_id}_site"),
        "support_id": obj.support,
    }


def _pose_for_step(step: SkillSpec, target: WorldObject | None) -> dict[str, Any] | None:
    pose = step.params.get("pose")
    if isinstance(pose, dict):
        return pose
    if target is not None:
        if target.pose_map is not None:
            return target.pose_map.to_dict()
        if target.pose_base is not None:
            return target.pose_base.to_dict()
    return None


def _readiness_for_step(
    step: SkillSpec,
    target: WorldObject | None,
    runtime: RuntimePlan,
    contract: CapabilityContract,
) -> tuple[str, tuple[str, ...]]:
    errors: list[str] = []
    warnings: list[str] = []
    if step.target_id and target is None:
        errors.append("target_missing_from_world")
    if step.name in {"navigate.goto", "navigate.approach_object"} and _pose_for_step(step, target) is None:
        errors.append("missing_navigation_pose")
    if step.name.startswith("manip.") and runtime.demo_kind == "navigation":
        errors.append("manipulation_skill_without_demo_runtime")
    if step.name not in {binding.skill_name for binding in runtime.bindings}:
        errors.append("runtime_binding_missing")
    errors.extend(contract.failed_errors)
    warnings.extend(f"warning:{name}" for name in contract.failed_warnings)
    return ("ready" if not errors else "needs_attention", tuple((*errors, *warnings)))


def _recovery_suggestions(steps: list[DispatchStep]) -> list[str]:
    suggestions: list[str] = []
    for step in steps:
        if step.contract is None:
            continue
        suggestions.extend(step.contract.recovery_suggestions)
    return list(dict.fromkeys(suggestions))
