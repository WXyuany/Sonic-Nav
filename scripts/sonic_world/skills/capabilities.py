from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .runtime import PhaseBinding, RuntimePlan
from .specs import SkillSpec
from ..world_model.entities import WorldObject, WorldState


@dataclass(frozen=True)
class CapabilityCheck:
    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CapabilityContract:
    capability: str
    handler: str
    checks: tuple[CapabilityCheck, ...]

    @property
    def ready(self) -> bool:
        return not self.failed_errors

    @property
    def failed_errors(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed and check.severity == "error")

    @property
    def failed_warnings(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed and check.severity == "warning")

    @property
    def recovery_suggestions(self) -> tuple[str, ...]:
        suggestions: list[str] = []
        for name in (*self.failed_errors, *self.failed_warnings):
            suggestions.extend(_recovery_suggestions_for_check(name))
        return tuple(dict.fromkeys(suggestions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "handler": self.handler,
            "ready": self.ready,
            "failed_errors": list(self.failed_errors),
            "failed_warnings": list(self.failed_warnings),
            "recovery_suggestions": list(self.recovery_suggestions),
            "checks": [check.to_dict() for check in self.checks],
        }


def capability_contract_for_step(
    step: SkillSpec,
    *,
    target: WorldObject | None,
    world: WorldState,
    runtime: RuntimePlan,
    binding: PhaseBinding | None,
    handler: str,
    capability: str,
    command: dict[str, Any],
) -> CapabilityContract:
    checks: list[CapabilityCheck] = [
        _check("handler.assigned", handler != "unassigned", handler),
        _check("command.type", isinstance(command.get("type"), str) and bool(command.get("type")), str(command.get("type"))),
    ]
    if step.target_id:
        checks.append(_check("target.exists", target is not None, str(step.target_id)))
    if command.get("type") == "runtime_phase_sequence":
        checks.append(_check("runtime.binding", binding is not None, step.name))
        checks.append(_check("runtime.phase_names", bool(binding and binding.phase_names), step.name))

    if capability == "navigation":
        pose = command.get("pose") or command.get("approach_pose")
        checks.append(_check("navigation.pose", _valid_pose_payload(pose), str(pose)))
    elif capability == "workspace_alignment":
        checks.append(_check("workspace.object_pose_base", target is not None and target.pose_base is not None, step.target_id))
    elif capability == "contact_grasp":
        checks.extend(_contact_grasp_checks(step, target, runtime))
    elif capability == "object_lift":
        checks.append(_check("lift.target_pose_base", target is not None and target.pose_base is not None, step.target_id))
    elif capability == "object_transport":
        params = command.get("params") if isinstance(command.get("params"), dict) else {}
        destination_id = params.get("destination_id")
        destination = world.get_object(str(destination_id)) if destination_id else None
        checks.append(_check("transport.destination_exists", destination is not None, str(destination_id)))
        checks.append(
            _check(
                "transport.destination_pose",
                _valid_pose_payload(params.get("destination_pose_map"))
                or _valid_pose_payload(params.get("destination_pose_base")),
                str(destination_id),
            )
        )
    elif capability == "object_place":
        params = command.get("params") if isinstance(command.get("params"), dict) else {}
        checks.append(
            _check(
                "place.pose",
                _valid_pose_payload(params.get("place_pose_map")) or _valid_pose_payload(params.get("place_pose_base")),
                step.target_id,
            )
        )
        checks.append(_check("place.support", bool(params.get("support")), str(params.get("support")), severity="warning"))
    elif capability == "hand_release":
        checks.append(_check("release.target", target is not None, step.target_id))

    return CapabilityContract(capability=capability, handler=handler, checks=tuple(checks))


def _contact_grasp_checks(
    step: SkillSpec,
    target: WorldObject | None,
    runtime: RuntimePlan,
) -> list[CapabilityCheck]:
    checks = [
        _check("grasp.target_pose_base", target is not None and target.pose_base is not None, step.target_id),
        _check("grasp.params", bool(step.params), step.name),
    ]
    if step.name == "manip.single_hand_pinch":
        checks.append(_check("grasp.radius", _positive(step.params.get("radius")), str(step.params.get("radius"))))
        checks.append(_check("grasp.hand", step.params.get("hand") in {"left", "right"}, str(step.params.get("hand"))))
    if step.name == "manip.side_grasp":
        checks.append(_check("grasp.radius", _positive(step.params.get("radius")), str(step.params.get("radius"))))
        checks.append(_check("grasp.hand", step.params.get("hand") in {"left", "right"}, str(step.params.get("hand"))))
    if step.name == "manip.top_grasp":
        checks.append(_check("grasp.aperture", _positive(step.params.get("aperture")), str(step.params.get("aperture"))))
        checks.append(_check("grasp.hand", step.params.get("hand") in {"left", "right"}, str(step.params.get("hand"))))
    if step.name == "manip.bimanual_clamp":
        checks.append(_check("grasp.open_y", _positive(step.params.get("open_y")), str(step.params.get("open_y"))))
        checks.append(_check("grasp.clamp_y", _positive(step.params.get("clamp_y")), str(step.params.get("clamp_y"))))
        checks.append(_check("grasp.runtime_kind", runtime.demo_kind == "box", runtime.demo_kind))
    return checks


def _valid_pose_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    position = value.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return False
    return all(_finite(item) for item in position[:2])


def _positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _check(name: str, passed: bool, detail: Any = "", *, severity: str = "error") -> CapabilityCheck:
    return CapabilityCheck(name=name, passed=bool(passed), severity=severity, detail=str(detail))


def _recovery_suggestions_for_check(name: str) -> tuple[str, ...]:
    if name == "target.exists":
        return ("refresh_world_memory", "request_object_anchor")
    if name in {
        "workspace.object_pose_base",
        "grasp.target_pose_base",
        "lift.target_pose_base",
    }:
        return (
            "reobserve_from_current_view",
            "publish_object_anchor_with_pose_base",
            "micro_adjust_base_for_observation",
        )
    if name == "navigation.pose":
        return ("estimate_approach_pose", "request_map_pose", "replan_approach_pose")
    if name in {"runtime.binding", "runtime.phase_names"}:
        return ("switch_runtime_template", "add_runtime_phase_binding", "fall_back_to_dry_run")
    if name == "handler.assigned":
        return ("register_skill_handler", "route_to_manual_review")
    if name == "command.type":
        return ("regenerate_dispatch_command",)
    if name == "grasp.params":
        return ("infer_grasp_from_shape", "request_explicit_affordance")
    if name in {"grasp.radius", "grasp.aperture", "grasp.hand", "grasp.open_y", "grasp.clamp_y"}:
        return ("repair_grasp_affordance", "request_explicit_affordance")
    if name == "grasp.runtime_kind":
        return ("switch_runtime_template", "regenerate_runtime_plan")
    if name in {"transport.destination_exists", "transport.destination_pose"}:
        return ("request_place_target_anchor", "choose_nearby_place_region")
    if name == "place.pose":
        return ("request_place_target_anchor", "choose_nearby_place_region")
    if name == "place.support":
        return ("infer_support_surface", "request_support_surface_anchor")
    if name == "release.target":
        return ("refresh_world_memory", "hold_pose")
    return ("manual_review",)
