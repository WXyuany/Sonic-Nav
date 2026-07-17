from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from ..world_model import WorldState, anchor_to_world


@dataclass(frozen=True)
class OracleCheck:
    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class TaskOracleResult:
    task_id: str
    success: bool
    checks: tuple[OracleCheck, ...]
    final_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sonic_task_success_oracle_v0",
            "task_id": self.task_id,
            "success": bool(self.success),
            "final_status": self.final_status,
            "checks": [check.to_dict() for check in self.checks],
            "metadata": self.metadata,
        }


def evaluate_task_success(
    task: Any,
    *,
    initial_world: WorldState | dict[str, Any] | None = None,
    final_world: WorldState | dict[str, Any] | None = None,
    world_history: list[WorldState | dict[str, Any]] | tuple[WorldState | dict[str, Any], ...] | None = None,
    rollout_summary: dict[str, Any] | None = None,
    place_tolerance_m: float = 0.14,
    lift_delta_m: float = 0.055,
    drop_z_margin_m: float = 0.12,
) -> TaskOracleResult:
    initial = _coerce_world(initial_world) or _coerce_world(_safe_call(task, "anchor"))
    final = _coerce_world(final_world)
    history = [world for item in (world_history or ()) if (world := _coerce_world(item)) is not None]
    summary = rollout_summary if isinstance(rollout_summary, dict) else {}
    task_id = str(getattr(task, "task_id", "") or _request_value(getattr(task, "request", None), "id") or "task")
    request = getattr(task, "request", None)
    verb = str(getattr(request, "verb", "") or _request_value(request, "task") or "")
    object_id = str(getattr(request, "object_id", "") or _request_value(request, "object") or "")
    target_id = str(getattr(request, "target_id", "") or _request_value(request, "target") or "")
    checks: list[OracleCheck] = []

    if final is None:
        checks.append(
            _check(
                "final_world.available",
                False,
                "no final world/anchor was provided; using rollout summary only",
                severity="warning",
            )
        )
        checks.extend(_summary_checks(summary, verb=verb))
        success = _summary_success(summary)
        return TaskOracleResult(
            task_id=task_id,
            success=success,
            checks=tuple(checks),
            final_status="success" if success else str(summary.get("final_status") or "unknown"),
            metadata={"mode": "summary_only", "verb": verb, "object_id": object_id, "target_id": target_id},
        )

    obj = final.get_object(object_id) if object_id else final.primary_object()
    initial_obj = initial.get_object(object_id) if initial is not None and object_id else None
    checks.append(_check("object.exists", obj is not None, object_id))
    if obj is not None:
        checks.append(_not_dropped_check(obj, initial_obj, margin=drop_z_margin_m))
        if history:
            checks.append(_trajectory_not_dropped_check(history, object_id, initial_obj, margin=drop_z_margin_m))
    checks.append(_check("robot.stable", bool(final.robot.stable), str(final.robot.stable)))

    if verb in {"pick_place", "move", "place"}:
        target = final.get_object(target_id) if target_id else _first_target(final)
        checks.append(_check("target.exists", target is not None, target_id or "place_target"))
        if obj is not None and target is not None:
            checks.append(_placed_at_target_check(obj, target, tolerance=place_tolerance_m))
            checks.append(_support_check(final, obj.object_id, target.support))
            checks.append(
                _ever_lifted_check(
                    history,
                    object_id,
                    initial_obj,
                    summary,
                    lift_delta_m=lift_delta_m,
                )
            )
            if history:
                checks.append(_target_stability_check(history, object_id, target, tolerance=place_tolerance_m))
    elif verb in {"pick", "grasp"}:
        if obj is not None:
            checks.append(_lifted_check(obj, initial_obj, summary, lift_delta_m=lift_delta_m))
    elif verb in {"navigate", "go_to", "goto"}:
        target = obj
        checks.append(_navigation_check(final, target, summary))
    else:
        checks.extend(_summary_checks(summary, verb=verb))

    checks.extend(_constraint_checks(final))
    failed_errors = [check for check in checks if not check.passed and check.severity == "error"]
    summary_failed = summary.get("final_status") == "failed"
    success = not failed_errors and not summary_failed
    return TaskOracleResult(
        task_id=task_id,
        success=success,
        checks=tuple(checks),
        final_status="success" if success else "failed",
        metadata={
            "mode": "world_geometry",
            "verb": verb,
            "object_id": object_id,
            "target_id": target_id,
            "summary_final_status": summary.get("final_status"),
            "history_samples": len(history),
        },
    )


def _coerce_world(value: Any) -> WorldState | None:
    if value is None:
        return None
    if isinstance(value, WorldState):
        return value
    if isinstance(value, dict):
        try:
            return anchor_to_world(value)
        except Exception:
            world = value.get("world") if isinstance(value.get("world"), dict) else None
            if world:
                return _world_from_payload(world)
    return None


def _world_from_payload(payload: dict[str, Any]) -> WorldState | None:
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        return None
    anchor_objects = []
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        record = dict(obj)
        record.setdefault("object_id", object_id)
        anchor_objects.append(record)
    return anchor_to_world(
        {
            "frame_id": payload.get("frame_id", "map"),
            "objects": anchor_objects,
            "relations": payload.get("relations") or [],
            "properties": payload.get("properties") or {},
        }
    )


def _summary_checks(summary: dict[str, Any], *, verb: str) -> list[OracleCheck]:
    if not summary:
        return [_check("rollout_summary.available", False, "missing rollout summary", severity="warning")]
    checks = [_check("rollout.final_status", summary.get("final_status") == "success", str(summary.get("final_status")))]
    if verb in {"pick", "grasp", "pick_place", "move", "place"}:
        checks.append(_check("rollout.lift_success", bool(summary.get("lift_success")), str(summary.get("lift_success"))))
    return checks


def _summary_success(summary: dict[str, Any]) -> bool:
    return bool(summary) and summary.get("final_status") == "success"


def _not_dropped_check(obj: Any, initial_obj: Any, *, margin: float) -> OracleCheck:
    pose = obj.pose_map or obj.pose_base
    initial_pose = initial_obj.pose_map or initial_obj.pose_base if initial_obj is not None else None
    if pose is None:
        return _check("object.not_dropped", False, "missing final object pose")
    floor = -margin
    if initial_pose is not None:
        floor = float(initial_pose.position[2]) - margin
    passed = float(pose.position[2]) >= floor
    return _check(
        "object.not_dropped",
        passed,
        f"final_z={pose.position[2]:.3f} floor={floor:.3f}",
        metrics={"final_z": pose.position[2], "floor_z": floor},
    )


def _placed_at_target_check(obj: Any, target: Any, *, tolerance: float) -> OracleCheck:
    obj_pose = obj.pose_map or obj.pose_base
    target_pose = target.pose_map or target.pose_base
    if obj_pose is None or target_pose is None:
        return _check("object.at_target", False, "missing object or target pose")
    distance = math.hypot(
        float(obj_pose.position[0]) - float(target_pose.position[0]),
        float(obj_pose.position[1]) - float(target_pose.position[1]),
    )
    return _check(
        "object.at_target",
        distance <= tolerance,
        f"xy_distance={distance:.3f} tolerance={tolerance:.3f}",
        metrics={"xy_distance": distance, "tolerance": tolerance},
    )


def _support_check(world: WorldState, object_id: str, expected_support: str | None) -> OracleCheck:
    if not expected_support:
        return _check("object.support", True, "no expected support", severity="warning")
    for relation in world.relations:
        if relation.subject_id == object_id and relation.relation == "on" and relation.object_id == expected_support:
            return _check("object.support", True, f"on {expected_support}")
    return _check("object.support", False, f"missing on relation to {expected_support}")


def _ever_lifted_check(
    history: list[WorldState],
    object_id: str,
    initial_obj: Any,
    summary: dict[str, Any],
    *,
    lift_delta_m: float,
) -> OracleCheck:
    if summary.get("lift_success"):
        return _check("object.ever_lifted", True, "rollout lift_check success")
    initial_pose = initial_obj.pose_map or initial_obj.pose_base if initial_obj is not None else None
    if initial_pose is None or not history:
        return _check("object.ever_lifted", False, "missing trajectory or initial object pose")
    peak_delta = -math.inf
    for world in history:
        obj = world.get_object(object_id)
        pose = (obj.pose_map or obj.pose_base) if obj is not None else None
        if pose is not None:
            peak_delta = max(peak_delta, float(pose.position[2]) - float(initial_pose.position[2]))
    passed = math.isfinite(peak_delta) and peak_delta >= lift_delta_m
    return _check(
        "object.ever_lifted",
        passed,
        f"peak_dz={peak_delta:.3f} threshold={lift_delta_m:.3f}" if math.isfinite(peak_delta) else "no object samples",
        metrics={"peak_z_delta": peak_delta if math.isfinite(peak_delta) else None, "threshold": lift_delta_m},
    )


def _trajectory_not_dropped_check(
    history: list[WorldState],
    object_id: str,
    initial_obj: Any,
    *,
    margin: float,
) -> OracleCheck:
    initial_pose = initial_obj.pose_map or initial_obj.pose_base if initial_obj is not None else None
    floor = float(initial_pose.position[2]) - margin if initial_pose is not None else -margin
    minimum = math.inf
    for world in history:
        obj = world.get_object(object_id)
        pose = (obj.pose_map or obj.pose_base) if obj is not None else None
        if pose is not None:
            minimum = min(minimum, float(pose.position[2]))
    passed = math.isfinite(minimum) and minimum >= floor
    return _check(
        "object.never_dropped",
        passed,
        f"min_z={minimum:.3f} floor={floor:.3f}" if math.isfinite(minimum) else "no object samples",
        metrics={"minimum_z": minimum if math.isfinite(minimum) else None, "floor_z": floor},
    )


def _target_stability_check(
    history: list[WorldState],
    object_id: str,
    target: Any,
    *,
    tolerance: float,
    required_samples: int = 3,
) -> OracleCheck:
    target_pose = target.pose_map or target.pose_base
    if target_pose is None:
        return _check("object.stable_at_target", False, "target pose missing")
    tail = history[-required_samples:]
    distances: list[float] = []
    for world in tail:
        obj = world.get_object(object_id)
        pose = (obj.pose_map or obj.pose_base) if obj is not None else None
        if pose is None:
            continue
        distances.append(math.hypot(pose.position[0] - target_pose.position[0], pose.position[1] - target_pose.position[1]))
    passed = len(distances) >= required_samples and all(distance <= tolerance for distance in distances)
    return _check(
        "object.stable_at_target",
        passed,
        f"samples={len(distances)}/{required_samples} max_distance={max(distances):.3f}"
        if distances
        else "no target stability samples",
        metrics={"samples": len(distances), "required_samples": required_samples, "max_distance": max(distances) if distances else None},
    )


def _lifted_check(obj: Any, initial_obj: Any, summary: dict[str, Any], *, lift_delta_m: float) -> OracleCheck:
    if summary.get("lift_success"):
        return _check("object.lifted", True, "rollout lift_check success")
    pose = obj.pose_map or obj.pose_base
    initial_pose = initial_obj.pose_map or initial_obj.pose_base if initial_obj is not None else None
    if pose is None or initial_pose is None:
        return _check("object.lifted", False, "missing initial or final pose", severity="warning")
    dz = float(pose.position[2]) - float(initial_pose.position[2])
    return _check(
        "object.lifted",
        dz >= lift_delta_m,
        f"dz={dz:.3f} threshold={lift_delta_m:.3f}",
        metrics={"z_delta": dz, "threshold": lift_delta_m},
    )


def _navigation_check(final: WorldState, target: Any, summary: dict[str, Any]) -> OracleCheck:
    if summary.get("final_status") == "success":
        return _check("navigation.reached", True, "rollout final status success")
    if target is None or target.pose_map is None or final.robot.base_map is None:
        return _check("navigation.reached", False, "missing base or target map pose", severity="warning")
    distance = math.hypot(
        float(final.robot.base_map.position[0]) - float(target.pose_map.position[0]),
        float(final.robot.base_map.position[1]) - float(target.pose_map.position[1]),
    )
    tolerance = float(target.properties.get("goal_tolerance") or 0.45)
    return _check(
        "navigation.reached",
        distance <= tolerance,
        f"xy_distance={distance:.3f} tolerance={tolerance:.3f}",
        metrics={"xy_distance": distance, "tolerance": tolerance},
    )


def _constraint_checks(world: WorldState) -> list[OracleCheck]:
    collisions = world.properties.get("collisions")
    if not collisions:
        return []
    if isinstance(collisions, list):
        violations = [item for item in collisions if _collision_is_violation(item)]
        return [_check("constraints.no_forbidden_collision", not violations, f"violations={len(violations)}")]
    return []


def _collision_is_violation(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    if item.get("allowed") is True:
        return False
    return bool(item.get("forbidden", True))


def _first_target(world: WorldState) -> Any:
    for obj in world.objects.values():
        if obj.category in {"place_target", "navigation_goal"}:
            return obj
    return None


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = "error",
    metrics: dict[str, Any] | None = None,
) -> OracleCheck:
    return OracleCheck(name=name, passed=bool(passed), detail=detail, severity=severity, metrics=metrics or {})


def _request_value(request: Any, key: str) -> Any:
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _safe_call(value: Any, name: str) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None
