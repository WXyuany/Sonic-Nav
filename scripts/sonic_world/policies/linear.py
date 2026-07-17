from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import numpy as np

from .heuristic import HeuristicSkillPolicy
from .schema import PolicyAction
from ..planners import PlanningResult


TARGET_PATHS = (
    "base_goal.position.0",
    "base_goal.position.1",
    "hand_pose_target.contact.palm.0",
    "hand_pose_target.contact.palm.1",
    "hand_pose_target.contact.palm.2",
    "wrist_target.pitch",
    "grasp_close_ratio.close_ratio",
    "lift_place_targets.lift_pose_base.0",
    "lift_place_targets.lift_pose_base.1",
    "lift_place_targets.lift_pose_base.2",
)


class LinearTaskPolicyBackend:
    """Dependency-light learned task-space regressor with heuristic structural fallback."""

    def __init__(self, model: dict[str, Any]) -> None:
        if model.get("schema") != "sonic_linear_task_policy_v0":
            raise ValueError(f"unsupported linear policy schema {model.get('schema')!r}")
        self.model = model
        self.policy_id = str(model.get("policy_id") or model.get("model_id") or "linear_task_policy")
        self.feature_names = [str(item) for item in model.get("feature_names") or []]
        self.targets = model.get("targets") if isinstance(model.get("targets"), dict) else {}
        self.teacher = HeuristicSkillPolicy()
        if not self.feature_names or not self.targets:
            raise ValueError("linear policy requires feature_names and targets")

    def act(self, result: PlanningResult) -> PolicyAction:
        base = self.teacher.act(result)
        payload = base.to_dict()
        features = feature_vector(result, self.feature_names)
        predicted: dict[str, float] = {}
        for path, record in self.targets.items():
            if not isinstance(record, dict):
                continue
            weights = np.asarray(record.get("weights") or [], dtype=np.float64)
            if len(weights) != len(features):
                continue
            value = float(np.dot(features, weights))
            lower = _finite(record.get("min"), -math.inf)
            upper = _finite(record.get("max"), math.inf)
            value = max(lower, min(upper, value))
            if _set_path(payload, str(path), value):
                predicted[str(path)] = value
        metadata = dict(payload.get("metadata") or {})
        metadata["policy_backend"] = {
            "type": "linear_learned",
            "model_id": self.model.get("model_id"),
            "model_version": self.model.get("version"),
            "checkpoint_hash": (self.model.get("manifest") or {}).get("checkpoint_hash"),
            "predicted_outputs": predicted,
            "structural_fallback": self.teacher.policy_id,
        }
        payload["metadata"] = metadata
        payload["policy_id"] = self.policy_id
        names = {item.name for item in fields(PolicyAction)}
        return PolicyAction(**{name: payload[name] for name in names if name in payload})


def feature_names(categories: list[str] | tuple[str, ...]) -> list[str]:
    return [
        "bias",
        "demo.ball",
        "demo.box",
        "verb.navigate",
        "verb.pick",
        "verb.move",
        "affordance.single_hand_pinch",
        "affordance.side_grasp",
        "affordance.top_grasp",
        "affordance.bimanual_clamp",
        "object.x",
        "object.y",
        "object.z",
        "object.radius",
        "object.size_x",
        "object.size_y",
        "object.size_z",
        "target.present",
        "target.x",
        "target.y",
        "target.z",
        *[f"category.{category}" for category in sorted(set(categories))],
    ]


def feature_vector(result: PlanningResult, names: list[str]) -> np.ndarray:
    request = result.request
    obj = result.world.get_object(request.object_id) if request.object_id else result.world.primary_object()
    target = result.world.get_object(request.target_id) if request.target_id else None
    object_pose = obj.pose_base if obj is not None else None
    target_pose = target.pose_base if target is not None else None
    shape = obj.shape if obj is not None else None
    values: dict[str, float] = {
        "bias": 1.0,
        f"demo.{result.runtime_plan.demo_kind}": 1.0,
        f"verb.{request.verb}": 1.0,
        f"affordance.{result.skill_graph.metadata.get('grasp_affordance')}": 1.0,
        f"category.{obj.category if obj is not None else 'unknown'}": 1.0,
        "target.present": 1.0 if target is not None else 0.0,
    }
    _pose_features(values, "object", object_pose)
    _pose_features(values, "target", target_pose)
    values["object.radius"] = float(shape.radius or 0.0) if shape is not None else 0.0
    size = shape.size if shape is not None else None
    for index, suffix in enumerate(("x", "y", "z")):
        values[f"object.size_{suffix}"] = float(size[index]) if size is not None else 0.0
    return np.asarray([values.get(name, 0.0) for name in names], dtype=np.float64)


def observation_feature_values(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, float | str]:
    intent = action.get("task_intent") if isinstance(action.get("task_intent"), dict) else {}
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    objects = observation.get("objects") if isinstance(observation.get("objects"), list) else []
    object_id = str(intent.get("object_id") or "")
    target_id = str(intent.get("target_id") or "")
    obj = next((item for item in objects if isinstance(item, dict) and str(item.get("object_id")) == object_id), None)
    target = next((item for item in objects if isinstance(item, dict) and str(item.get("object_id")) == target_id), None)
    values: dict[str, float | str] = {
        "demo": str(intent.get("demo_kind") or ""),
        "verb": str(intent.get("verb") or ""),
        "affordance": str(metadata.get("grasp_affordance") or ""),
        "category": str(intent.get("object_category") or (obj or {}).get("category") or "unknown"),
        "target.present": 1.0 if target is not None else 0.0,
    }
    _dict_pose_features(values, "object", (obj or {}).get("pose_base"))
    _dict_pose_features(values, "target", (target or {}).get("pose_base"))
    shape = (obj or {}).get("shape") if isinstance((obj or {}).get("shape"), dict) else {}
    values["object.radius"] = _finite(shape.get("radius"), 0.0)
    size = shape.get("size") if isinstance(shape.get("size"), (list, tuple)) else []
    for index, suffix in enumerate(("x", "y", "z")):
        values[f"object.size_{suffix}"] = _finite(size[index], 0.0) if len(size) > index else 0.0
    return values


def vector_from_values(values: dict[str, float | str], names: list[str]) -> np.ndarray:
    categorical = {
        f"demo.{values.get('demo')}",
        f"verb.{values.get('verb')}",
        f"affordance.{values.get('affordance')}",
        f"category.{values.get('category')}",
    }
    return np.asarray(
        [1.0 if name == "bias" or name in categorical else _finite(values.get(name), 0.0) for name in names],
        dtype=np.float64,
    )


def get_path(payload: dict[str, Any], path: str) -> float | None:
    value: Any = payload
    for token in path.split("."):
        if isinstance(value, dict):
            value = value.get(token)
        elif isinstance(value, (list, tuple)) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _set_path(payload: dict[str, Any], path: str, value: float) -> bool:
    tokens = path.split(".")
    cursor: Any = payload
    for token in tokens[:-1]:
        if isinstance(cursor, dict):
            cursor = cursor.get(token)
        elif isinstance(cursor, list) and token.isdigit() and int(token) < len(cursor):
            cursor = cursor[int(token)]
        else:
            return False
        if cursor is None:
            return False
    last = tokens[-1]
    if isinstance(cursor, dict) and last in cursor:
        cursor[last] = value
        return True
    if isinstance(cursor, list) and last.isdigit() and int(last) < len(cursor):
        cursor[int(last)] = value
        return True
    return False


def _pose_features(values: dict[str, float], prefix: str, pose: Any) -> None:
    position = pose.position if pose is not None else (0.0, 0.0, 0.0)
    values.update({f"{prefix}.x": float(position[0]), f"{prefix}.y": float(position[1]), f"{prefix}.z": float(position[2])})


def _dict_pose_features(values: dict[str, float | str], prefix: str, pose: Any) -> None:
    position = pose.get("position") if isinstance(pose, dict) else None
    for index, suffix in enumerate(("x", "y", "z")):
        values[f"{prefix}.{suffix}"] = _finite(position[index], 0.0) if isinstance(position, (list, tuple)) and len(position) > index else 0.0


def _finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)
