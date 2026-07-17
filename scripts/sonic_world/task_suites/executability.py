from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CURRENT_BALL_AFFORDANCES = {"single_hand_pinch", "side_grasp", "top_grasp"}
CURRENT_BALL_CATEGORIES = {
    "apple",
    "ball",
    "book",
    "bottle",
    "bowl",
    "can",
    "cloth",
    "cup",
    "fruit",
    "mug",
    "orange",
    "plate",
    "remote",
    "sponge",
    "tool",
    "utensil",
}
CURRENT_BOX_AFFORDANCES = {"bimanual_clamp"}
CURRENT_BOX_CATEGORIES = {"box", "cube", "package", "small_box", "small_package", "snack_box"}


@dataclass(frozen=True)
class TaskExecutability:
    task_id: str
    demo_kind: str
    category: str
    affordance: str
    executable: bool
    reason: str
    pose_base: tuple[float, float, float] | None
    execution_object_y: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "demo_kind": self.demo_kind,
            "category": self.category,
            "affordance": self.affordance,
            "executable": self.executable,
            "reason": self.reason,
            "pose_base": list(self.pose_base) if self.pose_base is not None else None,
            "execution_object_y": self.execution_object_y,
        }


def task_executability(task: Any, *, tier: str = "current") -> TaskExecutability:
    """Classify whether a task fits the currently wired Sonic rollout primitives.

    This is intentionally a task/skill-layer gate. It does not change SONIC low-level
    control and it does not remove tasks from the benchmark suite; it only prevents
    the current primitive runners from collecting misleading failures on tasks whose
    affordance is not implemented yet.
    """

    task_id = str(getattr(task, "task_id", "") or "")
    expectation = task.expectation if isinstance(getattr(task, "expectation", None), dict) else {}
    request = getattr(task, "request", None)
    object_id = str(getattr(request, "object_id", "") or _request_value(request, "object") or "")
    target = _object_by_id(getattr(task, "objects", ()), object_id)
    category = str((target or {}).get("category") or _category_from_tags(getattr(task, "tags", ())) or "unknown")
    affordance = str(
        expectation.get("grasp_affordance")
        or _request_metadata_value(request, "preferred_grasp_affordance")
        or "unknown"
    )
    demo_kind = str(expectation.get("demo_kind") or ("box" if affordance == "bimanual_clamp" else "ball"))
    pose_base = _pose_position((target or {}).get("pose_base"))
    execution_y = _finite_or_none((getattr(task, "metadata", {}) or {}).get("execution_object_y"))
    if execution_y is None and pose_base is not None:
        execution_y = pose_base[1]

    if tier in {"all", "any", "benchmark"}:
        return TaskExecutability(task_id, demo_kind, category, affordance, True, "benchmark_pool", pose_base, execution_y)
    if tier != "current":
        return TaskExecutability(task_id, demo_kind, category, affordance, False, f"unknown_tier:{tier}", pose_base, execution_y)

    reason = _current_reason(demo_kind, affordance, category, pose_base, execution_y)
    return TaskExecutability(
        task_id=task_id,
        demo_kind=demo_kind,
        category=category,
        affordance=affordance,
        executable=reason == "ready",
        reason=reason,
        pose_base=pose_base,
        execution_object_y=execution_y,
    )


def _current_reason(
    demo_kind: str,
    affordance: str,
    category: str,
    pose_base: tuple[float, float, float] | None,
    execution_y: float | None,
) -> str:
    if demo_kind == "ball":
        if affordance not in CURRENT_BALL_AFFORDANCES:
            return f"unsupported_affordance:{affordance}"
        if category not in CURRENT_BALL_CATEGORIES:
            return f"unsupported_category:{category}"
        if pose_base is None:
            return "missing_pose_base"
        x, y, z = pose_base
        y_check = execution_y if execution_y is not None else y
        if not (0.34 <= x <= 0.74):
            return "object_x_out_of_current_range"
        if not (-0.44 <= y_check <= -0.04):
            return "object_y_out_of_current_right_hand_range"
        if not (-0.04 <= z <= 0.16):
            return "object_z_out_of_current_range"
        return "ready"

    if demo_kind == "box":
        if affordance not in CURRENT_BOX_AFFORDANCES:
            return f"unsupported_affordance:{affordance}"
        if category not in CURRENT_BOX_CATEGORIES:
            return f"unsupported_category:{category}"
        if pose_base is None:
            return "missing_pose_base"
        x, y, z = pose_base
        y_check = execution_y if execution_y is not None else y
        if not (0.32 <= x <= 0.78):
            return "object_x_out_of_current_range"
        if abs(y_check) > 0.36:
            return "object_y_out_of_current_bimanual_range"
        if not (-0.04 <= z <= 0.18):
            return "object_z_out_of_current_range"
        return "ready"

    return f"unsupported_demo:{demo_kind}"


def _object_by_id(objects: Any, object_id: str) -> dict[str, Any] | None:
    for obj in objects or ():
        if isinstance(obj, dict) and str(obj.get("object_id") or obj.get("id") or "") == object_id:
            return obj
    return None


def _pose_position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, dict):
        return None
    position = value.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        xyz = tuple(float(item) for item in position[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in xyz):
        return None
    return xyz  # type: ignore[return-value]


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _request_value(request: Any, key: str) -> Any:
    if isinstance(request, dict):
        return request.get(key)
    return getattr(request, key, None)


def _request_metadata_value(request: Any, key: str) -> Any:
    metadata = request.get("metadata") if isinstance(request, dict) else getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _category_from_tags(tags: Any) -> str:
    skip = {
        "move",
        "pick",
        "place",
        "short",
        "tabletop",
        "clutter",
        "dense",
        "generated",
        "generated_sequence",
        "long_sequence",
        "medium_horizon",
        "navigation_manipulation",
        "single_hand_pinch",
        "side_grasp",
        "top_grasp",
        "bimanual_clamp",
    }
    for tag in tags or ():
        text = str(tag)
        if text not in skip:
            return text
    return ""
