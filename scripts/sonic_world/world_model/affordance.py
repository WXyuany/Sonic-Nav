from __future__ import annotations

from typing import Any

from .entities import Affordance, Pose3, WorldObject, finite_float, vec3


def _approach_params(grasp: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    base_target = grasp.get("base_target_map")
    if isinstance(base_target, list) and len(base_target) >= 3:
        params["base_target_map"] = [float(base_target[0]), float(base_target[1]), float(base_target[2])]
    if "approach_target_x" in grasp:
        params["standoff"] = finite_float(grasp.get("approach_target_x"))
    if "walk_speed" in grasp:
        params["walk_speed"] = finite_float(grasp.get("walk_speed"))
    if "walk_duration" in grasp:
        params["walk_duration"] = finite_float(grasp.get("walk_duration"))
    return params


def infer_object_affordances(obj: WorldObject) -> list[Affordance]:
    grasp = obj.properties.get("grasp") or {}
    affordances: list[Affordance] = []
    approach = _approach_params(grasp)
    if approach:
        affordances.append(Affordance("approach_pose", score=1.0, params=approach, source="anchor"))

    category = obj.category.lower()
    shape_kind = obj.shape.kind.lower()
    if shape_kind == "box":
        affordances.extend(_box_affordances(obj, grasp, category))
    elif shape_kind == "sphere":
        affordances.append(_single_hand_pinch(obj, grasp, source="shape_rule", score=0.78))
    elif shape_kind == "cylinder":
        affordances.extend(_cylinder_affordances(obj, grasp, category))
    elif _looks_flat(obj):
        affordances.append(_top_grasp(obj, grasp, source="flat_shape_rule", score=0.72))
    elif _looks_small(obj):
        affordances.append(_single_hand_pinch(obj, grasp, source="small_object_rule", score=0.58))
    elif category in {"handle", "drawer_handle", "door_handle", "tool"}:
        affordances.append(_side_grasp(obj, grasp, source="category_rule", score=0.70))

    if obj.category == "place_target":
        affordances.append(Affordance("place_region", score=1.0, params={}, source="anchor"))
    if obj.category in {"table", "counter", "shelf", "support_surface"}:
        affordances.append(Affordance("support_surface", score=1.0, params={}, source="category_rule"))
    if obj.category == "navigation_goal":
        affordances.append(
            Affordance(
                "navigation_target",
                score=1.0,
                params={"goal_tolerance": obj.properties.get("goal_tolerance", 0.45)},
                source="anchor",
            )
        )
    return affordances


def select_grasp_affordance(obj: WorldObject, preferred: str | None = None) -> Affordance | None:
    candidates = [
        aff
        for aff in obj.affordances
        if aff.name in {"bimanual_clamp", "single_hand_pinch", "top_grasp", "side_grasp"}
    ]
    if preferred is not None:
        for aff in candidates:
            if aff.name == preferred:
                return aff
    if not candidates:
        return None
    return max(candidates, key=lambda aff: aff.score)


def approach_pose_from_affordance(obj: WorldObject) -> Pose3 | None:
    for aff in obj.affordances:
        if aff.name != "approach_pose":
            continue
        base_target = aff.params.get("base_target_map")
        point = vec3(base_target)
        if not all(value == value for value in point):
            return None
        yaw = finite_float(base_target[2]) if isinstance(base_target, list) and len(base_target) >= 3 else None
        return Pose3(frame_id="map", position=point, yaw=yaw)
    return None


def _box_affordances(obj: WorldObject, grasp: dict[str, Any], category: str) -> list[Affordance]:
    out = [
        Affordance(
            "bimanual_clamp",
            score=0.85,
            source="shape_rule",
            params={
                "open_y": finite_float(grasp.get("open_y"), 0.24),
                "clamp_y": finite_float(grasp.get("clamp_y"), 0.12),
                "reach_x": finite_float(grasp.get("reach_x"), 0.42),
                "reach_z": finite_float(grasp.get("reach_z"), 0.02),
                "lift_z": finite_float(grasp.get("lift_z"), 0.10),
            },
        )
    ]
    if _looks_small(obj) or category in {"small_package", "fruit_box", "cube"}:
        out.append(_single_hand_pinch(obj, grasp, source="small_box_rule", score=0.62))
    if _looks_flat(obj):
        out.append(_top_grasp(obj, grasp, source="flat_box_rule", score=0.60))
    return out


def _cylinder_affordances(obj: WorldObject, grasp: dict[str, Any], category: str) -> list[Affordance]:
    side_score = 0.82
    if category in {"cup", "mug", "bottle", "can", "tool"}:
        side_score = 0.86
    return [
        _side_grasp(obj, grasp, source="cylinder_rule", score=side_score),
        _top_grasp(obj, grasp, source="cylinder_rule", score=0.60),
    ]


def _single_hand_pinch(
    obj: WorldObject,
    grasp: dict[str, Any],
    *,
    source: str,
    score: float,
) -> Affordance:
    radius = obj.shape.radius
    if radius is None:
        radius = _size_radius(obj)
    radius = finite_float(grasp.get("radius") or grasp.get("ball_radius"), float(radius))
    return Affordance(
        "single_hand_pinch",
        score=score,
        source=source,
        params={
            "hand": str(grasp.get("hand") or "right"),
            "radius": float(radius),
            "target_y": finite_float(grasp.get("target_y"), -0.24),
            "reach_x": finite_float(grasp.get("reach_x"), 0.54),
            "reach_z": finite_float(grasp.get("reach_z"), 0.03),
            "contact_model": str(grasp.get("contact_model") or "three_finger"),
        },
    )


def _side_grasp(
    obj: WorldObject,
    grasp: dict[str, Any],
    *,
    source: str,
    score: float,
) -> Affordance:
    radius = obj.shape.radius if obj.shape.radius is not None else _size_radius(obj)
    height = obj.shape.size[2] if obj.shape.size is not None else finite_float(grasp.get("height"), 0.12)
    return Affordance(
        "side_grasp",
        score=score,
        source=source,
        params={
            "hand": str(grasp.get("hand") or "right"),
            "radius": finite_float(grasp.get("radius"), float(radius)),
            "height": finite_float(grasp.get("height"), float(height)),
            "target_y": finite_float(grasp.get("target_y"), -0.20),
            "reach_x": finite_float(grasp.get("reach_x"), 0.52),
            "reach_z": finite_float(grasp.get("reach_z"), 0.06),
            "approach_axis": str(grasp.get("approach_axis") or "y"),
            "contact_model": str(grasp.get("contact_model") or "side_finger_wrap"),
        },
    )


def _top_grasp(
    obj: WorldObject,
    grasp: dict[str, Any],
    *,
    source: str,
    score: float,
) -> Affordance:
    aperture = finite_float(grasp.get("aperture"), max(0.04, _size_radius(obj) * 2.0))
    return Affordance(
        "top_grasp",
        score=score,
        source=source,
        params={
            "hand": str(grasp.get("hand") or "right"),
            "aperture": aperture,
            "target_y": finite_float(grasp.get("target_y"), -0.18),
            "reach_x": finite_float(grasp.get("reach_x"), 0.50),
            "reach_z": finite_float(grasp.get("reach_z"), 0.08),
            "approach_axis": "z",
            "contact_model": str(grasp.get("contact_model") or "top_pinch"),
        },
    )


def _looks_small(obj: WorldObject) -> bool:
    if obj.shape.radius is not None:
        return obj.shape.radius <= 0.055
    if obj.shape.size is None:
        return False
    return max(obj.shape.size) <= 0.14


def _looks_flat(obj: WorldObject) -> bool:
    if obj.shape.size is None:
        return obj.shape.kind.lower() in {"flat", "thin_box", "sheet"}
    sx, sy, sz = obj.shape.size
    return sz <= 0.04 and max(sx, sy) >= 0.08


def _size_radius(obj: WorldObject) -> float:
    if obj.shape.radius is not None:
        return float(obj.shape.radius)
    if obj.shape.size is None:
        return 0.045
    sx, sy, _ = obj.shape.size
    return max(0.02, min(abs(float(sx)), abs(float(sy))) * 0.5)
