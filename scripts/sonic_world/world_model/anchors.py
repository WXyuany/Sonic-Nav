from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .affordance import infer_object_affordances
from .entities import Affordance, ObjectShape, Pose3, WorldObject, WorldRelation, WorldState, finite_float, vec3


def load_anchor_payload(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text_or_path = str(value)
    path = Path(text_or_path)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(text_or_path)


def detect_anchor_kind(anchor: dict[str, Any]) -> str:
    if isinstance(anchor.get("objects"), list):
        return "objects"
    if _looks_like_generic_object(anchor):
        return "object"
    if "box_center_map" in anchor or "box_point_base" in anchor:
        return "box"
    if "ball_center_map" in anchor or "ball_point_base" in anchor:
        return "ball"
    if "goal_center_map" in anchor or "goal_point_base" in anchor:
        return "navigation_goal"
    raise ValueError("unsupported anchor payload: expected objects, object_*, box_*, ball_*, or goal_* fields")


def _looks_like_generic_object(anchor: dict[str, Any]) -> bool:
    if "object_id" in anchor or "id" in anchor:
        return True
    if "category" in anchor and ("shape" in anchor or "pose_map" in anchor or "center_map" in anchor):
        return True
    return False


def _pose(frame_id: str, point: Any) -> Pose3 | None:
    yaw = None
    orientation = None
    raw_point = point
    if isinstance(point, dict):
        frame_id = str(point.get("frame_id") or frame_id)
        raw_point = point.get("position") or point.get("xyz") or point.get("point")
        if point.get("yaw") is not None:
            yaw = finite_float(point.get("yaw"))
        raw_orientation = point.get("orientation_xyzw")
        if isinstance(raw_orientation, (list, tuple)) and len(raw_orientation) >= 4:
            orientation = (
                finite_float(raw_orientation[0]),
                finite_float(raw_orientation[1]),
                finite_float(raw_orientation[2]),
                finite_float(raw_orientation[3]),
            )
    position = vec3(raw_point)
    if not all(value == value for value in position):
        return None
    return Pose3(frame_id=frame_id, position=position, yaw=yaw, orientation_xyzw=orientation)


def _stamp(anchor: dict[str, Any]) -> float | None:
    raw = anchor.get("stamp")
    if not isinstance(raw, dict):
        return None
    sec = finite_float(raw.get("sec"), 0.0)
    nsec = finite_float(raw.get("nanosec"), 0.0)
    return sec + nsec * 1e-9


def _box_object(anchor: dict[str, Any]) -> WorldObject:
    frame_id = str(anchor.get("frame_id", "map"))
    obj = WorldObject(
        object_id=str(anchor.get("box_name", "box")),
        category="box",
        shape=ObjectShape.box(anchor.get("box_size", [0.24, 0.16, 0.16])),
        pose_map=_pose(frame_id, anchor.get("box_center_map")),
        pose_base=_pose("base_link", anchor.get("box_point_base")),
        pose_camera=_pose("camera_depth_optical_frame", anchor.get("box_point_camera_depth")),
        source=str(anchor.get("source", "anchor")),
        support="table",
        properties={
            "scene": anchor.get("scene"),
            "pixel": anchor.get("box_pixel"),
            "grasp": anchor.get("grasp") or {},
            "raw_anchor_kind": "box",
        },
    )
    obj.affordances = infer_object_affordances(obj)
    return obj


def _ball_object(anchor: dict[str, Any]) -> WorldObject:
    frame_id = str(anchor.get("frame_id", "map"))
    radius = finite_float(anchor.get("ball_radius"), 0.045)
    obj = WorldObject(
        object_id=str(anchor.get("ball_name", "ball")),
        category="ball",
        shape=ObjectShape.sphere(radius),
        pose_map=_pose(frame_id, anchor.get("ball_center_map")),
        pose_base=_pose("base_link", anchor.get("ball_point_base")),
        pose_camera=_pose("camera_depth_optical_frame", anchor.get("ball_point_camera_depth")),
        source=str(anchor.get("source", "anchor")),
        support="table",
        properties={
            "scene": anchor.get("scene"),
            "pixel": anchor.get("ball_pixel"),
            "grasp": anchor.get("grasp") or {},
            "raw_anchor_kind": "ball",
        },
    )
    obj.affordances = infer_object_affordances(obj)
    return obj


def _place_target(anchor: dict[str, Any]) -> WorldObject | None:
    if "place_center_map" not in anchor and "place_point_base" not in anchor:
        return None
    frame_id = str(anchor.get("frame_id", "map"))
    obj = WorldObject(
        object_id=str(anchor.get("place_name") or "place_target"),
        category="place_target",
        shape=ObjectShape.target(),
        pose_map=_pose(frame_id, anchor.get("place_center_map")),
        pose_base=_pose("base_link", anchor.get("place_point_base")),
        source=str(anchor.get("source", "anchor")),
        support="table",
        properties={"raw_anchor_kind": "place_target"},
    )
    obj.affordances = infer_object_affordances(obj)
    return obj


def _generic_object(record: dict[str, Any], anchor: dict[str, Any]) -> WorldObject:
    frame_id = str(record.get("frame_id") or anchor.get("frame_id") or "map")
    category = str(record.get("category") or record.get("object_category") or record.get("class") or "object")
    properties = dict(record.get("properties") or {})
    properties.setdefault("scene", anchor.get("scene"))
    properties.setdefault("pixel", record.get("pixel"))
    properties.setdefault("grasp", record.get("grasp") or {})
    properties.setdefault("raw_anchor_kind", "object")
    for key in ("confidence", "tracking_id", "track_id", "uncertainty", "bbox", "mask_id"):
        if key in record and key not in properties:
            properties[key] = record.get(key)
    if "track_id" in properties and "tracking_id" not in properties:
        properties["tracking_id"] = properties["track_id"]
    obj = WorldObject(
        object_id=str(record.get("object_id") or record.get("id") or record.get("name") or category),
        category=category,
        shape=_generic_shape(record, category),
        pose_map=_pose(
            frame_id,
            _first_present(record, "pose_map", "center_map", "object_center_map", "position_map", "map_position"),
        ),
        pose_base=_pose(
            "base_link",
            _first_present(record, "pose_base", "point_base", "object_point_base", "base_position"),
        ),
        pose_camera=_pose(
            "camera_depth_optical_frame",
            _first_present(record, "pose_camera", "point_camera", "point_camera_depth", "camera_position"),
        ),
        source=str(record.get("source") or anchor.get("source") or "anchor"),
        support=record.get("support") or record.get("support_surface"),
        properties=properties,
    )
    obj.affordances = [
        *infer_object_affordances(obj),
        *_explicit_affordances(record.get("affordances")),
    ]
    return obj


def _generic_shape(record: dict[str, Any], category: str) -> ObjectShape:
    raw = record.get("shape")
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or raw.get("type") or "unknown")
        if kind == "box":
            return ObjectShape.box(raw.get("size") or record.get("size") or [0.0, 0.0, 0.0])
        if kind == "sphere":
            return ObjectShape.sphere(finite_float(raw.get("radius") or record.get("radius"), 0.0))
        if kind == "target":
            return ObjectShape.target()
        size = raw.get("size") or record.get("size")
        radius = raw.get("radius") or record.get("radius")
        return ObjectShape(
            kind=kind,
            size=vec3(size, default=(0.0, 0.0, 0.0)) if size is not None else None,
            radius=finite_float(radius, 0.0) if radius is not None else None,
        )
    if isinstance(raw, str):
        if raw == "target":
            return ObjectShape.target()
        if raw == "sphere":
            return ObjectShape.sphere(finite_float(record.get("radius"), 0.0))
        if raw == "box":
            return ObjectShape.box(record.get("size") or [0.0, 0.0, 0.0])
        return ObjectShape(raw)
    if category in {"place_target", "navigation_goal"}:
        return ObjectShape.target()
    if record.get("radius") is not None:
        return ObjectShape.sphere(finite_float(record.get("radius"), 0.0))
    if record.get("size") is not None:
        return ObjectShape.box(record.get("size"))
    return ObjectShape("unknown")


def _explicit_affordances(raw: Any) -> list[Affordance]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("object affordances must be a list")
    out: list[Affordance] = []
    for item in raw:
        if isinstance(item, str):
            out.append(Affordance(item, source="anchor"))
            continue
        if not isinstance(item, dict):
            raise ValueError("object affordance entries must be strings or objects")
        name = str(item.get("name") or item.get("type") or "").strip()
        if not name:
            raise ValueError("object affordance missing name")
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("object affordance params must be an object")
        out.append(
            Affordance(
                name=name,
                score=finite_float(item.get("score"), 1.0),
                params=params,
                source=str(item.get("source") or "anchor"),
            )
        )
    return out


def _generic_objects(anchor: dict[str, Any], kind: str) -> list[WorldObject]:
    records = anchor.get("objects") if kind == "objects" else [anchor]
    if not isinstance(records, list) or not records:
        raise ValueError("generic object anchor must contain at least one object")
    out: list[WorldObject] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("generic object records must be objects")
        out.append(_generic_object(record, anchor))
    return out


def _generic_relation(raw: dict[str, Any]) -> WorldRelation:
    return WorldRelation(
        subject_id=str(raw.get("subject_id") or raw.get("subject") or ""),
        relation=str(raw.get("relation") or "related_to"),
        object_id=str(raw.get("object_id") or raw.get("object") or ""),
        confidence=finite_float(raw.get("confidence"), 1.0),
    )


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _navigation_goal(anchor: dict[str, Any]) -> WorldObject:
    frame_id = str(anchor.get("frame_id", "map"))
    yaw = anchor.get("goal_yaw")
    obj = WorldObject(
        object_id=str(anchor.get("goal_name", "navigation_goal")),
        category="navigation_goal",
        shape=ObjectShape.target(),
        pose_map=_pose(frame_id, anchor.get("goal_center_map")),
        pose_base=_pose("base_link", anchor.get("goal_point_base")),
        source=str(anchor.get("source", "anchor")),
        support=None,
        properties={
            "raw_anchor_kind": "navigation_goal",
            "goal_tolerance": finite_float(anchor.get("goal_tolerance"), 0.45),
            "goal_yaw": finite_float(yaw) if yaw is not None else None,
        },
    )
    if obj.pose_map is not None and obj.properties.get("goal_yaw") is not None:
        obj.pose_map = Pose3(obj.pose_map.frame_id, obj.pose_map.position, yaw=obj.properties["goal_yaw"])
    obj.affordances = infer_object_affordances(obj)
    return obj


def anchor_to_world(anchor_payload: str | Path | dict[str, Any]) -> WorldState:
    anchor = load_anchor_payload(anchor_payload)
    kind = detect_anchor_kind(anchor)
    stamp = _stamp(anchor)
    properties = {
        "scene": anchor.get("scene"),
        "anchor_kind": kind,
        "source": anchor.get("source"),
    }
    raw_properties = anchor.get("properties")
    if isinstance(raw_properties, dict):
        properties.update(raw_properties)
    world = WorldState(
        frame_id=str(anchor.get("frame_id", "map")),
        stamp=stamp if stamp is not None else WorldState().stamp,
        properties=properties,
    )
    _apply_robot_state(world, anchor)
    if kind in {"object", "objects"}:
        for obj in _generic_objects(anchor, kind):
            world.upsert_object(obj)
            if obj.support is not None:
                world.add_relation(WorldRelation(obj.object_id, "on", str(obj.support), confidence=0.8))
        for raw_relation in anchor.get("relations") or []:
            if isinstance(raw_relation, dict):
                relation = _generic_relation(raw_relation)
                if relation.subject_id and relation.object_id:
                    world.add_relation(relation)
        return world
    if kind == "box":
        obj = _box_object(anchor)
    elif kind == "ball":
        obj = _ball_object(anchor)
    else:
        obj = _navigation_goal(anchor)
    world.upsert_object(obj)
    if obj.support is not None:
        world.add_relation(WorldRelation(obj.object_id, "on", obj.support, confidence=0.8))

    target = _place_target(anchor)
    if target is not None:
        world.upsert_object(target)
        world.add_relation(WorldRelation(target.object_id, "on", target.support or "table", confidence=0.8))
    return world


def _apply_robot_state(world: WorldState, anchor: dict[str, Any]) -> None:
    frame_id = str(anchor.get("frame_id", "map"))
    raw_pose = _first_present(anchor, "robot_pose_map", "robot_start_map", "base_pose_map", "base_start_map")
    pose = _pose(frame_id, raw_pose)
    if pose is None:
        return
    yaw = anchor.get("robot_yaw")
    if yaw is None:
        yaw = anchor.get("robot_start_yaw")
    if yaw is not None:
        pose = Pose3(pose.frame_id, pose.position, yaw=finite_float(yaw))
    world.robot.base_map = pose
