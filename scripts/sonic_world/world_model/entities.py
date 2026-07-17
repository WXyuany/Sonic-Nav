from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterable


Vec3 = tuple[float, float, float]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def vec3(value: Any, default: Vec3 = (math.nan, math.nan, math.nan)) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return default
    out = (finite_float(value[0]), finite_float(value[1]), finite_float(value[2]))
    if not all(math.isfinite(v) for v in out):
        return default
    return out


def finite_vec3(value: Any) -> bool:
    return all(math.isfinite(v) for v in vec3(value))


def vec3_distance_xy(a: Vec3, b: Vec3) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@dataclass(frozen=True)
class Pose3:
    frame_id: str
    position: Vec3
    yaw: float | None = None
    orientation_xyzw: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "position": list(self.position),
        }
        if self.yaw is not None and math.isfinite(float(self.yaw)):
            payload["yaw"] = float(self.yaw)
        if self.orientation_xyzw is not None:
            payload["orientation_xyzw"] = list(self.orientation_xyzw)
        return payload


@dataclass(frozen=True)
class ObjectShape:
    kind: str
    size: Vec3 | None = None
    radius: float | None = None

    @classmethod
    def box(cls, size: Iterable[float]) -> "ObjectShape":
        return cls("box", size=vec3(list(size), default=(0.0, 0.0, 0.0)))

    @classmethod
    def sphere(cls, radius: float) -> "ObjectShape":
        return cls("sphere", radius=max(0.0, finite_float(radius, 0.0)))

    @classmethod
    def target(cls) -> "ObjectShape":
        return cls("target")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.size is not None:
            payload["size"] = list(self.size)
        if self.radius is not None:
            payload["radius"] = float(self.radius)
        return payload


@dataclass(frozen=True)
class Affordance:
    name: str
    score: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": float(self.score),
            "source": self.source,
            "params": self.params,
        }


@dataclass
class WorldObject:
    object_id: str
    category: str
    shape: ObjectShape
    pose_map: Pose3 | None = None
    pose_base: Pose3 | None = None
    pose_camera: Pose3 | None = None
    source: str = "unknown"
    support: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    affordances: list[Affordance] = field(default_factory=list)

    def pose(self, frame_id: str) -> Pose3 | None:
        if frame_id in {"map", "world"}:
            return self.pose_map
        if frame_id in {"base", "base_link"}:
            return self.pose_base
        if frame_id in {"camera", "camera_depth_optical_frame"}:
            return self.pose_camera
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "shape": self.shape.to_dict(),
            "pose_map": self.pose_map.to_dict() if self.pose_map else None,
            "pose_base": self.pose_base.to_dict() if self.pose_base else None,
            "pose_camera": self.pose_camera.to_dict() if self.pose_camera else None,
            "source": self.source,
            "support": self.support,
            "properties": self.properties,
            "affordances": [aff.to_dict() for aff in self.affordances],
        }


@dataclass(frozen=True)
class WorldRelation:
    subject_id: str
    relation: str
    object_id: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "relation": self.relation,
            "object_id": self.object_id,
            "confidence": float(self.confidence),
        }


@dataclass
class RobotState:
    base_map: Pose3 | None = None
    base_velocity: Vec3 | None = None
    stable: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_map": self.base_map.to_dict() if self.base_map else None,
            "base_velocity": list(self.base_velocity) if self.base_velocity else None,
            "stable": bool(self.stable),
            "properties": self.properties,
        }


@dataclass
class WorldState:
    frame_id: str = "map"
    stamp: float = field(default_factory=time.time)
    robot: RobotState = field(default_factory=RobotState)
    objects: dict[str, WorldObject] = field(default_factory=dict)
    relations: list[WorldRelation] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def upsert_object(self, obj: WorldObject) -> WorldObject:
        self.objects[obj.object_id] = obj
        return obj

    def add_relation(self, relation: WorldRelation) -> None:
        self.relations.append(relation)

    def get_object(self, object_id: str) -> WorldObject | None:
        return self.objects.get(object_id)

    def find_by_category(self, category: str) -> list[WorldObject]:
        return [obj for obj in self.objects.values() if obj.category == category]

    def primary_object(self) -> WorldObject | None:
        candidates = [
            obj
            for obj in self.objects.values()
            if obj.category not in {"place_target", "support_surface", "navigation_goal"}
        ]
        if candidates:
            return candidates[0]
        return next(iter(self.objects.values()), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "stamp": float(self.stamp),
            "robot": self.robot.to_dict(),
            "objects": {key: obj.to_dict() for key, obj in self.objects.items()},
            "relations": [relation.to_dict() for relation in self.relations],
            "properties": self.properties,
        }
