from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .entities import WorldObject, WorldState


@dataclass(frozen=True)
class TaskObjectRecord:
    object_id: str
    category: str
    role: str = "object"
    anchor_id: str | None = None
    policy_object_id: str | None = None
    geom_name: str | None = None
    body_name: str | None = None
    joint_name: str | None = None
    site_name: str | None = None
    support_id: str | None = None
    source: str = "registry"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "role": self.role,
            "anchor_id": self.anchor_id,
            "policy_object_id": self.policy_object_id,
            "geom_name": self.geom_name,
            "body_name": self.body_name,
            "joint_name": self.joint_name,
            "site_name": self.site_name,
            "support_id": self.support_id,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskObjectRegistryValidation:
    task_id: str | None
    scene: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sonic_task_object_registry_validation_v0",
            "task_id": self.task_id,
            "scene": self.scene,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TaskObjectRegistry:
    task_id: str | None = None
    scene: str | None = None
    records: tuple[TaskObjectRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_world(
        cls,
        world: WorldState,
        *,
        task_id: str | None = None,
        scene: str | None = None,
    ) -> "TaskObjectRegistry":
        records = tuple(_record_from_world_object(obj) for obj in world.objects.values())
        return cls(
            task_id=task_id or _string_or_none(world.properties.get("task_id")),
            scene=scene or _string_or_none(world.properties.get("scene")),
            records=records,
            metadata={"source": "world_state", "object_count": len(records)},
        )

    @classmethod
    def from_task_case(cls, task: Any) -> "TaskObjectRegistry":
        task_id = str(getattr(task, "task_id", "") or "")
        scene = str(getattr(getattr(task, "scene", None), "scene_xml", "") or "")
        objects = getattr(task, "objects", ()) or ()
        records = tuple(_record_from_task_object(obj, task_id=task_id, scene=scene) for obj in objects if isinstance(obj, dict))
        return cls(
            task_id=task_id or None,
            scene=scene or None,
            records=records,
            metadata={"source": "task_suite", "object_count": len(records)},
        )

    def by_object_id(self) -> dict[str, TaskObjectRecord]:
        return {record.object_id: record for record in self.records}

    def get(self, object_id: str) -> TaskObjectRecord | None:
        return self.by_object_id().get(object_id)

    def resolve_anchor_name(self, object_id: str, *, demo_kind: str | None = None) -> str:
        record = self.get(object_id)
        if record is not None:
            if demo_kind == "ball" and record.geom_name:
                return record.geom_name
            if demo_kind == "box" and record.geom_name:
                return record.geom_name
            return record.anchor_id or record.geom_name or record.object_id
        return object_id

    def validate(
        self,
        *,
        task: Any | None = None,
        repo_root: str | Path | None = None,
        check_scene_names: bool = True,
    ) -> TaskObjectRegistryValidation:
        errors: list[str] = []
        warnings: list[str] = []
        by_id: dict[str, TaskObjectRecord] = {record.object_id: record for record in self.records if record.object_id}
        seen: set[str] = set()
        for record in self.records:
            if not record.object_id:
                errors.append("record missing object_id")
                continue
            if record.object_id in seen:
                errors.append(f"duplicate object_id {record.object_id!r}")
            seen.add(record.object_id)
            if not record.anchor_id:
                warnings.append(f"{record.object_id}: missing anchor_id")
            if not record.policy_object_id:
                warnings.append(f"{record.object_id}: missing policy_object_id")
            if record.role == "object" and not record.geom_name:
                errors.append(f"{record.object_id}: object record missing geom_name")
            if record.role == "target" and not (record.site_name or record.geom_name or record.anchor_id):
                errors.append(f"{record.object_id}: target record missing site/geom/anchor name")
            if record.support_id and record.support_id not in by_id:
                warnings.append(f"{record.object_id}: support_id {record.support_id!r} is not registered")

        if task is not None:
            request = getattr(task, "request", None)
            object_id = _string_or_none(getattr(request, "object_id", None))
            target_id = _string_or_none(getattr(request, "target_id", None))
            if object_id and object_id not in by_id:
                errors.append(f"request object_id {object_id!r} is not registered")
            if target_id and target_id not in by_id:
                errors.append(f"request target_id {target_id!r} is not registered")
            if target_id and target_id in by_id and by_id[target_id].role not in {"target", "support"}:
                warnings.append(f"request target_id {target_id!r} has role {by_id[target_id].role!r}")

        if check_scene_names:
            scene_path = _scene_path(self.scene, repo_root=repo_root)
            if scene_path is not None and scene_path.exists():
                names = _mjcf_names(scene_path)
                for record in self.records:
                    required_names: list[tuple[str, str | None]] = []
                    if record.role in {"object", "distractor"}:
                        required_names.append(("geom", record.geom_name))
                    if record.role == "target":
                        required_names.append(("site", record.site_name))
                    for label, name in required_names:
                        if name and name not in names.get(label, set()):
                            warnings.append(f"{record.object_id}: {label}_name {name!r} not found in scene XML")
                    if record.body_name and record.body_name not in names["body"]:
                        warnings.append(f"{record.object_id}: body_name {record.body_name!r} not found in scene XML")
                    if record.joint_name and record.joint_name not in names["joint"]:
                        warnings.append(f"{record.object_id}: joint_name {record.joint_name!r} not found in scene XML")
            elif self.scene:
                warnings.append(f"scene XML not found for registry validation: {self.scene}")

        return TaskObjectRegistryValidation(
            task_id=self.task_id,
            scene=self.scene,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sonic_task_object_registry_v0",
            "task_id": self.task_id,
            "scene": self.scene,
            "records": [record.to_dict() for record in self.records],
            "metadata": self.metadata,
        }


def _record_from_world_object(obj: WorldObject) -> TaskObjectRecord:
    props = obj.properties if isinstance(obj.properties, dict) else {}
    object_id = str(obj.object_id)
    role = _role_for_category(obj.category, props)
    return TaskObjectRecord(
        object_id=object_id,
        category=str(obj.category),
        role=role,
        anchor_id=str(props.get("anchor_id") or object_id),
        policy_object_id=str(props.get("policy_object_id") or object_id),
        geom_name=_string_or_none(props.get("geom_name")) or _generated_name(object_id, "geom"),
        body_name=_string_or_none(props.get("body_name")) or _generated_name(object_id, "body"),
        joint_name=_string_or_none(props.get("joint_name")) or _generated_name(object_id, "joint"),
        site_name=_string_or_none(props.get("site_name")) or (_generated_name(object_id, "site") if role == "target" else None),
        support_id=obj.support,
        source=obj.source,
        metadata={
            "raw_anchor_kind": props.get("raw_anchor_kind"),
            "tracking_id": props.get("tracking_id"),
            "uncertainty": props.get("uncertainty"),
        },
    )


def _record_from_task_object(obj: dict[str, Any], *, task_id: str, scene: str) -> TaskObjectRecord:
    object_id = str(obj.get("object_id") or obj.get("id") or obj.get("name") or "")
    category = str(obj.get("category") or obj.get("object_category") or obj.get("class") or "object")
    props = obj.get("properties") if isinstance(obj.get("properties"), dict) else {}
    role = _role_for_category(category, props)
    generated = "scene_sonic_task_" in scene or bool(task_id and object_id.startswith((f"{task_id}_", "sg_")))
    generated_container = f"{task_id}_generated_task" if generated else object_id
    default_geom = f"{object_id}_collision" if generated and role == "support" else (_generated_name(object_id, "geom") if generated else object_id)
    default_body = generated_container if generated and role in {"support", "target"} else object_id
    return TaskObjectRecord(
        object_id=object_id,
        category=category,
        role=role,
        anchor_id=str(props.get("anchor_id") or object_id),
        policy_object_id=str(props.get("policy_object_id") or object_id),
        geom_name=_string_or_none(props.get("geom_name")) or default_geom,
        body_name=_string_or_none(props.get("body_name")) or default_body,
        joint_name=_string_or_none(props.get("joint_name"))
        or (_generated_name(object_id, "freejoint") if generated and role == "object" else None),
        site_name=_string_or_none(props.get("site_name")) or (_generated_name(object_id, "site") if role == "target" and generated else None),
        support_id=_string_or_none(obj.get("support") or obj.get("support_surface")),
        source="task_suite",
        metadata={
            "generated_scene": generated,
            "target_of_task": props.get("target_of_task"),
            "role": props.get("role"),
        },
    )


def _role_for_category(category: str, props: dict[str, Any]) -> str:
    role = props.get("role")
    if role:
        return str(role)
    if category in {"place_target", "navigation_goal"}:
        return "target"
    if category in {"table", "counter", "shelf", "support_surface"}:
        return "support"
    if props.get("target_of_task") is False:
        return "distractor"
    return "object"


def _generated_name(object_id: str, suffix: str) -> str:
    return f"{object_id}_{suffix}"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _scene_path(scene: str | None, *, repo_root: str | Path | None) -> Path | None:
    if not scene:
        return None
    path = Path(scene).expanduser()
    if path.is_absolute():
        return path
    if repo_root is None:
        return path
    return Path(repo_root).expanduser() / path


def _mjcf_names(scene_path: Path) -> dict[str, set[str]]:
    names = {kind: set() for kind in ("body", "joint", "geom", "site")}
    root = ET.parse(scene_path).getroot()
    for kind in names:
        for element in root.iter(kind):
            name = element.get("name")
            if name:
                names[kind].add(name)
    for element in root.iter("freejoint"):
        name = element.get("name")
        if name:
            names["joint"].add(name)
    return names
