from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..planners import TaskRequest
from ..scene_providers import SceneProviderRecord, resolve_robocasa_scene


DEFAULT_ROBOCASA_SUITE = Path("configs/world_model/task_suites/robocasa_v0.yaml")


@dataclass(frozen=True)
class RobocasaTaskCase:
    task_id: str
    scene: SceneProviderRecord
    request: TaskRequest
    objects: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()
    expectation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, repo_root: str | Path | None = None) -> "RobocasaTaskCase":
        if not isinstance(payload, dict):
            raise ValueError("RoboCasa task case must be a mapping")
        task_id = _required_string(payload, "id")
        scene_name = _required_string(payload, "scene")
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"RoboCasa task {task_id!r} must contain non-empty objects")
        request = payload.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"RoboCasa task {task_id!r} must contain request mapping")
        expectation = payload.get("expect") or {}
        if not isinstance(expectation, dict):
            raise ValueError(f"RoboCasa task {task_id!r} expect must be a mapping")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"RoboCasa task {task_id!r} metadata must be a mapping")
        relations = payload.get("relations") or []
        if not isinstance(relations, list):
            raise ValueError(f"RoboCasa task {task_id!r} relations must be a list")
        return cls(
            task_id=task_id,
            scene=resolve_robocasa_scene(scene_name, repo_root=repo_root),
            request=TaskRequest.from_dict({**request, "id": request.get("id") or task_id}),
            objects=tuple(_copy_mapping(obj, f"object in {task_id}") for obj in objects),
            relations=tuple(_copy_mapping(rel, f"relation in {task_id}") for rel in relations),
            description=str(payload.get("description") or ""),
            tags=tuple(str(tag) for tag in payload.get("tags") or ()),
            expectation=deepcopy(expectation),
            metadata=deepcopy(metadata),
        )

    def anchor(self) -> dict[str, Any]:
        return {
            "scene": self.scene.scene_name,
            "source": "robocasa_task_suite",
            "frame_id": "map",
            "objects": [deepcopy(obj) for obj in self.objects],
            "relations": [deepcopy(rel) for rel in self.relations],
            "properties": {
                "task_suite": "robocasa_v0",
                "task_id": self.task_id,
                "scene_provider": self.scene.to_dict(),
                "tags": list(self.tags),
                **deepcopy(self.metadata),
            },
        }

    def scenario(self) -> dict[str, Any]:
        return {
            "name": f"robocasa_{self.task_id}",
            "metadata": {
                "provider": "robocasa",
                "scene": self.scene.to_dict(),
                "description": self.description,
                "tags": list(self.tags),
            },
            "anchors": [self.anchor()],
            "expected_objects": [str(obj.get("object_id") or obj.get("id")) for obj in self.objects],
            "tasks": [
                {
                    "name": self.task_id,
                    "request": self.request.to_dict(),
                    "expect": deepcopy(self.expectation),
                    "source": f"robocasa_task_suite:{self.task_id}",
                }
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "description": self.description,
            "tags": list(self.tags),
            "scene": self.scene.to_dict(),
            "request": self.request.to_dict(),
            "anchor": self.anchor(),
            "expect": deepcopy(self.expectation),
        }


@dataclass(frozen=True)
class RobocasaTaskSuite:
    name: str
    version: str
    description: str
    tasks: tuple[RobocasaTaskCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, repo_root: str | Path | None = None) -> "RobocasaTaskSuite":
        if not isinstance(payload, dict):
            raise ValueError("RoboCasa task suite YAML must be a mapping")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("RoboCasa task suite must contain non-empty tasks")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("RoboCasa task suite metadata must be a mapping")
        return cls(
            name=str(payload.get("name") or "robocasa_task_suite"),
            version=str(payload.get("version") or "v0"),
            description=str(payload.get("description") or ""),
            tasks=tuple(RobocasaTaskCase.from_dict(task, repo_root=repo_root) for task in tasks),
            metadata=deepcopy(metadata),
        )

    def get_task(self, task_id: str) -> RobocasaTaskCase:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        valid = ", ".join(task.task_id for task in self.tasks)
        raise ValueError(f"unknown RoboCasa task {task_id!r}; expected one of: {valid}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": dict(self.metadata),
            "tasks": [task.to_dict() for task in self.tasks],
        }


def load_robocasa_task_suite(
    path: str | Path = DEFAULT_ROBOCASA_SUITE,
    *,
    repo_root: str | Path | None = None,
) -> RobocasaTaskSuite:
    suite_path = Path(path).expanduser()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd()
    if not suite_path.is_absolute():
        suite_path = root / suite_path
    with suite_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return RobocasaTaskSuite.from_dict(payload, repo_root=root)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"missing required field {key!r}")
    return text


def _copy_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(value)
