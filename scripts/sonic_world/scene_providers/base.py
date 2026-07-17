from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gear_sonic.utils.mujoco_sim.scene_registry import SceneSelection, resolve_scene


@dataclass(frozen=True)
class SceneProviderRecord:
    provider: str
    scene_name: str
    scene_xml: str
    description: str = ""
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_selection(
        cls,
        selection: SceneSelection,
        *,
        provider: str,
        metadata: dict[str, Any] | None = None,
    ) -> "SceneProviderRecord":
        return cls(
            provider=provider,
            scene_name=selection.name,
            scene_xml=selection.rel_path_posix,
            description=selection.description,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "scene_name": self.scene_name,
            "scene_xml": self.scene_xml,
            "description": self.description,
            "metadata": dict(self.metadata or {}),
        }


def resolve_provider_scene(
    scene: str | Path,
    *,
    provider: str,
    repo_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> SceneProviderRecord:
    return SceneProviderRecord.from_selection(
        resolve_scene(scene, repo_root=repo_root),
        provider=provider,
        metadata=metadata,
    )
