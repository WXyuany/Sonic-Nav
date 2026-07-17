from __future__ import annotations

from pathlib import Path

from .base import SceneProviderRecord, resolve_provider_scene


ROBOCASA_SCENES: tuple[str, ...] = (
    "robocasa_kitchen",
    "robocasa_galley",
    "robocasa_apartment",
    "robocasa_cafe",
)


def resolve_robocasa_scene(scene: str, *, repo_root: str | Path | None = None) -> SceneProviderRecord:
    metadata = {
        "role": "physical_interaction_scene",
        "visual_source": "mujoco_assets",
        "physics": "mujoco_collision_proxy",
    }
    if scene in ROBOCASA_SCENES:
        return resolve_provider_scene(scene, provider="robocasa", repo_root=repo_root, metadata=metadata)
    try:
        record = resolve_provider_scene(scene, provider="robocasa_generated", repo_root=repo_root, metadata=metadata)
    except ValueError as exc:
        valid = ", ".join(ROBOCASA_SCENES)
        raise ValueError(f"unknown RoboCasa scene {scene!r}; expected one of: {valid}, or an existing XML path") from exc
    return record


def robocasa_scene_catalog(*, repo_root: str | Path | None = None) -> list[SceneProviderRecord]:
    return [resolve_robocasa_scene(scene, repo_root=repo_root) for scene in ROBOCASA_SCENES]
