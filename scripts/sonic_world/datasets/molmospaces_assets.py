from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import xml.etree.ElementTree as ET
from typing import Any

from .molmospaces import MolmoSpacesEpisode


@dataclass(frozen=True)
class MolmoSpacesRealScene:
    scene_source: str
    scene_archive: str
    scene_xml: Path
    variant: str = "base"
    installed_object_archives: dict[str, list[str]] = field(default_factory=dict)


def resolve_real_scene_assets(
    episode: MolmoSpacesEpisode,
    *,
    variant: str = "base",
    install: bool = True,
    install_objects: bool = True,
    assets_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> MolmoSpacesRealScene:
    """Resolve and optionally install the real MolmoSpaces MJCF scene for an episode.

    This uses a minimal resource-manager version table. It does not initialize the
    full MolmoSpaces asset universe, which would pull eager robot/test-data sources.
    """

    deps = _molmospaces_deps()
    scene_source = scene_source_for_episode(episode)
    scene_version = deps["versions"]["scenes"][scene_source]
    selected_assets_dir = Path(assets_dir or deps["assets_dir"])
    selected_cache_dir = Path(cache_dir or deps["cache_dir"])
    versions = {"scenes": {scene_source: scene_version}}
    manager = deps["resource_manager"](
        deps["remote_storage"]("mujoco-thor-resources"),
        versions,
        selected_assets_dir,
        selected_cache_dir,
        force_install=False,
    )
    manager.setup()

    scene_xml, archive = _scene_xml_from_manager(manager, episode, scene_source, variant)
    if install:
        manager.install_packages("scenes", {scene_source: [archive]})

    installed_object_archives: dict[str, list[str]] = {}
    if install and install_objects:
        installed_object_archives = install_object_assets_for_scene(
            scene_xml,
            assets_dir=selected_assets_dir,
            cache_dir=selected_cache_dir,
        )
    return MolmoSpacesRealScene(
        scene_source=scene_source,
        scene_archive=archive,
        scene_xml=scene_xml,
        variant=variant,
        installed_object_archives=installed_object_archives,
    )


def install_object_assets_for_scene(
    scene_xml: str | Path,
    *,
    assets_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    deps = _molmospaces_deps()
    selected_assets_dir = Path(assets_dir or deps["assets_dir"])
    selected_cache_dir = Path(cache_dir or deps["cache_dir"])
    versions = {
        "objects": {
            "objaverse": deps["versions"]["objects"]["objaverse"],
            "thor": deps["versions"]["objects"]["thor"],
        }
    }
    manager = deps["resource_manager"](
        deps["remote_storage"]("mujoco-thor-resources"),
        versions,
        selected_assets_dir,
        selected_cache_dir,
        force_install=False,
        source_overrides={("objects", "thor"): {"install_mode": deps["install_mode"].ON_DEMAND}},
    )
    manager.setup()

    refs = object_refs_for_scene(scene_xml, assets_dir=selected_assets_dir)
    installed: dict[str, list[str]] = {}
    for source, paths in refs.items():
        if source not in versions["objects"]:
            continue
        unique_paths = _unique_paths(paths)
        if not unique_paths:
            continue
        archives = manager.find_archives("objects", source, unique_paths)
        manager.install_packages("objects", {source: archives})
        installed[source] = archives
    return installed


def object_refs_for_scene(
    scene_xml: str | Path,
    *,
    assets_dir: str | Path,
) -> dict[str, list[Path]]:
    scene_path = Path(os.path.abspath(scene_xml))
    objects_root = Path(os.path.abspath(Path(assets_dir) / "objects"))
    root = ET.parse(scene_path).getroot()
    refs: dict[str, list[Path]] = {}
    for elem in root.findall(".//asset/*"):
        file_attr = elem.attrib.get("file")
        if not file_attr or not file_attr.startswith("../../objects/"):
            continue
        full = Path(os.path.normpath(os.path.join(str(scene_path.parent), file_attr)))
        rel = full.relative_to(objects_root)
        source = rel.parts[0]
        refs.setdefault(source, []).append(Path(*rel.parts[1:]))
    return refs


def scene_source_for_episode(episode: MolmoSpacesEpisode) -> str:
    dataset = episode.scene_dataset
    split = episode.data_split
    if dataset == "ithor":
        return "ithor"
    if dataset in {"procthor-10k", "procthor-objaverse", "holodeck-objaverse"}:
        return f"{dataset}-{split}"
    return f"{dataset}-{split}"


def _scene_xml_from_manager(
    manager: Any,
    episode: MolmoSpacesEpisode,
    scene_source: str,
    variant: str,
) -> tuple[Path, str]:
    if episode.house_index is None:
        raise ValueError("MolmoSpaces episode has no house_index; cannot resolve a real scene XML")
    prefix = "FloorPlan" if scene_source == "ithor" else str(episode.data_split)
    if scene_source == "ithor":
        target_name = f"FloorPlan{episode.house_index}_physics.xml"
    else:
        suffix = "_ceiling.xml" if variant == "ceiling" else ".xml"
        target_name = f"{prefix}_{episode.house_index}{suffix}"

    info = manager.source_info("scenes", scene_source, recursive=False)
    matches: list[tuple[str, str]] = []
    for archive, rel_paths in info["archive_to_relative_paths"].items():
        for rel_path in rel_paths:
            rel_text = str(rel_path)
            if rel_text == target_name or rel_text.endswith(f"/{target_name}"):
                matches.append((archive, rel_text))
    if not matches:
        raise FileNotFoundError(
            f"MolmoSpaces scene XML {target_name!r} not found in source {scene_source!r}"
        )
    archive, rel_path = matches[0]
    scene_xml = Path(manager.symlink_dir) / "scenes" / scene_source / rel_path
    return scene_xml, archive


def _molmospaces_deps() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    molmospaces_src = repo / "external_dependencies" / "molmospaces-src"
    if molmospaces_src.exists():
        import sys

        src_text = str(molmospaces_src)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(repo / "MolmoSpacesAssets"))
    os.environ.setdefault("MLSPACES_CACHE_DIR", str(repo / ".cache" / "molmo-spaces-resources"))
    try:
        from molmospaces_resources import R2RemoteStorage
        from molmospaces_resources.behaviors import InstallMode
        from molmospaces_resources.manager import ResourceManager
        from molmo_spaces.molmo_spaces_constants import (
            ASSETS_DIR,
            DATA_CACHE_DIR,
            DATA_TYPE_TO_SOURCE_TO_VERSION,
        )
    except Exception as exc:
        raise RuntimeError(
            "MolmoSpaces resource dependencies are not installed. Install the minimal "
            "resource stack with: python3 -m pip install --user "
            "compress-json molmospaces-resources==0.0.1b4 boto3 pydantic"
        ) from exc
    return {
        "assets_dir": ASSETS_DIR,
        "cache_dir": DATA_CACHE_DIR,
        "install_mode": InstallMode,
        "remote_storage": R2RemoteStorage,
        "resource_manager": ResourceManager,
        "versions": DATA_TYPE_TO_SOURCE_TO_VERSION,
    }


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        out.append(path)
    return out
