#!/usr/bin/env python3
"""Materialize a sequence_id as one MuJoCo scene plus an episode manifest.

The regular task-suite scenes are deliberately atomic.  This tool is the
boundary between their planning benchmark and a carry-state rollout: every
stage object and target is placed in one XML, and the simulator is started
once for the whole episode.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
for path in (SCRIPT_DIR, SCRIPTS_DIR, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene
from generate_sonic_task_suite import _dynamic_object_body, _ensure_task_materials, _indent_xml, _support_geom, _target_site
from sonic_world.task_suites import load_robocasa_task_suite
from sonic_world.world_model import TaskObjectRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a carry-state MuJoCo scene and manifest from one task-suite sequence.")
    parser.add_argument("--suite", default="configs/world_model/task_suites/sonic_general_v0.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--scene-output-dir", default="gear_sonic/data/robot_model/model_data/g1/episodes")
    parser.add_argument("--manifest-output-dir", default="configs/world_model/episodes")
    parser.add_argument("--name", default="", help="Artifact name; defaults to the sequence id.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_robocasa_task_suite(_repo_path(args.suite), repo_root=REPO)
    stages = _sequence_stages(suite.tasks, args.sequence)
    artifact = _safe_name(args.name or args.sequence)
    scene_output = _repo_path(args.scene_output_dir) / f"scene_sonic_episode_{artifact}.xml"
    manifest_output = _repo_path(args.manifest_output_dir) / f"{artifact}.json"
    if not args.overwrite:
        existing = [path for path in (scene_output, manifest_output) if path.exists()]
        if existing:
            raise SystemExit("artifact already exists; pass --overwrite: " + ", ".join(str(path) for path in existing))

    manifest = materialize_episode(stages, sequence_id=args.sequence, scene_output=scene_output, suite_path=args.suite)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"episode={args.sequence} stages={len(stages)} scene={_rel(scene_output)}")
    print(f"manifest={_rel(manifest_output)}")
    return 0


def materialize_episode(stages: list[Any], *, sequence_id: str, scene_output: Path, suite_path: str) -> dict[str, Any]:
    if not stages:
        raise ValueError("episode requires at least one stage")
    sources = {_source_scene(task) for task in stages}
    if len(sources) != 1:
        raise ValueError(f"sequence {sequence_id!r} spans source scenes and cannot preserve one physical state: {sorted(sources)}")
    source_scene = next(iter(sources))
    selection = resolve_scene(source_scene, repo_root=REPO)
    tree = ET.parse(selection.abs_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"scene has no worldbody: {selection.abs_path}")
    _ensure_task_materials(asset)

    support_added: set[str] = set()
    object_ids: set[str] = set()
    target_ids: set[str] = set()
    for task in stages:
        objects = [deepcopy(item) for item in task.objects]
        request = task.request
        object_id = str(request.object_id or "")
        target_id = str(request.target_id or "")
        pickup = _object(objects, object_id)
        if pickup is None:
            raise ValueError(f"stage {task.task_id} is missing requested object {object_id!r}")
        support_id = str(pickup.get("support") or "")
        support = _object(objects, support_id)
        if support is not None and support_id not in support_added:
            worldbody.append(_support_geom(support))
            support_added.add(support_id)
        if object_id in object_ids:
            raise ValueError(f"duplicate episode object id {object_id!r}")
        worldbody.append(_dynamic_object_body(pickup))
        object_ids.add(object_id)
        if target_id:
            target = _object(objects, target_id)
            if target is None:
                raise ValueError(f"stage {task.task_id} is missing requested target {target_id!r}")
            if target_id in target_ids:
                raise ValueError(f"duplicate episode target id {target_id!r}")
            worldbody.append(_target_site(target))
            target_ids.add(target_id)

    scene_output.parent.mkdir(parents=True, exist_ok=True)
    _rebase_direct_includes(root, source_dir=selection.abs_path.parent, output_dir=scene_output.parent)
    _set_assetdir(root, selection.abs_path.parent)
    _indent_xml(root)
    tree.write(scene_output, encoding="utf-8", xml_declaration=False)
    _validate_scene(scene_output)
    return {
        "schema": "sonic_world_model_episode_manifest_v0",
        "sequence_id": str(sequence_id),
        "execution_mode": "single_scene_carry_state",
        "suite_path": str(suite_path),
        "source_scene": str(source_scene),
        "scene": _rel(scene_output),
        "stage_count": len(stages),
        "stages": [_stage_manifest(task, index + 1) for index, task in enumerate(stages)],
    }


def _sequence_stages(tasks: tuple[Any, ...], sequence_id: str) -> list[Any]:
    stages = [task for task in tasks if str(task.metadata.get("sequence_id") or "") == str(sequence_id)]
    if not stages:
        raise ValueError(f"unknown sequence {sequence_id!r}")
    stages.sort(key=_stage_key)
    return stages


def _stage_key(task: Any) -> tuple[int, str]:
    value = task.metadata.get("sequence_stage") or task.metadata.get("stage_index")
    try:
        return int(value), str(task.task_id)
    except (TypeError, ValueError):
        match = re.search(r"stage_(\d+)", str(task.task_id))
        return (int(match.group(1)) if match else 9999, str(task.task_id))


def _source_scene(task: Any) -> str:
    source = str(task.metadata.get("source_scene") or "").strip()
    if source:
        return source
    scene = str(task.scene.scene_xml)
    if "scene_sonic_task_" in scene:
        raise ValueError(f"stage {task.task_id} has no source_scene metadata")
    return scene


def _stage_manifest(task: Any, stage_index: int) -> dict[str, Any]:
    registry = TaskObjectRegistry.from_task_case(task)
    request = task.request.to_dict()
    request["metadata"] = {**dict(request.get("metadata") or {}), "request_id": task.task_id, "sequence_id": task.metadata.get("sequence_id"), "stage_index": stage_index}
    return {
        "stage_index": stage_index,
        "task_id": task.task_id,
        "description": task.description,
        "request": request,
        "objects": [deepcopy(item) for item in task.objects],
        "object_registry": registry.to_dict(),
        "expect": deepcopy(task.expectation),
    }


def _object(objects: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    for obj in objects:
        if str(obj.get("object_id") or obj.get("id") or "") == object_id:
            return obj
    return None


def _validate_scene(path: Path) -> None:
    import mujoco

    mujoco.MjModel.from_xml_path(str(path))


def _rebase_direct_includes(root: ET.Element, *, source_dir: Path, output_dir: Path) -> None:
    """Keep source-scene include resolution valid when an episode is written elsewhere."""
    for include in root.findall("include"):
        raw = str(include.attrib.get("file") or "").strip()
        if not raw or Path(raw).is_absolute():
            continue
        source = (source_dir / raw).resolve()
        if source.exists():
            # The G1 include sets ``meshdir=meshes``.  MuJoCo resolves that
            # compiler path against the generated top-level XML, so clone it
            # with an absolute mesh root when the episode lives elsewhere.
            if _include_has_meshdir(source):
                rewritten = _write_rebased_include(source, output_dir)
                include.set("file", os.path.relpath(rewritten, start=output_dir))
            else:
                include.set("file", os.path.relpath(source, start=output_dir))


def _include_has_meshdir(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().find("compiler[@meshdir]") is not None
    except ET.ParseError:
        return False


def _write_rebased_include(source: Path, output_dir: Path) -> Path:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    output = output_dir / f"{source.stem}_episode_{digest}.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        compiler.set("meshdir", str((source.parent / str(compiler.get("meshdir"))).resolve()))
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def _set_assetdir(root: ET.Element, source_dir: Path) -> None:
    """Anchor main-scene textures and robot meshes to their original asset root."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("assetdir", str(source_dir.resolve()))


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
