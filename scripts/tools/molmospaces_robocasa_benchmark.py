#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene
from sonic_world.datasets import (
    DEFAULT_MOLMOSPACES_BENCHMARK,
    MolmoSpacesBenchmark,
    MolmoSpacesEpisode,
    explicit_affordances_for,
)
from sonic_world.scenarios import ScenarioSpec, replay_scenario
from sonic_world.task_suites import load_robocasa_task_suite


MODEL_DIR = REPO / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
DEFAULT_SUITE_OUT = REPO / "configs" / "world_model" / "task_suites" / "molmospaces_robocasa_v0.yaml"
DEFAULT_SCENE_DIR = MODEL_DIR
SCENE_TEMPLATES = ("robocasa_kitchen", "robocasa_galley", "robocasa_apartment", "robocasa_cafe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sonic/RoboCasa benchmark suite and matching MuJoCo scenes from MolmoSpaces episodes."
    )
    parser.add_argument("benchmark", nargs="?", default=str(DEFAULT_MOLMOSPACES_BENCHMARK))
    parser.add_argument("--output-suite", default=str(DEFAULT_SUITE_OUT))
    parser.add_argument("--scene-dir", default=str(DEFAULT_SCENE_DIR))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-kind", action="append", choices=["pick", "pick_place", "navigate", "door_open", "open_close"])
    parser.add_argument("--scene-dataset")
    parser.add_argument("--house-index", type=int)
    parser.add_argument("--surface-height", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-xml", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_kinds = tuple(args.task_kind or ("pick", "pick_place"))
    benchmark = MolmoSpacesBenchmark(Path(args.benchmark))
    episodes = _select_episodes(
        benchmark,
        task_kinds=task_kinds,
        start_index=args.start_index,
        limit=args.limit,
        scene_dataset=args.scene_dataset,
        house_index=args.house_index,
    )
    if not episodes:
        raise SystemExit("No MolmoSpaces episodes matched the requested filters.")

    scene_dir = _repo_path(args.scene_dir)
    suite_path = _repo_path(args.output_suite)
    if suite_path.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output suite already exists: {suite_path}. Use --overwrite to replace it.")

    tasks: list[dict[str, Any]] = []
    scene_paths: list[Path] = []
    for order, episode in enumerate(episodes):
        task, scene_xml = _task_from_episode(
            benchmark,
            episode,
            order=order,
            scene_dir=scene_dir,
            surface_height_override=args.surface_height,
        )
        tasks.append(task)
        scene_paths.append(scene_xml)
        if not args.dry_run:
            scene_xml.parent.mkdir(parents=True, exist_ok=True)
            _write_scene_xml(task, scene_xml)

    suite = _suite_payload(args.benchmark, tasks, episodes, scene_dir)
    if args.dry_run:
        print(yaml.safe_dump(suite, sort_keys=False, allow_unicode=False))
        return 0

    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=False), encoding="utf-8")

    if args.validate_xml:
        _validate_xml(scene_paths)
    if args.validate:
        loaded = load_robocasa_task_suite(suite_path, repo_root=REPO)
        for task in loaded.tasks:
            replay = replay_scenario(ScenarioSpec.from_dict(task.scenario()))
            if not replay.passed:
                raise RuntimeError(f"Generated task failed validation: {task.task_id}\n{json.dumps(replay.to_dict(), indent=2)}")

    print(f"Wrote MolmoSpaces->RoboCasa suite: {suite_path.relative_to(REPO)}")
    print(f"Wrote generated scenes: {scene_dir.relative_to(REPO)} ({len(scene_paths)} files)")
    for task in tasks:
        print(f"  {task['id']:28s} scene={task['scene']} request={task['request']['task']}")
    print("Preview with:")
    print(f"  python3 scripts/tools/robocasa_task_preview.py --suite {suite_path.relative_to(REPO)} --list")
    print("Launch one task with:")
    print(f"  python3 scripts/start_robocasa_task.py {tasks[0]['id']} --suite {suite_path.relative_to(REPO)}")
    return 0


def _select_episodes(
    benchmark: MolmoSpacesBenchmark,
    *,
    task_kinds: tuple[str, ...],
    start_index: int,
    limit: int,
    scene_dataset: str | None,
    house_index: int | None,
) -> list[MolmoSpacesEpisode]:
    selected: list[MolmoSpacesEpisode] = []
    for episode in benchmark.episodes(scene_dataset=scene_dataset, house_index=house_index):
        if episode.index < start_index:
            continue
        if episode.task_kind not in task_kinds:
            continue
        if episode.pickup_object_id is None:
            continue
        selected.append(episode)
        if len(selected) >= limit:
            break
    return selected


def _task_from_episode(
    benchmark: MolmoSpacesBenchmark,
    episode: MolmoSpacesEpisode,
    *,
    order: int,
    scene_dir: Path,
    surface_height_override: float | None,
) -> tuple[dict[str, Any], Path]:
    request = benchmark.episode_task_request(episode)
    anchor = benchmark.episode_anchor(episode, include_context=False, max_context_objects=0)
    source_record = _record_by_id(anchor.get("objects") or [], request.object_id) or _first_pickup(anchor.get("objects") or [])
    if source_record is None:
        raise ValueError(f"Episode {episode.index} has no usable pickup object record.")

    category = str(source_record.get("category") or "object")
    shape = _normalize_shape(source_record.get("shape"), category)
    affordance = _preferred_affordance(category, shape)
    task_id = _task_id(episode, category, request.verb)
    scene_template = _scene_template_for(category, request.verb, order)
    surface_height = surface_height_override if surface_height_override is not None else _surface_height_for(category, shape)
    support_id = f"{task_id}_support"
    object_id = f"{task_id}_object"
    target_id = f"{task_id}_target"

    object_y = _clamp(_source_y(source_record), -0.26, 0.26)
    object_x = 1.46 + 0.08 * (order % 3)
    object_z = surface_height + _shape_half_height(shape)
    reach_x = _reach_x_for(affordance)
    base_target_x = object_x - reach_x
    target_y = _target_y(object_y)
    target_z = surface_height + 0.012

    grasp = _grasp_params(affordance, shape, object_y, reach_x, object_z - surface_height, base_target_x)
    task_object: dict[str, Any] = {
        "object_id": object_id,
        "category": category,
        "shape": shape,
        "pose_map": {"frame_id": "map", "position": [_round(object_x), _round(object_y), _round(object_z)]},
        "pose_base": {"frame_id": "base_link", "position": [_round(reach_x), _round(object_y), _round(object_z - surface_height)]},
        "pose_camera": {"frame_id": "camera_depth_optical_frame", "position": [_round(0.18), _round(object_y * -0.25), _round(0.62)]},
        "support": support_id,
        "grasp": grasp,
        "properties": {
            "molmospaces_episode_index": episode.index,
            "molmospaces_scene_key": episode.scene_key,
            "molmospaces_object_id": request.object_id,
            "language": episode.language_description,
        },
    }
    explicit = explicit_affordances_for(category, shape, "pickup")
    if explicit:
        task_object["affordances"] = explicit

    objects = [
        task_object,
        _support_object(support_id, surface_height, object_x, object_y, category),
    ]
    relations = [{"subject": object_id, "relation": "on", "object": support_id, "confidence": 1.0}]
    request_payload: dict[str, Any] = {"task": "pick", "object": object_id}
    tags = ["molmospaces", episode.task_kind, affordance, category]
    if request.verb == "pick_place":
        objects.insert(
            1,
            {
                "object_id": target_id,
                "category": "place_target",
                "shape": "target",
                "pose_map": {"frame_id": "map", "position": [_round(object_x), _round(target_y), _round(target_z)]},
                "pose_base": {"frame_id": "base_link", "position": [_round(reach_x), _round(target_y), _round(target_z - surface_height)]},
                "support": support_id,
            },
        )
        relations.append({"subject": target_id, "relation": "on", "object": support_id, "confidence": 1.0})
        request_payload = {"task": "move", "object": object_id, "target": target_id}

    scene_xml = scene_dir / f"scene_{task_id}.xml"
    task = {
        "id": task_id,
        "description": episode.language_description,
        "scene": _rel(scene_xml),
        "tags": tags,
        "metadata": {
            "source": "molmospaces_robocasa_generator",
            "molmospaces_episode_index": episode.index,
            "molmospaces_episode_id": episode.episode_id,
            "molmospaces_scene_key": episode.scene_key,
            "molmospaces_task_kind": episode.task_kind,
            "scene_template": scene_template,
            "generated_scene_xml": _rel(scene_xml),
        },
        "objects": objects,
        "relations": relations,
        "request": request_payload,
        "expect": _expectation(affordance, request.verb),
    }
    task["_scene_generation"] = {
        "template": scene_template,
        "object_id": object_id,
        "target_id": target_id if request.verb == "pick_place" else None,
        "support_id": support_id,
        "surface_height": surface_height,
        "object_pose": [object_x, object_y, object_z],
        "target_pose": [object_x, target_y, target_z],
        "shape": shape,
        "category": category,
        "affordance": affordance,
    }
    return task, scene_xml


def _suite_payload(source_benchmark: str, tasks: list[dict[str, Any]], episodes: list[MolmoSpacesEpisode], scene_dir: Path) -> dict[str, Any]:
    clean_tasks = []
    for task in tasks:
        item = dict(task)
        item.pop("_scene_generation", None)
        clean_tasks.append(item)
    return {
        "name": "molmospaces_robocasa_benchmark",
        "version": "v0",
        "description": (
            "MolmoSpaces-derived benchmark tasks instantiated as RoboCasa/Sonic MuJoCo scenes. "
            "MolmoSpaces supplies language, object categories, shapes, and task intent; generated "
            "RoboCasa-style scenes supply stable G1 physics and contacts."
        ),
        "metadata": {
            "provider": "molmospaces_to_robocasa",
            "source_benchmark": str(source_benchmark),
            "generated_scene_dir": _rel(scene_dir),
            "episode_indices": [episode.index for episode in episodes],
            "evaluation_modes": ["privileged_anchor", "visual_perception"],
        },
        "tasks": clean_tasks,
    }


def _write_scene_xml(task: dict[str, Any], output: Path) -> None:
    generation = task["_scene_generation"]
    template = resolve_scene(generation["template"], repo_root=REPO)
    root = ET.parse(template.abs_path).getroot()
    _rewrite_file_attrs(root, template.abs_path.parent, output.parent)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _ensure_materials(asset)
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    for elem in _task_scene_elements(task):
        worldbody.append(elem)
    root.set("model", f"g1_{task['id']}")
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _rewrite_file_attrs(root: ET.Element, source_dir: Path, output_dir: Path) -> None:
    for elem in root.iter():
        file_attr = elem.attrib.get("file")
        if not file_attr or os.path.isabs(file_attr):
            continue
        source_file = (source_dir / file_attr).resolve()
        elem.set("file", os.path.relpath(source_file, output_dir))


def _ensure_materials(asset: ET.Element) -> None:
    existing = {child.attrib.get("name") for child in asset if child.tag == "material"}
    for name, rgba in {
        "ms_bench_support_mat": "0.48 0.38 0.28 1",
        "ms_bench_support_top_mat": "0.68 0.64 0.56 1",
        "ms_bench_object_mat": "0.16 0.55 0.92 1",
        "ms_bench_box_mat": "0.90 0.72 0.22 1",
        "ms_bench_sphere_mat": "0.20 0.75 0.38 1",
        "ms_bench_target_mat": "0.18 0.90 0.36 0.42",
    }.items():
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _task_scene_elements(task: dict[str, Any]) -> list[ET.Element]:
    generation = task["_scene_generation"]
    static_body = ET.Element("body", {"name": f"{task['id']}_generated_task"})
    x, y, z = generation["object_pose"]
    tx, ty, tz = generation["target_pose"]
    support_id = _safe_name(generation["support_id"])
    support_height = float(generation["surface_height"])
    support_size = _support_size(generation["category"])
    ET.SubElement(
        static_body,
        "geom",
        {
            "name": f"{support_id}_collision",
            "type": "box",
            "size": f"{support_size[0]:.4f} {support_size[1]:.4f} {support_height * 0.5:.4f}",
            "pos": f"{x:.4f} {y:.4f} {support_height * 0.5:.4f}",
            "material": "ms_bench_support_mat",
            "friction": "1.2 0.04 0.002",
        },
    )
    ET.SubElement(
        static_body,
        "site",
        {
            "name": f"{support_id}_top",
            "type": "box",
            "size": f"{support_size[0] + 0.04:.4f} {support_size[1] + 0.04:.4f} 0.0180",
            "pos": f"{x:.4f} {y:.4f} {support_height + 0.018:.4f}",
            "material": "ms_bench_support_top_mat",
        },
    )
    elements = [
        static_body,
        _dynamic_object_body(_safe_name(generation["object_id"]), generation["shape"], generation["category"], (x, y, z)),
    ]
    if generation.get("target_id"):
        ET.SubElement(
            static_body,
            "site",
            {
                "name": f"{_safe_name(generation['target_id'])}_site",
                "type": "cylinder",
                "size": "0.1200 0.0060",
                "pos": f"{tx:.4f} {ty:.4f} {tz:.4f}",
                "material": "ms_bench_target_mat",
            },
        )
    return elements


def _dynamic_object_body(name: str, shape: dict[str, Any], category: str, pos: tuple[float, float, float]) -> ET.Element:
    body = ET.Element("body", {"name": name, "pos": f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"})
    ET.SubElement(body, "freejoint", {"name": f"{name}_freejoint"})
    mass = _mass_for_shape(shape)
    inertia = max(1e-6, mass * 0.001)
    ET.SubElement(
        body,
        "inertial",
        {"pos": "0 0 0", "mass": f"{mass:.5f}", "diaginertia": f"{inertia:.8f} {inertia:.8f} {inertia:.8f}"},
    )
    attrs = _geom_attrs(shape)
    material = "ms_bench_box_mat" if shape["kind"] == "box" else "ms_bench_sphere_mat" if shape["kind"] == "sphere" else "ms_bench_object_mat"
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_geom",
            **attrs,
            "material": material,
            "condim": "6",
            "friction": "3.5 0.30 0.03",
            "solref": "0.003 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )
    return body


def _validate_xml(paths: list[Path]) -> None:
    import mujoco

    for path in paths:
        mujoco.MjModel.from_xml_path(str(path))


def _record_by_id(records: list[Any], object_id: str | None) -> dict[str, Any] | None:
    if object_id is None:
        return None
    for record in records:
        if isinstance(record, dict) and record.get("object_id") == object_id:
            return record
    return None


def _first_pickup(records: list[Any]) -> dict[str, Any] | None:
    for record in records:
        if isinstance(record, dict) and (record.get("properties") or {}).get("role") == "pickup":
            return record
    for record in records:
        if isinstance(record, dict):
            return record
    return None


def _normalize_shape(raw: Any, category: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or "box").lower()
        out = dict(raw)
        out["kind"] = kind
    elif isinstance(raw, str) and raw == "target":
        out = {"kind": "target"}
    else:
        out = {"kind": "box", "size": [0.08, 0.08, 0.06]}
    if out["kind"] == "sphere":
        out["radius"] = float(out.get("radius") or 0.045)
    elif out["kind"] == "cylinder":
        radius = float(out.get("radius") or 0.045)
        height = _shape_size(out)[2]
        out["radius"] = radius
        out["size"] = [radius * 2.0, radius * 2.0, height]
    elif out["kind"] != "target":
        out["kind"] = "box"
        out["size"] = list(_shape_size(out))
    if category in {"package", "box", "cube"} and out["kind"] == "box":
        sx, sy, sz = _shape_size(out)
        out["size"] = [max(0.14, sx), max(0.10, sy), max(0.08, sz)]
    return out


def _preferred_affordance(category: str, shape: dict[str, Any]) -> str:
    explicit = explicit_affordances_for(category, shape, "pickup")
    if explicit:
        return str(explicit[0]["name"])
    kind = str(shape.get("kind") or "box")
    if kind == "sphere":
        return "single_hand_pinch"
    if kind == "cylinder":
        return "side_grasp"
    if kind == "box":
        sx, sy, sz = _shape_size(shape)
        if max(sx, sy, sz) <= 0.12:
            return "single_hand_pinch"
        if sz <= 0.04:
            return "top_grasp"
    return "bimanual_clamp"


def _scene_template_for(category: str, verb: str, order: int) -> str:
    if category in {"cup", "mug", "bottle", "can", "container", "jar"}:
        return "robocasa_galley" if order % 2 else "robocasa_kitchen"
    if category in {"ball", "fruit", "apple", "orange", "tomato", "lemon", "lime", "peach"}:
        return "robocasa_cafe"
    if category in {"package", "box", "pillow", "cube"}:
        return "robocasa_apartment"
    if verb == "pick_place":
        return "robocasa_kitchen"
    return SCENE_TEMPLATES[order % len(SCENE_TEMPLATES)]


def _support_object(support_id: str, surface_height: float, x: float, y: float, category: str) -> dict[str, Any]:
    sx, sy = _support_size(category)
    support_category = "counter" if surface_height >= 0.84 else "table"
    return {
        "object_id": support_id,
        "category": support_category,
        "shape": {"kind": "box", "size": [_round(sx * 2.0), _round(sy * 2.0), _round(surface_height)]},
        "pose_map": {"frame_id": "map", "position": [_round(x), _round(y), _round(surface_height * 0.5)]},
    }


def _support_size(category: str) -> tuple[float, float]:
    if category in {"cup", "mug", "bottle", "can", "container", "jar"}:
        return (0.72, 0.34)
    return (0.76, 0.38)


def _surface_height_for(category: str, shape: dict[str, Any]) -> float:
    if category in {"cup", "mug", "bottle", "can", "container", "jar"}:
        return 0.86
    return 0.78


def _source_y(record: dict[str, Any]) -> float:
    pose = record.get("pose_base") if isinstance(record.get("pose_base"), dict) else record.get("pose_map")
    if isinstance(pose, dict):
        pos = pose.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            try:
                return float(pos[1])
            except (TypeError, ValueError):
                pass
    return -0.18


def _target_y(object_y: float) -> float:
    if object_y <= 0:
        return _clamp(object_y + 0.36, -0.32, 0.36)
    return _clamp(object_y - 0.36, -0.32, 0.36)


def _grasp_params(
    affordance: str,
    shape: dict[str, Any],
    y: float,
    reach_x: float,
    reach_z: float,
    base_target_x: float,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "approach_target_x": _round(reach_x),
        "walk_duration": 4.2,
        "walk_speed": 0.23,
        "target_y": _round(y),
        "reach_x": _round(reach_x),
        "reach_z": _round(max(0.025, reach_z)),
        "base_target_map": [_round(base_target_x), _round(y * 0.65), 0.0],
    }
    if affordance == "bimanual_clamp":
        sx, sy, _ = _shape_size(shape)
        params.update({"open_y": _round(max(0.20, sy + 0.10)), "clamp_y": _round(max(0.08, sy * 0.50)), "lift_z": 0.12})
    elif affordance == "side_grasp":
        params.update({"hand": "right", "radius": _round(float(shape.get("radius") or _size_radius(shape))), "height": _round(_shape_size(shape)[2])})
    elif affordance == "top_grasp":
        params.update({"hand": "right", "aperture": _round(max(0.04, _size_radius(shape) * 2.0))})
    else:
        params.update({"hand": "right", "radius": _round(float(shape.get("radius") or _size_radius(shape)))})
    return params


def _expectation(affordance: str, verb: str) -> dict[str, Any]:
    steps = ["navigate.approach_object", "manip.align_workspace", f"manip.{affordance}", "manip.lift_object"]
    if verb == "pick_place":
        steps.extend(["manip.transport_object", "manip.place_object", "manip.release"])
    return {
        "steps": steps,
        "demo_kind": "box" if affordance == "bimanual_clamp" else "ball",
        "grasp_affordance": affordance,
        "missing_skills": [],
        "unready_count": 0,
        "contract_error_count": 0,
        "decision_status": "ready_to_execute",
    }


def _geom_attrs(shape: dict[str, Any]) -> dict[str, str]:
    kind = str(shape.get("kind") or "box")
    if kind == "sphere":
        return {"type": "sphere", "size": f"{float(shape.get('radius') or 0.045):.4f}"}
    if kind == "cylinder":
        radius = float(shape.get("radius") or 0.045)
        return {"type": "cylinder", "size": f"{radius:.4f} {_shape_half_height(shape):.4f}"}
    sx, sy, sz = _shape_size(shape)
    return {"type": "box", "size": f"{sx * 0.5:.4f} {sy * 0.5:.4f} {sz * 0.5:.4f}"}


def _shape_size(shape: dict[str, Any]) -> tuple[float, float, float]:
    size = shape.get("size")
    if isinstance(size, list) and len(size) >= 3:
        return (float(size[0]), float(size[1]), float(size[2]))
    radius = float(shape.get("radius") or 0.045)
    if shape.get("kind") == "sphere":
        return (radius * 2.0, radius * 2.0, radius * 2.0)
    return (0.08, 0.08, 0.10)


def _shape_half_height(shape: dict[str, Any]) -> float:
    if shape.get("kind") == "sphere":
        return float(shape.get("radius") or 0.045)
    return _shape_size(shape)[2] * 0.5


def _size_radius(shape: dict[str, Any]) -> float:
    sx, sy, _ = _shape_size(shape)
    return max(0.02, min(abs(sx), abs(sy)) * 0.5)


def _mass_for_shape(shape: dict[str, Any]) -> float:
    sx, sy, sz = _shape_size(shape)
    volume = max(1e-5, sx * sy * sz)
    return _clamp(volume * 22.0, 0.025, 0.18)


def _reach_x_for(affordance: str) -> float:
    if affordance == "bimanual_clamp":
        return 0.50
    if affordance == "side_grasp":
        return 0.52
    if affordance == "top_grasp":
        return 0.50
    return 0.54


def _task_id(episode: MolmoSpacesEpisode, category: str, verb: str) -> str:
    return _safe_name(f"ms_ep{episode.index:04d}_{category}_{verb}")[:72]


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_name(value: str) -> str:
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else "_")
    safe = "".join(out).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "task"


def _round(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(float(value)):
        return low
    return max(low, min(high, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
