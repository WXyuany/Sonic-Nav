#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.datasets import (
    DEFAULT_MOLMOSPACES_BENCHMARK,
    MolmoSpacesBenchmark,
    MolmoSpacesEpisode,
    resolve_real_scene_assets,
)
from sonic_world.planners import TaskRequest, task_request_to_json


MODEL_DIR = REPO / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
DEFAULT_ANCHOR_OUT = Path("/tmp/sonic_molmospaces_anchor.json")
DEFAULT_REQUEST_OUT = Path("/tmp/sonic_molmospaces_task_request.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sonic/MuJoCo G1 interaction scene from a MolmoSpaces benchmark episode."
    )
    parser.add_argument("benchmark", nargs="?", default=str(DEFAULT_MOLMOSPACES_BENCHMARK))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--scene-mode",
        choices=["real", "proxy"],
        default="real",
        help="real uses the MolmoSpaces MJCF room/assets; proxy builds a small G1-friendly tabletop stand-in.",
    )
    parser.add_argument("--output", help="Output XML path. Defaults under gear_sonic/.../g1/.")
    parser.add_argument("--anchor-output", default=str(DEFAULT_ANCHOR_OUT))
    parser.add_argument("--request-output", default=str(DEFAULT_REQUEST_OUT))
    parser.add_argument("--no-context", action="store_true", help="Only include task-relevant objects.")
    parser.add_argument("--max-context-objects", type=int, default=10)
    parser.add_argument("--object-x", type=float, default=1.34, help="Local G1-friendly x position for the task object.")
    parser.add_argument("--surface-height", type=float, default=0.79, help="Table/support top height in the generated scene.")
    parser.add_argument("--raw-local-frame", action="store_true", help="Use MolmoSpaces base-relative coordinates directly.")
    parser.add_argument("--asset-variant", choices=["base", "ceiling"], default="base")
    parser.add_argument("--install-assets", dest="install_assets", action="store_true", default=True)
    parser.add_argument("--no-install-assets", dest="install_assets", action="store_false")
    parser.add_argument("--install-object-assets", dest="install_object_assets", action="store_true", default=True)
    parser.add_argument("--no-install-object-assets", dest="install_object_assets", action="store_false")
    parser.add_argument("--real-collisions", action="store_true", help="Keep imported MolmoSpaces geom contacts.")
    parser.add_argument(
        "--real-z-shift",
        type=float,
        default=0.0,
        help="Extra vertical shift for the imported real-scene visuals. Negative lowers the room.",
    )
    parser.add_argument(
        "--real-z-align",
        dest="real_z_align",
        action="store_true",
        help="Also subtract MolmoSpaces robot_base_pose.z from the imported scene and anchors.",
    )
    parser.add_argument("--no-real-z-align", dest="real_z_align", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(real_z_align=False)
    parser.add_argument("--show-markers", action="store_true", help="Add small task/object markers in real scene mode.")
    parser.add_argument("--validate", action="store_true", help="Load generated XML with MuJoCo after writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = MolmoSpacesBenchmark(Path(args.benchmark))
    episode = benchmark.episode(args.episode_index)
    request = benchmark.episode_task_request(episode)
    source_anchor = benchmark.episode_anchor(
        episode,
        include_context=not args.no_context,
        max_context_objects=args.max_context_objects,
    )
    real_scene = None
    real_z_offset = 0.0
    if args.scene_mode == "real":
        real_scene = resolve_real_scene_assets(
            episode,
            variant=args.asset_variant,
            install=args.install_assets,
            install_objects=args.install_object_assets,
        )
        real_z_offset = _real_scene_z_offset(episode, extra_shift=args.real_z_shift) if args.real_z_align else float(args.real_z_shift)
        anchor = localize_anchor_to_robot_frame(source_anchor, episode, z_offset=real_z_offset)
    else:
        anchor = localize_anchor_for_g1(
            source_anchor,
            request,
            object_x=args.object_x,
            surface_height=args.surface_height,
            remap_for_g1=not args.raw_local_frame,
        )

    output = Path(args.output) if args.output else default_scene_xml_path(
        episode.index,
        episode.scene_key,
        scene_mode=args.scene_mode,
    )
    if not output.is_absolute():
        output = REPO / output
    output.parent.mkdir(parents=True, exist_ok=True)
    include_file = os.path.relpath(MODEL_DIR / "g1_29dof_with_hand.xml", output.parent)
    if args.scene_mode == "real":
        assert real_scene is not None
        xml = build_real_molmospaces_mujoco_scene(
            episode,
            request,
            real_scene.scene_xml,
            include_file=Path(include_file).as_posix(),
            visual_only=not args.real_collisions,
            show_markers=args.show_markers,
            z_offset=real_z_offset,
        )
    else:
        xml = build_molmospaces_mujoco_scene(anchor, request, include_file=Path(include_file).as_posix())
    output.write_text(xml, encoding="utf-8")

    anchor_output = Path(args.anchor_output)
    request_output = Path(args.request_output)
    anchor_output.write_text(json.dumps(anchor, indent=2, sort_keys=True), encoding="utf-8")
    request_output.write_text(task_request_to_json(request) + "\n", encoding="utf-8")

    if args.validate:
        validate_mujoco_xml(output)

    print(f"Wrote MolmoSpaces MuJoCo scene: {output}")
    if real_scene is not None:
        object_archives = sum(len(values) for values in real_scene.installed_object_archives.values())
        print(f"Real MolmoSpaces scene: {real_scene.scene_xml}")
        print(f"Real scene archive: {real_scene.scene_archive}")
        print(f"Installed object archives: {object_archives}")
        print(f"Real scene Z offset: {real_z_offset:.4f} m")
        print("Sonic collision floor: z=0.0000 m")
    print(f"Wrote Sonic object anchor: {anchor_output}")
    print(f"Wrote Sonic task request: {request_output}")
    print(f"Episode: {episode.index} {episode.scene_key} | {episode.language_description}")
    print(f"Task: {request.verb} object={request.object_id} target={request.target_id}")
    print(f"Launch GUI with: python scripts/start.py {output.relative_to(REPO)}")
    print("Publish world-model inputs after launch with:")
    print(f"  /usr/bin/python3 scripts/tools/world_model_object_anchor.py --file {anchor_output}")
    print(f"  /usr/bin/python3 scripts/tools/world_model_task_request.py --file {request_output}")
    return 0


def default_scene_xml_path(episode_index: int, scene_key: str, *, scene_mode: str) -> Path:
    safe_scene = _safe_name(scene_key)
    return MODEL_DIR / f"scene_molmospaces_{scene_mode}_{safe_scene}_ep{episode_index:04d}.xml"


def _real_scene_z_offset(episode: MolmoSpacesEpisode, *, extra_shift: float = 0.0) -> float:
    robot_pos = episode.robot_position_map or (0.0, 0.0, 0.0)
    return -float(robot_pos[2]) + float(extra_shift)


def localize_anchor_to_robot_frame(
    anchor: dict[str, Any],
    episode: MolmoSpacesEpisode,
    *,
    z_offset: float = 0.0,
) -> dict[str, Any]:
    localized = json.loads(json.dumps(anchor))
    robot_pos = episode.robot_position_map or (0.0, 0.0, 0.0)
    robot_yaw = episode.robot_yaw_map or 0.0
    for record in localized.get("objects") or []:
        if not isinstance(record, dict):
            continue
        pose_map = record.get("pose_map")
        transformed = _transform_pose_payload_to_robot_frame(pose_map, robot_pos, robot_yaw, z_offset=z_offset)
        if transformed is None:
            continue
        record["pose_map"] = transformed
        record["pose_base"] = {**transformed, "frame_id": "base_link"}
        properties = record.get("properties")
        if isinstance(properties, dict):
            properties.setdefault("molmospaces_original_pose_map", pose_map)
            grasp = properties.get("grasp")
            if isinstance(grasp, dict):
                raw_base_target = grasp.get("base_target_map")
                if isinstance(raw_base_target, list) and len(raw_base_target) >= 3:
                    tx, ty, _ = _transform_xy(
                        (float(raw_base_target[0]), float(raw_base_target[1]), 0.0),
                        robot_pos,
                        robot_yaw,
                        z_offset=z_offset,
                    )
                    grasp["base_target_map"] = [tx, ty, _wrap_pi(float(raw_base_target[2]) - robot_yaw)]
    localized["robot_start_map"] = [0.0, 0.0, 0.0]
    localized["robot_start_yaw"] = 0.0
    localized.setdefault("properties", {})["molmospaces_localization"] = {
        "mode": "real_scene_robot_frame",
        "original_robot_pose_map": list(robot_pos),
        "original_robot_yaw": robot_yaw,
        "scene_z_offset": z_offset,
    }
    return localized


def build_real_molmospaces_mujoco_scene(
    episode: MolmoSpacesEpisode,
    request: TaskRequest,
    real_scene_xml: str | Path,
    *,
    include_file: str = "g1_29dof_with_hand.xml",
    visual_only: bool = True,
    show_markers: bool = False,
    z_offset: float = 0.0,
) -> str:
    scene_path = Path(real_scene_xml)
    real_root = ET.parse(scene_path).getroot()
    assets = _real_asset_xml(real_root, scene_path)
    real_world_body = _real_worldbody_xml(real_root, scene_path, episode, visual_only=visual_only, z_offset=z_offset)
    markers = _real_scene_marker_xml(episode, request, z_offset=z_offset) if show_markers else ""
    language = _xml(episode.language_description)
    scene_key = _xml(episode.scene_key)
    return f"""<mujoco model="g1_molmospaces_real_episode">
  <!-- Generated from real MolmoSpaces MJCF: scene={scene_key} task={_xml(request.verb)} language={language} -->
  <include file="{_xml(include_file)}"/>

  <statistic center="0 0 0.9" extent="9.0"/>

  <visual>
    <headlight diffuse="0.58 0.58 0.58" ambient="0.32 0.32 0.32" specular="0.08 0.08 0.08"/>
    <rgba haze="0.08 0.09 0.10 1"/>
    <global azimuth="-135" elevation="-22" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
{assets}
  </asset>

  <worldbody>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>
    <geom name="sonic_collision_floor" type="plane" size="0 0 0.05" pos="0 0 0"
      friction="1.1 0.02 0.001" solimp="0.95 0.99 0.001" solref="0.005 1" rgba="0 0 0 0"/>
{real_world_body}
{markers}
  </worldbody>

  <default>
    <geom friction="1.0 0.02 0.001" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </default>
</mujoco>
"""


def _real_asset_xml(real_root: ET.Element, scene_path: Path) -> str:
    asset = real_root.find("asset")
    if asset is None:
        return ""
    out: list[str] = []
    for child in list(asset):
        copied = _copy_real_element(child, scene_path.parent, visual_only=False, drop_dynamics=False)
        if copied is not None:
            out.append(_indent_xml(ET.tostring(copied, encoding="unicode"), spaces=4))
    return "\n".join(out)


def _real_worldbody_xml(
    real_root: ET.Element,
    scene_path: Path,
    episode: MolmoSpacesEpisode,
    *,
    visual_only: bool,
    z_offset: float,
) -> str:
    worldbody = real_root.find("worldbody")
    if worldbody is None:
        return ""
    robot_pos = episode.robot_position_map or (0.0, 0.0, 0.0)
    robot_yaw = episode.robot_yaw_map or 0.0
    inv_yaw = -robot_yaw
    offset = _inverse_xy_offset(robot_pos, robot_yaw)
    wrapper = ET.Element(
        "body",
        {
            "name": "molmospaces_real_world",
            "pos": f"{offset[0]:.6f} {offset[1]:.6f} {z_offset:.6f}",
            "quat": _yaw_quat_wxyz(inv_yaw),
        },
    )
    for child in list(worldbody):
        copied = _copy_real_element(
            child,
            scene_path.parent,
            visual_only=visual_only,
            drop_dynamics=True,
        )
        if copied is not None:
            wrapper.append(copied)
    return _indent_xml(ET.tostring(wrapper, encoding="unicode"), spaces=4)


def _copy_real_element(
    elem: ET.Element,
    scene_dir: Path,
    *,
    visual_only: bool,
    drop_dynamics: bool,
) -> ET.Element | None:
    if drop_dynamics and elem.tag in {"joint", "freejoint", "inertial"}:
        return None
    if drop_dynamics and elem.tag in {"actuator", "sensor", "tendon"}:
        return None
    attrib = dict(elem.attrib)
    attrib.pop("class", None)
    if "file" in attrib:
        attrib["file"] = _real_asset_file(scene_dir, attrib["file"])
    if elem.tag == "geom" and visual_only:
        attrib["contype"] = "0"
        attrib["conaffinity"] = "0"
        attrib.setdefault("group", "1")
        attrib.pop("density", None)
        attrib.pop("mass", None)
    copied = ET.Element(elem.tag, attrib)
    copied.text = elem.text
    copied.tail = elem.tail
    for child in list(elem):
        child_copy = _copy_real_element(
            child,
            scene_dir,
            visual_only=visual_only,
            drop_dynamics=drop_dynamics,
        )
        if child_copy is not None:
            copied.append(child_copy)
    return copied


def _real_asset_file(scene_dir: Path, file_attr: str) -> str:
    if os.path.isabs(file_attr):
        return file_attr
    return os.path.abspath(os.path.normpath(os.path.join(str(scene_dir), file_attr)))


def _real_scene_marker_xml(episode: MolmoSpacesEpisode, request: TaskRequest, *, z_offset: float = 0.0) -> str:
    pickup_pose = None
    if request.object_id:
        pickup_pose = episode.scene_object_poses.get(request.object_id)
    if pickup_pose is None:
        return ""
    robot_pos = episode.robot_position_map or (0.0, 0.0, 0.0)
    robot_yaw = episode.robot_yaw_map or 0.0
    x, y, z = _transform_xy((pickup_pose[0], pickup_pose[1], pickup_pose[2]), robot_pos, robot_yaw, z_offset=z_offset)
    return (
        f'    <site name="molmospaces_task_object_marker" type="sphere" size="0.08" '
        f'pos="{x:.4f} {y:.4f} {z:.4f}" rgba="0.15 0.55 1.0 0.55"/>'
    )


def _transform_pose_payload_to_robot_frame(
    pose: Any,
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    *,
    z_offset: float = 0.0,
) -> dict[str, Any] | None:
    if not isinstance(pose, dict):
        return None
    raw = _position_from_pose(pose)
    if raw is None:
        return None
    x, y, z = _transform_xy(raw, robot_pos, robot_yaw, z_offset=z_offset)
    out: dict[str, Any] = {"frame_id": "map", "position": [x, y, z]}
    if pose.get("yaw") is not None:
        try:
            out["yaw"] = _wrap_pi(float(pose["yaw"]) - robot_yaw)
        except (TypeError, ValueError):
            pass
    if isinstance(pose.get("orientation_xyzw"), list):
        out["orientation_xyzw"] = list(pose["orientation_xyzw"])
    return out


def _transform_xy(
    point: tuple[float, float, float],
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
    *,
    z_offset: float = 0.0,
) -> tuple[float, float, float]:
    dx = float(point[0]) - float(robot_pos[0])
    dy = float(point[1]) - float(robot_pos[1])
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)
    return (c * dx + s * dy, -s * dx + c * dy, float(point[2]) + float(z_offset))


def _inverse_xy_offset(
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
) -> tuple[float, float]:
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)
    tx = -float(robot_pos[0])
    ty = -float(robot_pos[1])
    return (c * tx + s * ty, -s * tx + c * ty)


def _yaw_quat_wxyz(yaw: float) -> str:
    return f"{math.cos(yaw * 0.5):.8f} 0 0 {math.sin(yaw * 0.5):.8f}"


def _indent_xml(text: str, *, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def localize_anchor_for_g1(
    anchor: dict[str, Any],
    request: TaskRequest,
    *,
    object_x: float,
    surface_height: float,
    remap_for_g1: bool,
) -> dict[str, Any]:
    localized = json.loads(json.dumps(anchor))
    objects = [record for record in localized.get("objects", []) if isinstance(record, dict)]
    if not objects:
        return localized

    task_obj = _record_by_id(objects, request.object_id) or _first_graspable(objects) or objects[0]
    target_obj = _record_by_id(objects, request.target_id)
    raw_task_base = _position_from_pose(task_obj.get("pose_base")) or _position_from_pose(task_obj.get("pose_map")) or (0.55, 0.0, surface_height)

    if remap_for_g1:
        task_half_z = _object_half_height(task_obj)
        task_y = _clamp(raw_task_base[1], -0.32, 0.32)
        task_pos = (float(object_x), task_y, surface_height + task_half_z)
    else:
        for record in objects:
            raw_pos = _position_from_pose(record.get("pose_base")) or _position_from_pose(record.get("pose_map")) or (0.8, 0.0, surface_height)
            half_z = _object_half_height(record)
            pos = (raw_pos[0], raw_pos[1], max(raw_pos[2], surface_height + half_z))
            _set_local_pose(record, pos)
        _finalize_local_anchor(localized)
        return localized

    task_id = str(task_obj.get("object_id"))
    target_id = str(target_obj.get("object_id")) if target_obj is not None else None
    for idx, record in enumerate(objects):
        raw_pos = _position_from_pose(record.get("pose_base")) or _position_from_pose(record.get("pose_map")) or raw_task_base
        half_z = _object_half_height(record)
        object_id = str(record.get("object_id") or "")
        if object_id == task_id:
            pos = task_pos
        elif target_id is not None and object_id == target_id:
            raw_target_base = _position_from_pose(record.get("pose_base"))
            dy = _clamp((raw_target_base[1] - raw_task_base[1]) if raw_target_base else 0.34, -0.46, 0.46)
            if abs(dy) < 0.18:
                dy = 0.34
            pos = (task_pos[0], _clamp(task_pos[1] + dy, -0.48, 0.48), surface_height + 0.012)
        elif _category(record) == "navigation_goal":
            pos = (min(2.4, task_pos[0] + 0.35 + idx * 0.05), 0.0, 0.28)
        else:
            dx = _clamp((raw_pos[0] - raw_task_base[0]) * 0.55, -0.55, 0.55)
            dy = _clamp((raw_pos[1] - raw_task_base[1]) * 0.55, -0.46, 0.46)
            pos = (task_pos[0] + dx, _clamp(task_pos[1] + dy, -0.55, 0.55), surface_height + half_z)
            if _distance_xy(pos, task_pos) < 0.12:
                pos = (pos[0] + 0.16, pos[1] + 0.16, pos[2])
        _set_local_pose(record, pos)

    _finalize_local_anchor(localized)
    localized.setdefault("properties", {})["molmospaces_localization"] = {
        "mode": "g1_interaction_proxy",
        "object_x": object_x,
        "surface_height": surface_height,
        "note": "MolmoSpaces episode coordinates remapped to a G1-reachable local interaction scene.",
    }
    return localized


def build_molmospaces_mujoco_scene(
    anchor: dict[str, Any],
    request: TaskRequest,
    *,
    include_file: str = "g1_29dof_with_hand.xml",
) -> str:
    objects = [record for record in anchor.get("objects", []) if isinstance(record, dict)]
    task_record = _record_by_id(objects, request.object_id) or _first_graspable(objects) or (objects[0] if objects else None)
    positions = [_position_from_pose(record.get("pose_map")) for record in objects]
    positions = [pos for pos in positions if pos is not None]
    table_records = [
        record
        for record in objects
        if _category(record) not in {"navigation_goal", "place_target"} and _object_role(record) != "navigation_goal"
    ]
    table_positions = [_position_from_pose(record.get("pose_map")) for record in table_records]
    table_positions = [pos for pos in table_positions if pos is not None]
    if not table_positions and task_record is not None:
        pos = _position_from_pose(task_record.get("pose_map"))
        if pos is not None:
            table_positions = [pos]
    table_top_z = _support_top_height(table_records, default=0.79)
    table_center, table_half = _table_geometry(table_positions, table_top_z)
    extent = max(7.0, 3.5 + max((math.hypot(pos[0], pos[1]) for pos in positions), default=1.5))

    material_defs = _material_defs()
    object_xml = "\n".join(_object_xml(record, request) for record in objects)
    guide_xml = _guide_xml(task_record, request)
    metadata = anchor.get("properties") or {}
    language = _xml(str(metadata.get("task_description") or request.metadata.get("task_description") or ""))
    scene = _xml(str(anchor.get("scene") or "molmospaces"))
    return f"""<mujoco model="g1_molmospaces_episode">
  <!-- Generated from MolmoSpaces: scene={scene} task={_xml(request.verb)} language={language} -->
  <include file="{_xml(include_file)}"/>

  <statistic center="0.9 0 0.75" extent="{extent:.2f}"/>

  <visual>
    <headlight diffuse="0.58 0.58 0.58" ambient="0.32 0.32 0.32" specular="0.08 0.08 0.08"/>
    <rgba haze="0.10 0.12 0.14 1"/>
    <global azimuth="-135" elevation="-22" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.62 0.72 0.82" rgb2="0.04 0.05 0.07" width="512" height="3072"/>
    <texture type="2d" name="molmo_floor_grid" builtin="checker" rgb1="0.20 0.22 0.23" rgb2="0.14 0.15 0.16" width="256" height="256"/>
    <material name="molmo_floor_mat" texture="molmo_floor_grid" texuniform="true" texrepeat="10 10" reflectance="0.08"/>
{material_defs}  </asset>

  <worldbody>
    <light name="molmo_key" pos="0 0 7" dir="0 0 -1" directional="true" diffuse="0.64 0.64 0.60"/>
    <light name="molmo_fill" pos="-3 -4 4" dir="0.4 0.5 -1" diffuse="0.24 0.24 0.24"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="molmo_floor_mat" friction="1.1 0.02 0.001"/>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>
    <geom name="molmo_back_wall" type="box" size="3.8 0.035 1.2" pos="1.3 2.05 1.2" material="molmo_wall_mat" contype="0" conaffinity="0" group="1"/>
    <geom name="molmo_left_wall" type="box" size="0.035 2.0 1.2" pos="-1.15 0.1 1.2" material="molmo_wall_mat" contype="0" conaffinity="0" group="1"/>
    <geom name="molmo_scene_panel" type="box" size="0.025 0.85 0.55" pos="{table_center[0] + table_half[0] + 0.28:.4f} {table_center[1] + table_half[1] + 0.20:.4f} {table_top_z + 0.55:.4f}" material="molmo_panel_mat" contype="0" conaffinity="0" group="1"/>

    <site name="molmo_walk_target_strip" type="box" size="0.02 0.52 0.006" pos="{max(0.8, table_center[0] - table_half[0] - 0.36):.4f} {table_center[1]:.4f} 0.012" material="molmo_guide_mat"/>
    <geom name="molmo_support_top" type="box" size="{table_half[0]:.4f} {table_half[1]:.4f} 0.035" pos="{table_center[0]:.4f} {table_center[1]:.4f} {table_top_z - 0.035:.4f}" material="molmo_table_mat"
      condim="6" friction="1.6 0.12 0.01" solref="0.003 1" solimp="0.99 0.999 0.0001"/>
    <geom name="molmo_support_front_edge" type="box" size="{table_half[0] + 0.01:.4f} 0.018 0.045" pos="{table_center[0]:.4f} {table_center[1] - table_half[1]:.4f} {table_top_z - 0.065:.4f}" material="molmo_table_edge_mat" contype="0" conaffinity="0" group="1"/>
    <geom name="molmo_support_back_edge" type="box" size="{table_half[0] + 0.01:.4f} 0.018 0.045" pos="{table_center[0]:.4f} {table_center[1] + table_half[1]:.4f} {table_top_z - 0.065:.4f}" material="molmo_table_edge_mat" contype="0" conaffinity="0" group="1"/>
    <geom name="molmo_support_leg_fl" type="cylinder" size="0.025 {max(0.1, (table_top_z - 0.07) * 0.5):.4f}" pos="{table_center[0] - table_half[0] + 0.11:.4f} {table_center[1] - table_half[1] + 0.10:.4f} {(table_top_z - 0.07) * 0.5:.4f}" material="molmo_table_mat"/>
    <geom name="molmo_support_leg_fr" type="cylinder" size="0.025 {max(0.1, (table_top_z - 0.07) * 0.5):.4f}" pos="{table_center[0] + table_half[0] - 0.11:.4f} {table_center[1] - table_half[1] + 0.10:.4f} {(table_top_z - 0.07) * 0.5:.4f}" material="molmo_table_mat"/>
    <geom name="molmo_support_leg_bl" type="cylinder" size="0.025 {max(0.1, (table_top_z - 0.07) * 0.5):.4f}" pos="{table_center[0] - table_half[0] + 0.11:.4f} {table_center[1] + table_half[1] - 0.10:.4f} {(table_top_z - 0.07) * 0.5:.4f}" material="molmo_table_mat"/>
    <geom name="molmo_support_leg_br" type="cylinder" size="0.025 {max(0.1, (table_top_z - 0.07) * 0.5):.4f}" pos="{table_center[0] + table_half[0] - 0.11:.4f} {table_center[1] + table_half[1] - 0.10:.4f} {(table_top_z - 0.07) * 0.5:.4f}" material="molmo_table_mat"/>
{guide_xml}
{object_xml}
  </worldbody>

  <default>
    <geom friction="1.0 0.02 0.001" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </default>
</mujoco>
"""


def validate_mujoco_xml(xml_path: Path) -> None:
    try:
        import mujoco
    except Exception as exc:
        raise SystemExit("MuJoCo validation requested, but the current Python cannot import mujoco.") from exc
    mujoco.MjModel.from_xml_path(str(xml_path))


def _object_xml(record: dict[str, Any], request: TaskRequest) -> str:
    category = _category(record)
    role = _object_role(record)
    object_id = str(record.get("object_id") or category or "object")
    safe = _safe_name(object_id)
    pos = _position_from_pose(record.get("pose_map")) or (1.2, 0.0, 0.85)
    shape = record.get("shape") if isinstance(record.get("shape"), dict) else {"kind": str(record.get("shape") or "box")}
    if category == "place_target" or shape.get("kind") == "target":
        return (
            f'    <site name="{safe}_place_target" type="cylinder" size="0.105 0.004" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" material="molmo_target_mat"/>\n'
            f'    <site name="{safe}_place_post" type="cylinder" size="0.016 0.30" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2] + 0.30:.4f}" rgba="0.20 0.95 0.35 0.35"/>'
        )
    if category == "navigation_goal":
        return (
            f'    <site name="{safe}_nav_goal" type="sphere" size="0.16" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {max(0.28, pos[2]):.4f}" material="molmo_goal_mat"/>'
        )
    dynamic = object_id == request.object_id
    material = _material_for_category(category, dynamic=dynamic)
    geom = _geom_xml_for_shape(shape, safe, material)
    if dynamic:
        mass = _mass_for_shape(shape, category)
        inertia = max(1e-6, mass * 0.001)
        return f"""    <body name="{safe}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">
      <freejoint name="{safe}_freejoint"/>
      <inertial pos="0 0 0" mass="{mass:.5f}" diaginertia="{inertia:.8f} {inertia:.8f} {inertia:.8f}"/>
{geom}
    </body>"""
    return f"""    <geom name="{safe}_context" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" {_geom_attrs_for_shape(shape)} material="{material}" contype="0" conaffinity="0" group="2"/>"""


def _geom_xml_for_shape(shape: dict[str, Any], safe: str, material: str) -> str:
    attrs = _geom_attrs_for_shape(shape)
    return (
        f'      <geom name="{safe}_visual" {attrs} material="{material}" '
        'condim="6" friction="6.0 0.45 0.05" solref="0.003 1" solimp="0.99 0.999 0.0001"/>'
    )


def _geom_attrs_for_shape(shape: dict[str, Any]) -> str:
    kind = str(shape.get("kind") or "box").lower()
    if kind == "sphere":
        radius = float(shape.get("radius") or 0.045)
        return f'type="sphere" size="{radius:.4f}"'
    if kind == "cylinder":
        radius = float(shape.get("radius") or 0.045)
        half_height = _object_half_height({"shape": shape})
        return f'type="cylinder" size="{radius:.4f} {half_height:.4f}"'
    sx, sy, sz = _shape_size(shape)
    return f'type="box" size="{sx * 0.5:.4f} {sy * 0.5:.4f} {sz * 0.5:.4f}"'


def _guide_xml(task_record: dict[str, Any] | None, request: TaskRequest) -> str:
    if task_record is None:
        return ""
    pos = _position_from_pose(task_record.get("pose_map"))
    if pos is None:
        return ""
    return (
        f'    <site name="molmo_task_object_marker" type="cylinder" size="0.135 0.004" '
        f'pos="{pos[0]:.4f} {pos[1]:.4f} 0.018" material="molmo_task_mat"/>\n'
        f'    <site name="molmo_reach_line" type="capsule" size="0.018 {max(0.2, pos[0] * 0.5):.4f}" '
        f'pos="{pos[0] * 0.5:.4f} {pos[1] * 0.5:.4f} 0.045" euler="0 1.5708 {math.atan2(pos[1], pos[0]):.4f}" '
        'material="molmo_guide_mat"/>'
    )


def _material_defs() -> str:
    return """    <material name="molmo_wall_mat" rgba="0.55 0.58 0.60 1"/>
    <material name="molmo_panel_mat" rgba="0.72 0.66 0.56 0.55"/>
    <material name="molmo_table_mat" rgba="0.46 0.34 0.23 1"/>
    <material name="molmo_table_edge_mat" rgba="0.26 0.18 0.12 1"/>
    <material name="molmo_task_mat" rgba="0.15 0.55 1.00 0.35"/>
    <material name="molmo_goal_mat" rgba="0.20 0.95 0.35 1"/>
    <material name="molmo_target_mat" rgba="0.20 0.95 0.35 0.35"/>
    <material name="molmo_guide_mat" rgba="0.12 0.72 0.95 0.45"/>
    <material name="molmo_pick_sphere_mat" rgba="0.95 0.28 0.24 1"/>
    <material name="molmo_pick_cylinder_mat" rgba="0.18 0.55 0.92 1"/>
    <material name="molmo_pick_box_mat" rgba="0.92 0.72 0.25 1"/>
    <material name="molmo_context_mat" rgba="0.72 0.74 0.76 0.38"/>"""


def _material_for_category(category: str, *, dynamic: bool) -> str:
    if not dynamic:
        return "molmo_context_mat"
    if category in {"ball", "sphere", "fruit", "apple", "orange", "tomato", "lemon", "lime", "peach"}:
        return "molmo_pick_sphere_mat"
    if category in {"cup", "mug", "bottle", "can", "vase", "jar", "container", "bowl"}:
        return "molmo_pick_cylinder_mat"
    return "molmo_pick_box_mat"


def _set_local_pose(record: dict[str, Any], pos: tuple[float, float, float]) -> None:
    old_map = record.get("pose_map")
    old_base = record.get("pose_base")
    properties = record.setdefault("properties", {})
    if isinstance(properties, dict):
        properties.setdefault("molmospaces_original_pose_map", old_map)
        properties.setdefault("molmospaces_original_pose_base", old_base)
    yaw = _yaw_from_pose(old_base) or _yaw_from_pose(old_map)
    payload = {"frame_id": "map", "position": [float(pos[0]), float(pos[1]), float(pos[2])]}
    base_payload = {"frame_id": "base_link", "position": [float(pos[0]), float(pos[1]), float(pos[2])]}
    if yaw is not None:
        payload["yaw"] = yaw
        base_payload["yaw"] = yaw
    record["pose_map"] = payload
    record["pose_base"] = base_payload


def _finalize_local_anchor(anchor: dict[str, Any]) -> None:
    anchor["frame_id"] = "map"
    anchor["robot_start_map"] = [0.0, 0.0, 0.0]
    anchor["robot_start_yaw"] = 0.0
    anchor.setdefault("properties", {})["scene_frame"] = "g1_local_interaction_proxy"


def _record_by_id(objects: list[dict[str, Any]], object_id: str | None) -> dict[str, Any] | None:
    if object_id is None:
        return None
    for record in objects:
        if str(record.get("object_id") or "") == object_id:
            return record
    return None


def _first_graspable(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in objects:
        if _category(record) not in {"place_target", "navigation_goal", "support_surface"}:
            return record
    return None


def _category(record: dict[str, Any]) -> str:
    return str(record.get("category") or "object").lower()


def _object_role(record: dict[str, Any]) -> str:
    properties = record.get("properties")
    if isinstance(properties, dict):
        return str(properties.get("role") or "")
    return ""


def _position_from_pose(pose: Any) -> tuple[float, float, float] | None:
    if not isinstance(pose, dict):
        return None
    raw = pose.get("position")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None


def _yaw_from_pose(pose: Any) -> float | None:
    if not isinstance(pose, dict) or pose.get("yaw") is None:
        return None
    try:
        return float(pose["yaw"])
    except (TypeError, ValueError):
        return None


def _object_half_height(record: dict[str, Any]) -> float:
    shape = record.get("shape") if isinstance(record.get("shape"), dict) else {}
    kind = str(shape.get("kind") or "").lower()
    if kind == "sphere":
        return max(0.015, float(shape.get("radius") or 0.045))
    if kind == "cylinder":
        size = shape.get("size")
        if isinstance(size, list) and len(size) >= 3:
            return max(0.012, float(size[2]) * 0.5)
        return max(0.012, float(shape.get("height") or 0.10) * 0.5)
    if kind == "target":
        return 0.006
    _, _, sz = _shape_size(shape)
    return max(0.008, sz * 0.5)


def _shape_size(shape: dict[str, Any]) -> tuple[float, float, float]:
    size = shape.get("size")
    if isinstance(size, list) and len(size) >= 3:
        return (max(0.01, float(size[0])), max(0.01, float(size[1])), max(0.01, float(size[2])))
    return (0.08, 0.08, 0.06)


def _support_top_height(records: list[dict[str, Any]], *, default: float) -> float:
    tops: list[float] = []
    for record in records:
        pos = _position_from_pose(record.get("pose_map"))
        if pos is None:
            continue
        tops.append(pos[2] - _object_half_height(record))
    if not tops:
        return default
    return sum(tops) / len(tops)


def _table_geometry(
    positions: list[tuple[float, float, float]],
    table_top_z: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not positions:
        return (1.34, 0.0), (0.55, 0.42)
    min_x = min(pos[0] for pos in positions)
    max_x = max(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    max_y = max(pos[1] for pos in positions)
    center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
    half = (
        max(0.50, min(0.95, (max_x - min_x) * 0.5 + 0.34)),
        max(0.38, min(0.70, (max_y - min_y) * 0.5 + 0.30)),
    )
    return center, half


def _mass_for_shape(shape: dict[str, Any], category: str) -> float:
    if category in {"knife", "fork", "spoon", "pencil", "pen"}:
        return 0.018
    if shape.get("kind") == "sphere":
        return 0.012
    if shape.get("kind") == "cylinder":
        return 0.025
    return 0.035


def _safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "molmo_object"
    if out[0].isdigit():
        out = f"obj_{out}"
    return out[:96]


def _xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _distance_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


if __name__ == "__main__":
    raise SystemExit(main())
