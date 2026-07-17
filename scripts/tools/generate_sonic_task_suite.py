#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene


DEFAULT_OUTPUT = Path("configs/world_model/task_suites/sonic_general_v0.yaml")
DEFAULT_SCENE_DIR = Path("gear_sonic/data/robot_model/model_data/g1")
DEFAULT_TARGET_COUNT = 500
TASK_OBJECT_FRONT_OFFSET_X = 0.24
RIGHT_HAND_OBJECT_Y_BAND = (-0.26, -0.10)
BIMANUAL_OBJECT_Y_BAND = (-0.18, 0.18)


SCENES = ("robocasa_kitchen", "robocasa_galley", "robocasa_apartment", "robocasa_cafe")
SCENE_SHORT = {
    "robocasa_kitchen": "kitchen",
    "robocasa_galley": "galley",
    "robocasa_apartment": "apartment",
    "robocasa_cafe": "cafe",
}


OBJECT_SPECS: dict[str, dict[str, Any]] = {
    "ball": {
        "category": "ball",
        "shape": {"kind": "sphere", "radius": 0.045},
        "grasp_affordance": "single_hand_pinch",
        "demo_kind": "ball",
        "height": 0.84,
        "base_z": 0.03,
    },
    "fruit": {
        "category": "fruit",
        "shape": {"kind": "sphere", "radius": 0.040},
        "grasp_affordance": "single_hand_pinch",
        "demo_kind": "ball",
        "height": 0.83,
        "base_z": 0.035,
    },
    "apple": {
        "category": "fruit",
        "shape": {"kind": "sphere", "radius": 0.048},
        "grasp_affordance": "single_hand_pinch",
        "demo_kind": "ball",
        "height": 0.845,
        "base_z": 0.040,
    },
    "orange": {
        "category": "fruit",
        "shape": {"kind": "sphere", "radius": 0.055},
        "grasp_affordance": "single_hand_pinch",
        "demo_kind": "ball",
        "height": 0.852,
        "base_z": 0.050,
    },
    "cube": {
        "category": "cube",
        "shape": {"kind": "box", "size": [0.10, 0.10, 0.09]},
        "grasp_affordance": "bimanual_clamp",
        "demo_kind": "box",
        "height": 0.84,
        "base_z": 0.04,
        "grasp": {"open_y": 0.14, "clamp_y": 0.06, "radius": 0.04},
    },
    "small_box": {
        "category": "small_package",
        "shape": {"kind": "box", "size": [0.12, 0.09, 0.08]},
        "grasp_affordance": "bimanual_clamp",
        "demo_kind": "box",
        "height": 0.835,
        "base_z": 0.035,
        "grasp": {"open_y": 0.17, "clamp_y": 0.075, "lift_z": 0.10},
    },
    "snack_box": {
        "category": "box",
        "shape": {"kind": "box", "size": [0.16, 0.09, 0.19]},
        "grasp_affordance": "bimanual_clamp",
        "demo_kind": "box",
        "height": 0.895,
        "base_z": 0.095,
        "grasp": {"open_y": 0.19, "clamp_y": 0.085, "lift_z": 0.14},
    },
    "package": {
        "category": "package",
        "shape": {"kind": "box", "size": [0.20, 0.14, 0.13]},
        "grasp_affordance": "bimanual_clamp",
        "demo_kind": "box",
        "height": 0.86,
        "base_z": 0.05,
        "grasp": {"open_y": 0.24, "clamp_y": 0.11, "lift_z": 0.12},
    },
    "bottle": {
        "category": "bottle",
        "shape": {"kind": "cylinder", "radius": 0.035, "size": [0.07, 0.07, 0.18]},
        "grasp_affordance": "side_grasp",
        "demo_kind": "ball",
        "height": 0.91,
        "base_z": 0.09,
    },
    "can": {
        "category": "can",
        "shape": {"kind": "cylinder", "radius": 0.032, "size": [0.064, 0.064, 0.12]},
        "grasp_affordance": "side_grasp",
        "demo_kind": "ball",
        "height": 0.88,
        "base_z": 0.06,
    },
    "cup": {
        "category": "cup",
        "shape": {"kind": "cylinder", "radius": 0.042, "size": [0.084, 0.084, 0.10]},
        "grasp_affordance": "side_grasp",
        "demo_kind": "ball",
        "height": 0.875,
        "base_z": 0.055,
    },
    "mug": {
        "category": "mug",
        "shape": {"kind": "cylinder", "radius": 0.040, "size": [0.08, 0.08, 0.11]},
        "grasp_affordance": "side_grasp",
        "demo_kind": "ball",
        "height": 0.90,
        "base_z": 0.08,
    },
    "bowl": {
        "category": "bowl",
        "shape": {"kind": "cylinder", "radius": 0.070, "size": [0.14, 0.14, 0.07]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.845,
        "base_z": 0.040,
        "grasp": {"aperture": 0.14, "reach_z": 0.04},
    },
    "plate": {
        "category": "plate",
        "shape": {"kind": "flat", "size": [0.18, 0.18, 0.025]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.810,
        "base_z": 0.012,
        "grasp": {"aperture": 0.18, "reach_z": 0.02},
    },
    "cloth": {
        "category": "cloth",
        "shape": {"kind": "flat", "size": [0.18, 0.12, 0.018]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.80,
        "base_z": 0.01,
        "grasp": {"aperture": 0.12, "reach_z": 0.03},
    },
    "sponge": {
        "category": "sponge",
        "shape": {"kind": "flat", "size": [0.12, 0.08, 0.035]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.818,
        "base_z": 0.020,
        "grasp": {"aperture": 0.09, "reach_z": 0.025},
    },
    "book": {
        "category": "book",
        "shape": {"kind": "flat", "size": [0.22, 0.15, 0.035]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.822,
        "base_z": 0.020,
        "grasp": {"aperture": 0.16, "reach_z": 0.030},
    },
    "remote": {
        "category": "tool",
        "shape": {"kind": "box", "size": [0.16, 0.045, 0.028]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.815,
        "base_z": 0.018,
        "grasp": {"aperture": 0.07, "reach_z": 0.025},
    },
    "utensil": {
        "category": "tool",
        "shape": {"kind": "box", "size": [0.19, 0.025, 0.018]},
        "grasp_affordance": "top_grasp",
        "demo_kind": "ball",
        "height": 0.805,
        "base_z": 0.012,
        "grasp": {"aperture": 0.05, "reach_z": 0.020},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Sonic general world-model benchmark task suite.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scene-dir", default=str(DEFAULT_SCENE_DIR))
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--no-scenes", action="store_true", help="Only write the YAML task suite; do not materialize MuJoCo XML scenes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print", dest="print_yaml", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = build_suite(target_count=max(0, int(args.target_count)))
    if not args.no_scenes:
        _materialize_task_scenes(suite["tasks"], _repo_path(args.scene_dir))
    text = yaml.safe_dump(suite, sort_keys=False, allow_unicode=False, width=120)
    if args.print_yaml:
        print(text)
        return 0

    output = _repo_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    scene_note = "" if args.no_scenes else f" + {len(suite['tasks'])} MuJoCo scenes"
    print(f"Wrote task suite: {output.relative_to(REPO)} ({len(suite['tasks'])} tasks{scene_note})")
    return 0


def build_suite(*, target_count: int = DEFAULT_TARGET_COUNT) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    tasks.extend(_short_tabletop_tasks())
    tasks.extend(_dense_tabletop_tasks())
    tasks.extend(_clutter_tabletop_tasks())
    tasks.extend(_navigation_manipulation_tasks())
    tasks.extend(_cross_scene_generalization_tasks())
    tasks.extend(_long_sequence_tasks())
    tasks.extend(_programmatic_expansion_tasks(target_count=target_count, existing=tasks))
    coverage = _coverage_summary(tasks)
    return {
        "name": "sonic_general_benchmark",
        "version": "v0",
        "description": (
            "General Sonic world-model benchmark suite: tabletop manipulation, navigation-conditioned "
            "manipulation, and long-horizon sequence stages. Each task is atomic for planning/evaluation; "
            "sequence metadata groups stages into longer episodes."
        ),
        "metadata": {
            "provider": "sonic_world",
            "role": "general_simulator_benchmark",
            "task_count": len(tasks),
            "coverage": coverage,
            "reset_policy": "reuse_deploy_reset_mujoco_state",
            "sensor_contract": ["odom", "tf", "lidar", "rgb", "depth", "object_anchor", "world_model"],
            "training_route": "privileged_anchor_first_then_vla_anchor_policy",
            "controller_training": "frozen_sonic_no_low_level_training",
            "policy_training_scope": "task_and_skill_policy_only_no_raw_joint_control",
            "trainable_outputs": [
                "task_intent",
                "object_target_anchors",
                "skill_selection",
                "base_goal",
                "hand_pose_target",
                "wrist_target",
                "grasp_close_ratio",
                "grasp_offsets",
                "lift_place_targets",
                "recovery_decision",
            ],
        },
        "tasks": tasks,
    }


def _short_tabletop_tasks() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    specs = [
        ("ball_left_to_tray", "ball", "robocasa_kitchen", (-0.26, 0.12), "move"),
        ("fruit_right_to_plate", "fruit", "robocasa_cafe", (0.22, -0.14), "move"),
        ("small_cube_pick", "cube", "robocasa_apartment", (-0.16, None), "pick"),
        ("package_clamp_lift", "package", "robocasa_apartment", (-0.08, None), "pick"),
        ("bottle_to_tray", "bottle", "robocasa_kitchen", (-0.18, 0.16), "move"),
        ("mug_counter_to_table", "mug", "robocasa_galley", (-0.20, 0.22), "move"),
        ("cloth_top_pick", "cloth", "robocasa_galley", (-0.10, None), "pick"),
    ]
    for idx, (task_id, kind, scene, y_pair, verb) in enumerate(specs):
        object_y, target_y = y_pair
        out.append(
            _task(
                task_id=task_id,
                description=f"{verb} {kind} in a short tabletop interaction.",
                scene=scene,
                kind=kind,
                object_id=f"sg_{kind}",
                object_x=1.55 + 0.04 * (idx % 3),
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["tabletop", "short", verb],
            )
        )
    return out


def _dense_tabletop_tasks() -> list[dict[str, Any]]:
    specs = [
        ("apple_to_plate", "apple", "robocasa_kitchen", -0.30, 0.12, "move"),
        ("orange_to_bowl_zone", "orange", "robocasa_cafe", 0.26, -0.12, "move"),
        ("can_to_counter_target", "can", "robocasa_galley", -0.18, 0.18, "move"),
        ("cup_to_serving_spot", "cup", "robocasa_cafe", 0.20, -0.20, "move"),
        ("bowl_top_pick", "bowl", "robocasa_kitchen", -0.08, None, "pick"),
        ("plate_top_pick", "plate", "robocasa_apartment", 0.08, None, "pick"),
        ("sponge_pick_from_sink_side", "sponge", "robocasa_galley", -0.26, None, "pick"),
        ("book_pick_from_table", "book", "robocasa_apartment", 0.18, None, "pick"),
        ("remote_pick_from_sofa_table", "remote", "robocasa_apartment", -0.20, None, "pick"),
        ("utensil_pick_from_counter", "utensil", "robocasa_kitchen", 0.14, None, "pick"),
        ("small_box_clamp_lift", "small_box", "robocasa_galley", -0.14, None, "pick"),
        ("snack_box_clamp_lift", "snack_box", "robocasa_cafe", 0.10, None, "pick"),
    ]
    out: list[dict[str, Any]] = []
    for idx, (task_id, kind, scene, object_y, target_y, verb) in enumerate(specs):
        out.append(
            _task(
                task_id=task_id,
                description=f"{verb} {kind} in a dense tabletop manipulation task.",
                scene=scene,
                kind=kind,
                object_id=f"sg_{kind}_{idx}",
                object_x=1.52 + 0.05 * (idx % 4),
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["tabletop", "dense", verb],
                walk_duration=4.0 + 0.2 * (idx % 3),
            )
        )
    return out


def _clutter_tabletop_tasks() -> list[dict[str, Any]]:
    specs = [
        (
            "clutter_pick_ball_between_cup_book",
            "ball",
            "robocasa_kitchen",
            -0.22,
            None,
            "pick",
            [("cup", "distractor_cup", 0.08, -0.02), ("book", "distractor_book", -0.10, 0.20)],
        ),
        (
            "clutter_move_can_around_plate",
            "can",
            "robocasa_galley",
            0.16,
            -0.18,
            "move",
            [("plate", "distractor_plate", -0.08, 0.02), ("utensil", "distractor_utensil", 0.10, 0.28)],
        ),
        (
            "clutter_pick_remote_near_mug",
            "remote",
            "robocasa_apartment",
            -0.16,
            None,
            "pick",
            [("mug", "distractor_mug", 0.08, 0.00), ("sponge", "distractor_sponge", -0.10, -0.30)],
        ),
        (
            "clutter_clamp_small_box_near_fruit",
            "small_box",
            "robocasa_cafe",
            0.06,
            None,
            "pick",
            [("fruit", "distractor_fruit", 0.08, -0.18), ("cup", "distractor_cup", -0.10, 0.24)],
        ),
        (
            "clutter_move_apple_between_two_targets",
            "apple",
            "robocasa_kitchen",
            -0.30,
            0.22,
            "move",
            [("orange", "distractor_orange", 0.06, -0.08), ("bowl", "distractor_bowl", -0.08, 0.08)],
        ),
        (
            "clutter_pick_cloth_partly_occluded",
            "cloth",
            "robocasa_galley",
            0.20,
            None,
            "pick",
            [("plate", "distractor_plate", 0.06, 0.04), ("can", "distractor_can", -0.08, -0.18)],
        ),
        (
            "clutter_move_bottle_with_nearby_mug",
            "bottle",
            "robocasa_apartment",
            -0.18,
            0.16,
            "move",
            [("mug", "distractor_mug", 0.07, -0.02), ("book", "distractor_book", -0.10, 0.24)],
        ),
        (
            "clutter_pick_utensil_between_plate_bowl",
            "utensil",
            "robocasa_cafe",
            0.12,
            None,
            "pick",
            [("plate", "distractor_plate", 0.08, -0.04), ("bowl", "distractor_bowl", -0.10, 0.20)],
        ),
    ]
    out: list[dict[str, Any]] = []
    for idx, (task_id, kind, scene, object_y, target_y, verb, distractors) in enumerate(specs):
        out.append(
            _task(
                task_id=task_id,
                description=f"{verb} {kind} from a cluttered tabletop with distractors.",
                scene=scene,
                kind=kind,
                object_id=f"sg_clutter_{kind}_{idx}",
                object_x=1.58 + 0.03 * (idx % 3),
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["tabletop", "clutter", "distractors", verb],
                distractors=[
                    _distractor(kind=d_kind, object_id=f"{task_id}_{d_id}", dx=dx, y=y)
                    for d_kind, d_id, dx, y in distractors
                ],
            )
        )
    return out


def _navigation_manipulation_tasks() -> list[dict[str, Any]]:
    return [
        _task(
            task_id="walk_to_cafe_fruit_pick",
            description="Navigate farther to a cafe table and pick a small fruit.",
            scene="robocasa_cafe",
            kind="fruit",
            object_id="sg_far_fruit",
            object_x=2.10,
            object_y=-0.30,
            target_y=None,
            request_verb="pick",
            tags=["navigation_manipulation", "medium_horizon", "pick"],
            walk_duration=5.8,
            base_x=0.64,
        ),
        _task(
            task_id="walk_to_kitchen_bottle_place",
            description="Approach a kitchen counter, side-grasp a bottle, and place it into a tray.",
            scene="robocasa_kitchen",
            kind="bottle",
            object_id="sg_far_bottle",
            object_x=2.00,
            object_y=-0.18,
            target_y=0.18,
            request_verb="move",
            tags=["navigation_manipulation", "medium_horizon", "pick_place"],
            walk_duration=5.5,
            base_x=0.62,
        ),
        _task(
            task_id="apartment_package_table_pick",
            description="Walk to an apartment table and bimanually clamp a package.",
            scene="robocasa_apartment",
            kind="package",
            object_id="sg_far_package",
            object_x=2.05,
            object_y=-0.04,
            target_y=None,
            request_verb="pick",
            tags=["navigation_manipulation", "medium_horizon", "bimanual"],
            walk_duration=5.6,
            base_x=0.58,
        ),
    ]


def _cross_scene_generalization_tasks() -> list[dict[str, Any]]:
    specs = [
        ("galley_far_cup_place", "cup", "robocasa_galley", 2.25, -0.20, 0.16, "move", 6.0, 0.66),
        ("cafe_far_plate_pick", "plate", "robocasa_cafe", 2.20, 0.18, None, "pick", 5.8, 0.62),
        ("kitchen_far_can_place", "can", "robocasa_kitchen", 2.28, 0.22, -0.14, "move", 6.2, 0.66),
        ("apartment_far_book_pick", "book", "robocasa_apartment", 2.18, -0.24, None, "pick", 5.9, 0.64),
        ("cafe_far_snack_box_clamp", "snack_box", "robocasa_cafe", 2.30, -0.08, None, "pick", 6.1, 0.60),
        ("galley_far_sponge_pick", "sponge", "robocasa_galley", 2.10, 0.10, None, "pick", 5.6, 0.60),
        ("kitchen_far_orange_place", "orange", "robocasa_kitchen", 2.24, -0.26, 0.12, "move", 6.0, 0.64),
        ("apartment_far_remote_pick", "remote", "robocasa_apartment", 2.12, 0.20, None, "pick", 5.5, 0.60),
    ]
    out: list[dict[str, Any]] = []
    for task_id, kind, scene, object_x, object_y, target_y, verb, walk_duration, base_x in specs:
        out.append(
            _task(
                task_id=task_id,
                description=f"{verb} {kind} after a longer cross-scene approach.",
                scene=scene,
                kind=kind,
                object_id=f"sg_{task_id}",
                object_x=object_x,
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["navigation_manipulation", "cross_scene", "medium_horizon", verb],
                walk_duration=walk_duration,
                base_x=base_x,
            )
        )
    return out


def _distractor(*, kind: str, object_id: str, dx: float, y: float) -> dict[str, Any]:
    return {"kind": kind, "object_id": object_id, "dx": float(dx), "y": float(y)}


def _programmatic_expansion_tasks(*, target_count: int, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used = {str(task["id"]) for task in existing}
    out: list[dict[str, Any]] = []
    remaining = max(0, target_count - len(existing))
    if remaining <= 0:
        return out

    family_caps = [
        ("generated_tabletop_grid", 180, _iter_generated_tabletop_grid()),
        ("generated_clutter_grid", 120, _iter_generated_clutter_grid()),
        ("generated_navigation_grid", 90, _iter_generated_navigation_grid()),
        ("generated_sequence_grid", remaining, _iter_generated_sequence_grid()),
    ]
    for _family, cap, iterator in family_caps:
        taken = 0
        for task in iterator:
            task_id = str(task["id"])
            if task_id in used:
                continue
            used.add(task_id)
            out.append(task)
            taken += 1
            if len(out) >= remaining or taken >= cap:
                break
        if len(out) >= remaining:
            break
    return out[:remaining]


def _iter_generated_tabletop_grid():
    y_slots = [-0.32, -0.20, -0.08, 0.08, 0.20, 0.32]
    target_slots = [0.28, -0.28, 0.16, -0.16, 0.06, -0.06]
    for scene_idx, scene in enumerate(SCENES):
        scene_key = SCENE_SHORT[scene]
        for kind_idx, kind in enumerate(OBJECT_SPECS):
            for slot_idx, object_y in enumerate(y_slots):
                bimanual = _is_bimanual_kind(kind)
                verb = "pick" if bimanual or (slot_idx + kind_idx + scene_idx) % 3 == 0 else "move"
                target_y = None if verb == "pick" else target_slots[(slot_idx + kind_idx) % len(target_slots)]
                task_id = f"gen_tabletop_{scene_key}_{kind}_{slot_idx:02d}_{verb}"
                yield _task(
                    task_id=task_id,
                    description=f"Generated tabletop {verb} task for {kind} in {scene_key}.",
                    scene=scene,
                    kind=kind,
                    object_id=f"sg_{task_id}_obj",
                    object_x=1.46 + 0.045 * ((kind_idx + slot_idx) % 6),
                    object_y=object_y,
                    target_y=target_y,
                    request_verb=verb,
                    tags=["generated", "tabletop", "tabletop_grid", verb],
                    walk_duration=3.8 + 0.18 * (slot_idx % 4),
                    base_x=0.50 + 0.018 * (kind_idx % 5),
                )


def _iter_generated_clutter_grid():
    y_slots = [-0.28, -0.14, 0.00, 0.14, 0.28]
    target_slots = [0.24, -0.24, 0.12, -0.12]
    kinds = list(OBJECT_SPECS)
    for scene_idx, scene in enumerate(SCENES):
        scene_key = SCENE_SHORT[scene]
        for kind_idx, kind in enumerate(kinds):
            for slot_idx, object_y in enumerate(y_slots[:4]):
                bimanual = _is_bimanual_kind(kind)
                verb = "pick" if bimanual or (slot_idx + kind_idx) % 2 == 0 else "move"
                target_y = None if verb == "pick" else target_slots[(scene_idx + slot_idx) % len(target_slots)]
                d1 = kinds[(kind_idx + 3 + slot_idx) % len(kinds)]
                d2 = kinds[(kind_idx + 7 + scene_idx) % len(kinds)]
                if d1 == kind:
                    d1 = kinds[(kind_idx + 4) % len(kinds)]
                if d2 in {kind, d1}:
                    d2 = kinds[(kind_idx + 8) % len(kinds)]
                task_id = f"gen_clutter_{scene_key}_{kind}_{slot_idx:02d}_{verb}"
                yield _task(
                    task_id=task_id,
                    description=f"Generated cluttered {verb} task for {kind} with nearby distractors.",
                    scene=scene,
                    kind=kind,
                    object_id=f"sg_{task_id}_obj",
                    object_x=1.52 + 0.035 * ((kind_idx + scene_idx) % 5),
                    object_y=object_y,
                    target_y=target_y,
                    request_verb=verb,
                    tags=["generated", "tabletop", "clutter", "distractors", verb],
                    walk_duration=4.1 + 0.16 * (slot_idx % 3),
                    base_x=0.51 + 0.016 * (kind_idx % 4),
                    distractors=[
                        _distractor(kind=d1, object_id=f"{task_id}_near_left", dx=-0.09, y=_clamp_y(object_y + 0.18)),
                        _distractor(kind=d2, object_id=f"{task_id}_near_right", dx=0.09, y=_clamp_y(object_y - 0.18)),
                    ],
                )


def _iter_generated_navigation_grid():
    y_slots = [-0.34, -0.18, 0.00, 0.18, 0.34]
    target_slots = [0.22, -0.22, 0.10, -0.10]
    for scene_idx, scene in enumerate(SCENES):
        scene_key = SCENE_SHORT[scene]
        for kind_idx, kind in enumerate(OBJECT_SPECS):
            for dist_idx in range(3):
                bimanual = _is_bimanual_kind(kind)
                verb = "pick" if bimanual or (kind_idx + dist_idx) % 2 == 0 else "move"
                object_y = y_slots[(kind_idx + dist_idx + scene_idx) % len(y_slots)]
                target_y = None if verb == "pick" else target_slots[(kind_idx + scene_idx) % len(target_slots)]
                object_x = 1.90 + 0.18 * dist_idx + 0.035 * (kind_idx % 3)
                task_id = f"gen_nav_{scene_key}_{kind}_{dist_idx:02d}_{verb}"
                yield _task(
                    task_id=task_id,
                    description=f"Generated navigation-conditioned {verb} task for {kind}.",
                    scene=scene,
                    kind=kind,
                    object_id=f"sg_{task_id}_obj",
                    object_x=object_x,
                    object_y=object_y,
                    target_y=target_y,
                    request_verb=verb,
                    tags=["generated", "navigation_manipulation", "cross_scene", "medium_horizon", verb],
                    walk_duration=5.2 + 0.45 * dist_idx + 0.05 * (kind_idx % 5),
                    base_x=0.58 + 0.025 * (dist_idx % 3),
                )


def _iter_generated_sequence_grid():
    sequence_patterns = [
        ("counter_sort", "robocasa_kitchen", ["cup", "plate", "utensil", "apple"]),
        ("pantry_tidy", "robocasa_galley", ["can", "snack_box", "sponge", "bottle"]),
        ("living_reset", "robocasa_apartment", ["remote", "book", "small_box", "cloth"]),
        ("cafe_service", "robocasa_cafe", ["cup", "bowl", "fruit", "plate"]),
        ("mixed_table", "robocasa_kitchen", ["ball", "mug", "package", "orange"]),
        ("flat_object_sort", "robocasa_apartment", ["book", "remote", "plate", "utensil"]),
    ]
    seq_idx = 0
    while True:
        base_name, scene, kinds = sequence_patterns[seq_idx % len(sequence_patterns)]
        sequence_id = f"gen_{base_name}_{seq_idx:03d}"
        stage_count = len(kinds)
        for stage_index, kind in enumerate(kinds, start=1):
            bimanual = _is_bimanual_kind(kind)
            verb = "pick" if bimanual or stage_index % 2 == 0 else "move"
            object_y = _clamp_y(-0.30 + 0.18 * ((stage_index + seq_idx) % 4))
            target_y = None if verb == "pick" else _clamp_y(0.26 - 0.16 * ((stage_index + seq_idx) % 4))
            task_id = f"{sequence_id}_stage_{stage_index:02d}_{verb}_{kind}"
            task = _task(
                task_id=task_id,
                description=f"Generated stage {stage_index}/{stage_count} of {sequence_id}.",
                scene=scene,
                kind=kind,
                object_id=f"sg_{task_id}_obj",
                object_x=1.50 + 0.07 * ((stage_index + seq_idx) % 5),
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["generated", "long_sequence", "generated_sequence", sequence_id, verb],
                walk_duration=4.1 + 0.20 * (stage_index % 3),
                base_x=0.52 + 0.015 * (stage_index % 4),
            )
            task.setdefault("metadata", {})
            task["metadata"].update(
                {
                    "sequence_id": sequence_id,
                    "stage_index": stage_index,
                    "stage_count": stage_count,
                    "previous_task_id": (
                        f"{sequence_id}_stage_{stage_index - 1:02d}"
                        if stage_index > 1
                        else None
                    ),
                    "next_task_id": (
                        f"{sequence_id}_stage_{stage_index + 1:02d}"
                        if stage_index < stage_count
                        else None
                    ),
                }
            )
            yield task
        seq_idx += 1


def _coverage_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    affordances: dict[str, int] = {}
    tags: dict[str, int] = {}
    categories: dict[str, int] = {}
    for task in tasks:
        affordance = str((task.get("expect") or {}).get("grasp_affordance") or "unknown")
        affordances[affordance] = affordances.get(affordance, 0) + 1
        for tag in task.get("tags") or []:
            tags[str(tag)] = tags.get(str(tag), 0) + 1
        objects = task.get("objects") if isinstance(task.get("objects"), list) else []
        if objects:
            category = str(objects[0].get("category") or "unknown")
            categories[category] = categories.get(category, 0) + 1
    return {
        "affordances": dict(sorted(affordances.items())),
        "categories": dict(sorted(categories.items())),
        "tags": dict(sorted(tags.items())),
        "generated_count": sum(1 for task in tasks if "generated" in set(task.get("tags") or [])),
    }


def _is_bimanual_kind(kind: str) -> bool:
    return str(OBJECT_SPECS[kind]["grasp_affordance"]) == "bimanual_clamp"


def _clamp_y(value: float) -> float:
    return max(-0.34, min(0.34, float(value)))


def _long_sequence_tasks() -> list[dict[str, Any]]:
    sequences = [
        (
            "set_table_sequence",
            [
                ("set_table_stage_01_pick_mug", "mug", "robocasa_galley", -0.22, None, "pick"),
                ("set_table_stage_02_place_mug", "mug", "robocasa_galley", -0.22, 0.20, "move"),
                ("set_table_stage_03_place_fruit", "fruit", "robocasa_galley", -0.12, 0.24, "move"),
            ],
        ),
        (
            "tidy_table_sequence",
            [
                ("tidy_stage_01_move_ball", "ball", "robocasa_kitchen", -0.28, 0.14, "move"),
                ("tidy_stage_02_pick_cube", "cube", "robocasa_kitchen", -0.10, None, "pick"),
                ("tidy_stage_03_lift_package", "package", "robocasa_kitchen", 0.06, None, "pick"),
            ],
        ),
        (
            "counter_cleanup_sequence",
            [
                ("cleanup_stage_01_move_bottle", "bottle", "robocasa_apartment", -0.24, 0.18, "move"),
                ("cleanup_stage_02_pick_cloth", "cloth", "robocasa_apartment", -0.06, None, "pick"),
                ("cleanup_stage_03_move_mug", "mug", "robocasa_apartment", 0.14, -0.18, "move"),
            ],
        ),
        (
            "breakfast_prep_sequence",
            [
                ("breakfast_stage_01_move_cup", "cup", "robocasa_kitchen", -0.24, 0.16, "move"),
                ("breakfast_stage_02_move_bowl", "bowl", "robocasa_kitchen", -0.08, 0.22, "move"),
                ("breakfast_stage_03_pick_utensil", "utensil", "robocasa_kitchen", 0.12, None, "pick"),
                ("breakfast_stage_04_place_orange", "orange", "robocasa_kitchen", 0.24, -0.20, "move"),
            ],
        ),
        (
            "living_room_reset_sequence",
            [
                ("living_stage_01_pick_remote", "remote", "robocasa_apartment", -0.20, None, "pick"),
                ("living_stage_02_pick_book", "book", "robocasa_apartment", 0.04, None, "pick"),
                ("living_stage_03_lift_snack_box", "snack_box", "robocasa_apartment", 0.22, None, "pick"),
            ],
        ),
        (
            "cafe_service_sequence",
            [
                ("cafe_stage_01_move_cup", "cup", "robocasa_cafe", -0.24, 0.18, "move"),
                ("cafe_stage_02_move_plate", "plate", "robocasa_cafe", -0.04, 0.22, "move"),
                ("cafe_stage_03_move_fruit", "fruit", "robocasa_cafe", 0.18, -0.18, "move"),
                ("cafe_stage_04_pick_cloth", "cloth", "robocasa_cafe", 0.30, None, "pick"),
            ],
        ),
    ]
    out: list[dict[str, Any]] = []
    for sequence_id, stages in sequences:
        count = len(stages)
        for stage_index, (task_id, kind, scene, object_y, target_y, verb) in enumerate(stages, start=1):
            task = _task(
                task_id=task_id,
                description=f"Stage {stage_index}/{count} of {sequence_id}.",
                scene=scene,
                kind=kind,
                object_id=f"{sequence_id}_{kind}_{stage_index}",
                object_x=1.55 + 0.08 * (stage_index - 1),
                object_y=object_y,
                target_y=target_y,
                request_verb=verb,
                tags=["long_sequence", sequence_id, verb],
            )
            task.setdefault("metadata", {})
            task["metadata"].update(
                {
                    "sequence_id": sequence_id,
                    "stage_index": stage_index,
                    "stage_count": count,
                    "previous_task_id": stages[stage_index - 2][0] if stage_index > 1 else None,
                    "next_task_id": stages[stage_index][0] if stage_index < count else None,
                }
            )
            out.append(task)
    return out


def _task(
    *,
    task_id: str,
    description: str,
    scene: str,
    kind: str,
    object_id: str,
    object_x: float,
    object_y: float,
    target_y: float | None,
    request_verb: str,
    tags: list[str],
    walk_duration: float = 4.3,
    base_x: float = 0.54,
    distractors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = OBJECT_SPECS[kind]
    support_id = f"{task_id}_support"
    support_x = object_x
    pickup_x = object_x - TASK_OBJECT_FRONT_OFFSET_X
    requested_object_y = float(object_y)
    object_y = _execution_object_y(kind, requested_object_y)
    object_base_y = object_y
    target_id = f"{task_id}_target" if target_y is not None else None
    obj = {
        "object_id": object_id,
        "category": spec["category"],
        "shape": deepcopy(spec["shape"]),
        "pose_map": {"frame_id": "map", "position": [_round(pickup_x), _round(object_y), spec["height"]]},
        "pose_base": {"frame_id": "base_link", "position": [_round(base_x), _round(object_base_y), spec["base_z"]]},
        "support": support_id,
        "grasp": {
            **_grasp(kind, object_y=object_y, base_x=base_x, walk_duration=walk_duration),
            "preferred_affordance": spec["grasp_affordance"],
        },
    }
    objects = [
        obj,
        {
            "object_id": support_id,
            "category": "counter" if "counter" in description.lower() else "table",
            "shape": {"kind": "box", "size": [1.25, 0.72, 0.74]},
            "pose_map": {"frame_id": "map", "position": [_round(support_x), 0.0, 0.40]},
        },
    ]
    relations = [{"subject": object_id, "relation": "on", "object": support_id, "confidence": 1.0}]
    for idx, distractor in enumerate(distractors or []):
        d_kind = str(distractor.get("kind") or "cube")
        d_spec = OBJECT_SPECS[d_kind]
        d_id = str(distractor.get("object_id") or f"{task_id}_distractor_{idx}")
        d_x = pickup_x + float(distractor.get("dx") or 0.0)
        d_y = float(distractor.get("y") if distractor.get("y") is not None else object_y + 0.18)
        objects.append(
            {
                "object_id": d_id,
                "category": d_spec["category"],
                "shape": deepcopy(d_spec["shape"]),
                "pose_map": {"frame_id": "map", "position": [_round(d_x), _round(d_y), d_spec["height"]]},
                "pose_base": {
                    "frame_id": "base_link",
                    "position": [_round(base_x + float(distractor.get("dx") or 0.0)), _round(d_y), d_spec["base_z"]],
                },
                "support": support_id,
                "grasp": _grasp(d_kind, object_y=d_y, base_x=base_x, walk_duration=walk_duration),
                "properties": {"role": "distractor", "target_of_task": False},
            }
        )
        relations.append({"subject": d_id, "relation": "on", "object": support_id, "confidence": 1.0})
    request: dict[str, Any] = {
        "task": request_verb,
        "object": object_id,
        "metadata": {"preferred_grasp_affordance": spec["grasp_affordance"]},
    }
    if target_id is not None:
        target = {
            "object_id": target_id,
            "category": "place_target",
            "shape": "target",
            "pose_map": {"frame_id": "map", "position": [_round(pickup_x), _round(target_y), spec["height"]]},
            "pose_base": {"frame_id": "base_link", "position": [_round(base_x), _round(target_y), spec["base_z"]]},
            "support": support_id,
        }
        objects.append(target)
        relations.append({"subject": target_id, "relation": "on", "object": support_id, "confidence": 1.0})
        request["target"] = target_id

    return {
        "id": task_id,
        "description": description,
        "scene": scene,
        "tags": sorted(set([*tags, spec["category"], spec["grasp_affordance"]])),
        "objects": objects,
        "relations": relations,
        "request": request,
        "expect": _expect(spec["grasp_affordance"], spec["demo_kind"], target_id is not None),
        "metadata": {
            "benchmark_family": "sonic_general_v0",
            "reset_between_rollouts": True,
            "recommended_runner": "rollout_batch --reset-each-rollout --headless --no-camera",
            "sensor_use": ["lidar_localization", "rgbd_object_anchor", "privileged_physics_state"],
            "distractor_count": len(distractors or []),
            "requested_object_y": _round(requested_object_y),
            "execution_object_y": _round(object_y),
        },
    }


def _execution_object_y(kind: str, object_y: float) -> float:
    spec = OBJECT_SPECS[kind]
    y = float(object_y)
    if spec["demo_kind"] == "ball":
        lo, hi = RIGHT_HAND_OBJECT_Y_BAND
        magnitude = min(abs(lo), max(abs(hi), abs(y)))
        return -magnitude
    lo, hi = BIMANUAL_OBJECT_Y_BAND
    return max(lo, min(hi, y))


def _grasp(kind: str, *, object_y: float, base_x: float, walk_duration: float) -> dict[str, Any]:
    spec = OBJECT_SPECS[kind]
    grasp = {
        "approach_target_x": 0.56 if spec["demo_kind"] == "ball" else 0.50,
        "walk_duration": walk_duration,
        "walk_speed": 0.23,
        "target_y": _round(object_y),
        "reach_x": _round(min(0.58, max(0.46, base_x))),
        "reach_z": spec["base_z"],
        "base_target_map": [0.98, _round(object_y * 0.65), 0.0],
    }
    grasp.update(deepcopy(spec.get("grasp") or {}))
    return grasp


def _expect(grasp_affordance: str, demo_kind: str, has_target: bool) -> dict[str, Any]:
    steps = ["navigate.approach_object", "manip.align_workspace", f"manip.{grasp_affordance}", "manip.lift_object"]
    if has_target:
        steps.extend(["manip.transport_object", "manip.place_object", "manip.release"])
    return {
        "steps": steps,
        "demo_kind": demo_kind,
        "grasp_affordance": grasp_affordance,
        "missing_skills": [],
        "unready_count": 0,
        "contract_error_count": 0,
        "decision_status": "ready_to_execute",
    }


def _materialize_task_scenes(tasks: list[dict[str, Any]], scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        scene_dir / f"scene_sonic_task_{_safe_name(str(task['id']))}.xml"
        for task in tasks
    }
    for stale in scene_dir.glob("scene_sonic_task_*.xml"):
        if stale not in expected:
            stale.unlink()
    for task in tasks:
        source_scene = str(task["scene"])
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
        worldbody.extend(_task_scene_elements(task))
        task_id = _safe_name(str(task["id"]))
        out = scene_dir / f"scene_sonic_task_{task_id}.xml"
        _indent_xml(root)
        tree.write(out, encoding="utf-8", xml_declaration=False)
        rel = out.relative_to(REPO).as_posix()
        task.setdefault("metadata", {})
        task["metadata"]["source_scene"] = source_scene
        task["metadata"]["generated_scene_xml"] = rel
        task["scene"] = rel


def _ensure_task_materials(asset: ET.Element) -> None:
    existing = {child.attrib.get("name") for child in asset if child.tag == "material"}
    materials = {
        "sonic_task_support_mat": "0.48 0.38 0.28 1",
        "sonic_task_support_top_mat": "0.68 0.64 0.56 1",
        "sonic_task_object_mat": "0.16 0.55 0.92 1",
        "sonic_task_box_mat": "0.90 0.72 0.22 1",
        "sonic_task_sphere_mat": "0.20 0.75 0.38 1",
        "sonic_task_target_mat": "0.18 0.90 0.36 0.42",
    }
    for name, rgba in materials.items():
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _task_scene_elements(task: dict[str, Any]) -> list[ET.Element]:
    objects = list(task.get("objects") or [])
    request = task.get("request") or {}
    object_id = str(request.get("object") or request.get("object_id") or "")
    target_id = str(request.get("target") or request.get("target_id") or "")
    pickup = _object_by_id(objects, object_id)
    if pickup is None:
        raise RuntimeError(f"task {task.get('id')} has no pickup object {object_id!r}")
    support = _object_by_id(objects, str(pickup.get("support") or ""))
    target = _object_by_id(objects, target_id) if target_id else None

    root = ET.Element("body", {"name": f"{_safe_name(str(task['id']))}_generated_task"})
    if support is not None:
        root.append(_support_geom(support))
    elif pickup.get("support"):
        root.append(_fallback_support_geom(pickup))
    if target is not None:
        root.append(_target_site(target))
    elements = [root, _dynamic_object_body(pickup)]
    skip_ids = {object_id, target_id, str(pickup.get("support") or "")}
    for obj in objects:
        obj_id = str(obj.get("object_id") or obj.get("id") or "")
        if not obj_id or obj_id in skip_ids:
            continue
        category = str(obj.get("category") or "")
        if category in {"table", "counter", "support_surface", "place_target"}:
            continue
        elements.append(_dynamic_object_body(obj))
    return elements


def _support_geom(support: dict[str, Any]) -> ET.Element:
    pose = _pose_position(support.get("pose_map"), [1.55, 0.0, 0.40])
    sx, sy, sz = _shape_size(_normalize_scene_shape(support.get("shape"), str(support.get("category") or "table")))
    return ET.Element(
        "geom",
        {
            "name": f"{_safe_name(str(support.get('object_id') or 'support'))}_collision",
            "type": "box",
            "size": f"{sx * 0.5:.4f} {sy * 0.5:.4f} {sz * 0.5:.4f}",
            "pos": f"{pose[0]:.4f} {pose[1]:.4f} {pose[2]:.4f}",
            "material": "sonic_task_support_mat",
            "friction": "1.2 0.04 0.002",
        },
    )


def _fallback_support_geom(pickup: dict[str, Any]) -> ET.Element:
    pose = _pose_position(pickup.get("pose_map"), [1.55, 0.0, 0.84])
    height = 0.78
    return ET.Element(
        "geom",
        {
            "name": f"{_safe_name(str(pickup.get('support') or 'support'))}_collision",
            "type": "box",
            "size": "0.7200 0.3800 0.3900",
            "pos": f"{pose[0]:.4f} 0.0000 {height * 0.5:.4f}",
            "material": "sonic_task_support_mat",
            "friction": "1.2 0.04 0.002",
        },
    )


def _target_site(target: dict[str, Any]) -> ET.Element:
    pose = _pose_position(target.get("pose_map"), [1.55, 0.16, 0.84])
    return ET.Element(
        "site",
        {
            "name": f"{_safe_name(str(target.get('object_id') or 'target'))}_site",
            "type": "cylinder",
            "size": "0.1200 0.0060",
            "pos": f"{pose[0]:.4f} {pose[1]:.4f} {pose[2]:.4f}",
            "material": "sonic_task_target_mat",
        },
    )


def _dynamic_object_body(obj: dict[str, Any]) -> ET.Element:
    object_id = _safe_name(str(obj.get("object_id") or obj.get("id") or "task_object"))
    category = str(obj.get("category") or "object")
    shape = _normalize_scene_shape(obj.get("shape"), category)
    pos = _pose_position(obj.get("pose_map"), [1.55, -0.18, 0.84])
    body = ET.Element("body", {"name": object_id, "pos": f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"})
    ET.SubElement(body, "freejoint", {"name": f"{object_id}_freejoint"})
    mass = _mass_for_shape(shape)
    inertia = max(1e-6, mass * 0.001)
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{mass:.5f}",
            "diaginertia": f"{inertia:.8f} {inertia:.8f} {inertia:.8f}",
        },
    )
    material = (
        "sonic_task_box_mat"
        if shape["kind"] in {"box", "flat"}
        else "sonic_task_sphere_mat"
        if shape["kind"] == "sphere"
        else "sonic_task_object_mat"
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{object_id}_geom",
            **_geom_attrs(shape),
            "material": material,
            "condim": "6",
            "friction": "3.5 0.30 0.03",
            "solref": "0.003 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )
    return body


def _normalize_scene_shape(raw: Any, category: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        shape = dict(raw)
    elif raw == "target":
        return {"kind": "target"}
    else:
        shape = {"kind": "box", "size": [0.08, 0.08, 0.06]}
    kind = str(shape.get("kind") or shape.get("type") or "box").lower()
    shape["kind"] = kind
    if kind == "sphere":
        shape["radius"] = float(shape.get("radius") or 0.045)
    elif kind == "cylinder":
        radius = float(shape.get("radius") or 0.045)
        height = _shape_size(shape)[2]
        shape["radius"] = radius
        shape["size"] = [radius * 2.0, radius * 2.0, height]
    elif kind != "target":
        sx, sy, sz = _shape_size(shape)
        if category in {"package", "box", "cube"}:
            sx, sy, sz = max(0.10, sx), max(0.08, sy), max(0.06, sz)
        shape["size"] = [sx, sy, sz]
    return shape


def _geom_attrs(shape: dict[str, Any]) -> dict[str, str]:
    kind = str(shape.get("kind") or "box")
    if kind == "sphere":
        return {"type": "sphere", "size": f"{float(shape.get('radius') or 0.045):.4f}"}
    if kind == "cylinder":
        radius = float(shape.get("radius") or 0.045)
        return {"type": "cylinder", "size": f"{radius:.4f} {_shape_size(shape)[2] * 0.5:.4f}"}
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


def _mass_for_shape(shape: dict[str, Any]) -> float:
    sx, sy, sz = _shape_size(shape)
    volume = max(1e-5, sx * sy * sz)
    return max(0.025, min(0.18, volume * 22.0))


def _object_by_id(objects: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    if not object_id:
        return None
    for obj in objects:
        if str(obj.get("object_id") or obj.get("id")) == object_id:
            return obj
    return None


def _pose_position(raw: Any, default: list[float]) -> tuple[float, float, float]:
    if isinstance(raw, dict):
        pos = raw.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            try:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
            except (TypeError, ValueError):
                pass
    return (float(default[0]), float(default[1]), float(default[2]))


def _safe_name(value: str) -> str:
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else "_")
    safe = "".join(out).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "task"


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _round(value: float) -> float:
    return round(float(value), 4)


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


if __name__ == "__main__":
    raise SystemExit(main())
