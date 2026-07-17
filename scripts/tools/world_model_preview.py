#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.planners import TaskRequest, WorldModelPipeline, task_request_from_json
from sonic_world.world_model import WorldMemory, anchor_to_world


SAMPLE_BOX_ANCHOR = {
    "scene": "box_demo",
    "box_name": "demo_box",
    "source": "sample",
    "frame_id": "map",
    "box_center_map": [1.60, 0.0, 0.84],
    "box_size": [0.18, 0.12, 0.14],
    "box_point_base": [0.52, 0.0, 0.04],
    "box_point_camera_depth": [0.10, 0.05, 0.60],
    "grasp": {
        "approach_target_x": 0.50,
        "walk_duration": 4.0,
        "walk_speed": 0.22,
        "reach_x": 0.48,
        "open_y": 0.22,
        "clamp_y": 0.10,
        "reach_z": 0.04,
        "lift_z": 0.12,
        "base_target_map": [1.10, 0.0, 0.0],
    },
}


SAMPLE_BALL_ANCHOR = {
    "scene": "ball_demo",
    "ball_name": "demo_ball",
    "source": "sample",
    "frame_id": "map",
    "ball_center_map": [1.62, -0.28, 0.84],
    "ball_radius": 0.045,
    "ball_size": [0.09, 0.09, 0.09],
    "ball_point_base": [0.54, -0.24, 0.03],
    "ball_point_camera_depth": [0.18, 0.05, 0.63],
    "place_center_map": [1.62, -0.08, 0.84],
    "place_point_base": [0.54, -0.04, 0.03],
    "grasp": {
        "approach_target_x": 0.56,
        "walk_duration": 4.5,
        "walk_speed": 0.24,
        "target_y": -0.24,
        "reach_x": 0.54,
        "reach_z": 0.03,
        "base_target_map": [1.06, -0.18, 0.0],
    },
}


SAMPLE_NAV_ANCHOR = {
    "scene": "default",
    "goal_name": "rviz_goal",
    "source": "sample",
    "frame_id": "map",
    "goal_center_map": [2.0, 0.5, 0.0],
    "goal_yaw": 0.0,
    "goal_tolerance": 0.45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the lightweight world model and skill graph generated from a Sonic demo anchor."
    )
    parser.add_argument("--anchor-json", help="Path to an anchor JSON file or a raw JSON string.")
    parser.add_argument("--sample", choices=["box", "ball", "nav"], default="ball")
    parser.add_argument(
        "--verb",
        choices=["pick", "grasp", "pick_place", "move", "place", "approach", "navigate", "go_to", "goto"],
    )
    parser.add_argument("--object-id")
    parser.add_argument("--object-category")
    parser.add_argument("--target-id")
    parser.add_argument(
        "--request-json",
        help="Task request JSON or a path to one, for example '{\"task\":\"move\",\"object\":\"demo_ball\",\"target\":\"place_target\"}'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anchor_json:
        anchor = args.anchor_json
    else:
        anchor = {
            "box": SAMPLE_BOX_ANCHOR,
            "ball": SAMPLE_BALL_ANCHOR,
            "nav": SAMPLE_NAV_ANCHOR,
        }[args.sample]
    pipeline = WorldModelPipeline(memory=WorldMemory(stale_after_s=0.0))
    world = pipeline.memory.update(anchor_to_world(anchor))
    if args.request_json:
        request = task_request_from_json(_read_text_or_raw(args.request_json))
        result = pipeline.plan_current(request, kind="task_request", source="preview_request")
    else:
        verb = args.verb or ("navigate" if args.sample == "nav" else "pick_place")
        request = TaskRequest(
            verb=verb,
            object_id=args.object_id,
            object_category=args.object_category,
            target_id=args.target_id,
        )
        kind = "navigation_goal" if args.sample == "nav" else args.sample
        result = pipeline.plan(world, request, kind=kind, source="preview")
    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


def _read_text_or_raw(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text()
    return value


if __name__ == "__main__":
    main()
