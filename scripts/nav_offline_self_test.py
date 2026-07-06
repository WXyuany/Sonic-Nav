#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gear_sonic.nav.control import ControlConfig, SonicControlPayloadBuilder, VelocityLimiter
from gear_sonic.nav.costmap import (
    LocalCostmapConfig,
    build_local_costmap,
    filter_base_points,
    min_front_distance,
    occupied_points_from_grid,
    voxel_downsample_2d,
)
from gear_sonic.nav.metrics import NavigationMetrics
from gear_sonic.nav.params import load_yaml, default_config_path
from scripts.randomize_scene import randomize_scene


def _check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_configs():
    required = [
        "control",
        "costmap",
        "dwa",
        "mppi",
        "eval",
        "eval_scenarios",
        "scene_randomization",
        "scene_default",
    ]
    for name in required:
        cfg = load_yaml(default_config_path(name))
        _check(isinstance(cfg, dict) and cfg, f"{name}.yaml did not load as a non-empty mapping")
    scenarios = load_yaml(default_config_path("eval_scenarios")).get("scenarios", [])
    _check(len(scenarios) >= 3, "expected at least three navigation eval scenarios")


def test_control_payload():
    cfg = ControlConfig(max_v=0.5, max_w=0.7, max_dv=0.1, max_dw=0.2, v_deadband=0.02, w_deadband=0.02)
    limiter = VelocityLimiter(cfg)
    v, w = limiter.limit(2.0, 2.0)
    _check(math.isclose(v, 0.1) and math.isclose(w, 0.2), "velocity limiter did not apply slew limits")
    payload = SonicControlPayloadBuilder(cfg).payload(v, w, upper_body_mode="navigation")
    _check(payload["navigate_cmd"] == [v, 0.0, w], "payload navigate_cmd mismatch")
    _check(len(payload["wrist_pose"]) == 14, "navigation payload should lock wrist pose")
    _check(len(payload["left_hand_joint"]) == 7, "navigation payload should lock left hand")
    free_payload = SonicControlPayloadBuilder(cfg).payload(0.0, 0.0, upper_body_mode="manipulation")
    _check("wrist_pose" not in free_payload, "manipulation mode should not overwrite wrist pose")


def test_costmap():
    cfg = LocalCostmapConfig(
        resolution=0.1,
        forward_range=3.0,
        backward_range=0.5,
        lateral_range=1.5,
        obstacle_radius=0.05,
        inflation_radius=0.30,
        occupied_threshold=35,
    )
    cloud = np.array([
        [1.0, 0.0, 0.2],
        [1.1, 0.02, 0.3],
        [2.0, 0.7, 0.2],
        [0.2, 0.0, 1.5],
        [0.1, 0.0, 0.0],
    ], dtype=np.float32)
    pts = filter_base_points(cloud, robot_radius=0.25, max_range=4.0, min_z=-0.2, max_z=1.0)
    pts = voxel_downsample_2d(pts, 0.2, max_points=10)
    grid = build_local_costmap(pts, cfg)
    occupied = occupied_points_from_grid(grid, cfg)
    _check(grid.shape == (cfg.height, cfg.width), "costmap shape mismatch")
    _check(int(grid.max()) >= cfg.occupied_threshold, "costmap should contain occupied cells")
    _check(len(occupied) > 0, "inflated occupied points should not be empty")
    _check(min_front_distance(occupied) < 1.4, "front distance should see the synthetic obstacle")


def test_metrics():
    metrics = NavigationMetrics(goal_tolerance=0.2, collision_radius=0.1, stuck_speed=0.05, stuck_cmd=0.1, stuck_timeout=0.5)
    metrics.start(0.0, (0.0, 0.0, 0.0), (1.0, 0.0))
    metrics.update(0.3, (0.0, 0.0, 0.0), (0.2, 0.0), clearance=0.5)
    metrics.update(0.7, (0.0, 0.0, 0.0), (0.2, 0.0), clearance=0.08)
    metrics.update(1.0, (1.0, 0.0, 0.0), (0.2, 0.1), clearance=0.4)
    summary = metrics.summary(1.0)
    _check(summary["reached"], "metrics should mark goal reached")
    _check(summary["collision_count"] >= 1, "metrics should count low-clearance samples")
    _check(summary["stuck_events"] >= 1, "metrics should count a synthetic stuck event")
    _check(summary["path_length_m"] >= 0.9, "metrics path length too small")


def test_scene_randomization():
    with tempfile.TemporaryDirectory(prefix="sonic_nav_selftest_") as tmp:
        out = Path(tmp) / "scene_43dof_rand_test.xml"
        randomize_scene(
            "default",
            seed=7,
            count=3,
            output=str(out),
            output_dir=None,
            prefix="selftest_obstacle_",
            dry_run=False,
            switch=False,
        )
        tree = ET.parse(out)
        bodies = [
            body for body in tree.getroot().find("worldbody").findall("body")
            if body.get("name", "").startswith("selftest_obstacle_")
        ]
        _check(len(bodies) == 3, "randomized scene should contain three generated bodies")


def main() -> int:
    tests = [
        test_configs,
        test_control_payload,
        test_costmap,
        test_metrics,
        test_scene_randomization,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All offline navigation self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
