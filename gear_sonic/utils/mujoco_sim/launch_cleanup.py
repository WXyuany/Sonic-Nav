"""Helpers for keeping one Sonic navigation launch active at a time."""

from __future__ import annotations

import os
import signal
import subprocess
import time


STALE_SONIC_PROCESS_PATTERNS: tuple[str, ...] = (
    "gear_sonic/scripts/run_sim_loop.py",
    "scripts/perception/sensor_pub.py",
    "scripts/perception/mid360_pub.py",
    "scripts/perception/camera_pub.py",
    "scripts/perception/box_anchor_pub.py",
    "scripts/perception/ball_anchor_pub.py",
    "scripts/navigation/scene_map_server.py",
    "scripts/navigation/astar_global_planner.py",
    "scripts/navigation/nav_control_adapter.py",
    "scripts/navigation/local_costmap_server.py",
    "scripts/navigation/nav_metrics.py",
    "scripts/navigation/goal_follower.py",
    "scripts/navigation/mppi_nav.py",
    "scripts/navigation/dwa_nav.py",
    "scripts/manipulation/box_grasp_demo.py",
    "scripts/manipulation/ball_pick_place_demo.py",
    "scripts/tools/world_model_node.py",
    "scripts/tools/world_model_executor.py",
    "scripts/tools/world_model_recovery_coordinator.py",
    "scripts/tools/world_model_recovery_backends.py",
    "scripts/tools/world_model_primitive_runner.py",
    "scripts/tools/world_model_episode_anchor.py",
    "scripts/tools/world_model_autonomous_episode.py",
    "scripts/start.py",
    "scripts/start_box_demo.py",
    "scripts/start_ball_demo.py",
    "scripts/start_dwa.py",
    "scripts/start_mppi.py",
    # Legacy paths from the flat scripts/ layout; keep them to clean old launches.
    "scripts/sensor_pub.py",
    "scripts/mid360_pub.py",
    "scripts/camera_pub.py",
    "scripts/box_anchor_pub.py",
    "scripts/ball_anchor_pub.py",
    "scripts/scene_map_server.py",
    "scripts/astar_global_planner.py",
    "scripts/nav_control_adapter.py",
    "scripts/local_costmap_server.py",
    "scripts/nav_metrics.py",
    "scripts/goal_follower.py",
    "scripts/mppi_nav.py",
    "scripts/dwa_nav.py",
    "scripts/box_grasp_demo.py",
    "scripts/ball_pick_place_demo.py",
    "scripts/world_model_node.py",
    "scripts/world_model_executor.py",
    "scripts/world_model_recovery_coordinator.py",
    "scripts/world_model_recovery_backends.py",
    "scripts/world_model_primitive_runner.py",
    "scripts/world_model_episode_anchor.py",
    "scripts/world_model_autonomous_episode.py",
    "g1_ros2_nav/scripts/sensor_bridge.py",
    "g1_ros2_nav/g1_ros2_nav/sensor_bridge.py",
    "g1_ros2_nav/g1_ros2_nav/standalone_bridge.py",
    "g1_ros2_nav/g1_ros2_nav/g1_bridge.py",
    "g1_deploy_onnx_ref",
)


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pids_for_pattern(pattern: str) -> set[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    pids: set[int] = set()
    for raw in out.split():
        try:
            pids.add(int(raw))
        except ValueError:
            continue
    return pids


def cleanup_stale_sonic_processes(enabled: bool | None = None) -> list[int]:
    """Terminate orphaned Sonic launch children before starting a new stack.

    Set SONIC_CLEANUP_STALE=0 to disable this behavior for debugging.
    """

    if enabled is None:
        enabled = os.environ.get("SONIC_CLEANUP_STALE", "1") != "0"
    if not enabled:
        return []

    protected = {os.getpid(), os.getppid()}
    pids: set[int] = set()
    for pattern in STALE_SONIC_PROCESS_PATTERNS:
        pids.update(_pids_for_pattern(pattern))
    pids.difference_update(protected)

    if not pids:
        return []

    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(1.0)

    for pid in sorted(pids):
        if not _is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    return sorted(pids)
