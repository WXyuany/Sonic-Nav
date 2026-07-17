#!/usr/bin/env python3
"""Publish the active carry-state episode stage as a normalized live anchor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene
from sonic_world.planners import task_request_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish qpos-backed object/target anchors for active episode stages.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qpos-path", default="/tmp/sonic_qpos.npy")
    parser.add_argument("--task-request-topic", default="/sonic_world/task_request")
    parser.add_argument("--anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--rate", type=float, default=20.0)
    return parser.parse_args()


def _live_qos() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=8, reliability=QoSReliabilityPolicy.RELIABLE)


def _latched_qos() -> QoSProfile:
    return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class EpisodeAnchorNode(Node):
    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any]):
        super().__init__("sonic_world_episode_anchor")
        self.args, self.manifest = args, manifest
        selection = resolve_scene(args.scene, repo_root=REPO)
        self.model, self.data = mujoco.MjModel.from_xml_path(str(selection.abs_path)), None
        self.data = mujoco.MjData(self.model)
        ordered_stages = [stage for stage in manifest.get("stages", []) if isinstance(stage, dict)]
        self.stages = {str(stage.get("task_id") or ""): stage for stage in ordered_stages}
        # Publish the first stage immediately; later bootstrap requests select
        # subsequent stages. This breaks the task-request/anchor startup cycle.
        self.active: dict[str, Any] | None = ordered_stages[0] if ordered_stages else None
        self.pub = self.create_publisher(String, args.anchor_topic, _latched_qos())
        self.create_subscription(String, args.task_request_topic, self._task_cb, _live_qos())
        self.create_timer(1.0 / max(1.0, float(args.rate)), self._publish)

    def _task_cb(self, msg: String) -> None:
        try:
            request = task_request_from_json(msg.data)
        except Exception:
            return
        task_id = str(request.metadata.get("request_id") or "")
        stage = self.stages.get(task_id)
        if stage is None:
            for candidate in self.stages.values():
                payload = candidate.get("request") if isinstance(candidate.get("request"), dict) else {}
                if str(payload.get("object_id") or "") == str(request.object_id or ""):
                    stage = candidate
                    break
        if stage is not None:
            self.active = stage

    def _publish(self) -> None:
        if self.active is None:
            return
        try:
            qpos = np.load(self.args.qpos_path, allow_pickle=False)
            self.data.qpos[: min(len(qpos), self.model.nq)] = qpos[: self.model.nq]
        except Exception:
            return
        mujoco.mj_forward(self.model, self.data)
        base_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        if base_id < 0:
            return
        base_pos, base_rot = self.data.xpos[base_id], self.data.xmat[base_id].reshape(3, 3)
        objects = []
        for raw in self.active.get("objects", []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            object_id = str(item.get("object_id") or item.get("id") or "")
            body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id))
            site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{object_id}_site"))
            if body_id >= 0:
                point = self.data.xpos[body_id]
            elif site_id >= 0:
                point = self.data.site_xpos[site_id]
            else:
                continue
            item["pose_map"] = {"frame_id": "map", "position": [float(value) for value in point]}
            item["pose_base"] = {"frame_id": "base_link", "position": [float(value) for value in base_rot.T @ (point - base_pos)]}
            objects.append(item)
        payload = {"schema": "sonic_episode_live_anchor_v0", "scene": self.manifest.get("scene"), "frame_id": "map", "source": "mujoco_qpos_episode", "objects": objects, "properties": {"sequence_id": self.manifest.get("sequence_id"), "task_id": self.active.get("task_id")}}
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pub.publish(msg)


def main() -> int:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    rclpy.init()
    node = EpisodeAnchorNode(args, manifest)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
