#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Sequence

os.environ.update(
    {
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_LOCALHOST_ONLY": "1",
        "ROS_DOMAIN_ID": "42",
    }
)

import rclpy
import zmq
import mujoco
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
)
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene  # noqa: E402

try:
    from g1_ros2_nav.tmp_io import load_npy_if_ready  # noqa: E402
except ImportError:  # pragma: no cover - demo fallback for minimal shells
    load_npy_if_ready = None


GRASP_ASSIST_FILE = os.environ.get(
    "SONIC_BOX_GRASP_ASSIST_FILE", "/tmp/sonic_box_grasp_assist.json"
)

LOCO_IDLE = 0
LOCO_SLOW_WALK = 1

NEUTRAL_HAND = [0.0] * 7
LEFT_CLOSED_FULL = [0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75]
RIGHT_CLOSED_FULL = [0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75]

# The deploy path expects upper_body_position in IsaacLab upper-body order:
# waist yaw/roll/pitch, paired left/right shoulders, elbows, and wrists.
UPPER_BODY_ORDER = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_UPPER_BODY = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

JOINT_LIMITS = {
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (0.19, 2.2515),
    "right_shoulder_roll_joint": (-2.2515, -0.19),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "right_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.9722, 1.9722),
    "right_wrist_roll_joint": (-1.9722, 1.9722),
    "left_wrist_pitch_joint": (-1.6144, 1.6144),
    "right_wrist_pitch_joint": (-1.6144, 1.6144),
    "left_wrist_yaw_joint": (-1.6144, 1.6144),
    "right_wrist_yaw_joint": (-1.6144, 1.6144),
}


@dataclass(frozen=True)
class DemoCommand:
    mode: int
    movement: tuple[float, float, float]
    facing: tuple[float, float, float]
    speed: float
    height: float
    upper_body: list[float]
    left_hand: list[float]
    right_hand: list[float]


@dataclass(frozen=True)
class DemoPhase:
    name: str
    duration: float
    start: DemoCommand
    end: DemoCommand


def _log(text: str) -> None:
    print(f"[box_grasp_demo] {text}", flush=True)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def _lerp_list(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    return [_lerp(float(x), float(y), t) for x, y in zip(a, b)]


def _closed_hands(ratio: float) -> tuple[list[float], list[float]]:
    ratio = _clamp(ratio, 0.0, 1.0)
    return [ratio * v for v in LEFT_CLOSED_FULL], [ratio * v for v in RIGHT_CLOSED_FULL]


def _named_upper_body(**overrides: float) -> list[float]:
    pose = dict(DEFAULT_UPPER_BODY)
    pose.update({k: float(v) for k, v in overrides.items()})
    out = []
    for name in UPPER_BODY_ORDER:
        lo, hi = JOINT_LIMITS[name]
        out.append(_clamp(float(pose[name]), lo, hi))
    return out


def _symmetric_upper_body(
    *,
    waist_pitch: float = 0.0,
    shoulder_pitch: float,
    shoulder_roll: float,
    shoulder_yaw: float,
    elbow: float,
    wrist_pitch: float = 0.0,
) -> list[float]:
    roll = abs(float(shoulder_roll))
    yaw = abs(float(shoulder_yaw))
    return _named_upper_body(
        waist_pitch_joint=waist_pitch,
        left_shoulder_pitch_joint=shoulder_pitch,
        right_shoulder_pitch_joint=shoulder_pitch,
        left_shoulder_roll_joint=roll,
        right_shoulder_roll_joint=-roll,
        left_shoulder_yaw_joint=-yaw,
        right_shoulder_yaw_joint=yaw,
        left_elbow_joint=elbow,
        right_elbow_joint=elbow,
        left_wrist_pitch_joint=wrist_pitch,
        right_wrist_pitch_joint=wrist_pitch,
    )


def _command(
    *,
    mode: int,
    upper_body: list[float],
    height: float,
    speed: float = -1.0,
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    facing: tuple[float, float, float] = (1.0, 0.0, 0.0),
    left_hand: list[float] | None = None,
    right_hand: list[float] | None = None,
) -> DemoCommand:
    return DemoCommand(
        mode=int(mode),
        movement=tuple(float(v) for v in movement),
        facing=tuple(float(v) for v in facing),
        speed=float(speed),
        height=float(height),
        upper_body=list(upper_body),
        left_hand=list(NEUTRAL_HAND if left_hand is None else left_hand),
        right_hand=list(NEUTRAL_HAND if right_hand is None else right_hand),
    )


def _interp_command(a: DemoCommand, b: DemoCommand, t: float) -> DemoCommand:
    s = _smoothstep(t)
    return DemoCommand(
        mode=b.mode,
        movement=tuple(_lerp(x, y, s) for x, y in zip(a.movement, b.movement)),
        facing=b.facing,
        speed=_lerp(a.speed, b.speed, s),
        height=_lerp(a.height, b.height, s),
        upper_body=_lerp_list(a.upper_body, b.upper_body, s),
        left_hand=_lerp_list(a.left_hand, b.left_hand, s),
        right_hand=_lerp_list(a.right_hand, b.right_hand, s),
    )


def _velocity(
    current: Sequence[float],
    previous: Sequence[float] | None,
    dt: float,
    max_velocity: float,
) -> list[float]:
    if previous is None or dt <= 1e-4:
        return [0.0] * len(current)
    return [
        _clamp((float(c) - float(p)) / dt, -max_velocity, max_velocity)
        for c, p in zip(current, previous)
    ]


class MujocoUpperBodyIK:
    def __init__(self, scene: str, *, qpos_path: str = "/tmp/sonic_qpos.npy"):
        selection = resolve_scene(scene, repo_root=REPO)
        self.model = mujoco.MjModel.from_xml_path(str(selection.abs_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_path = qpos_path
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_middle_0_link"
        )
        self.right_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_0_link"
        )
        if min(self.base_body_id, self.left_body_id, self.right_body_id) < 0:
            raise RuntimeError("required G1 hand/base bodies were not found for IK")

        self.joint_qpos_ids = []
        self.joint_dof_ids = []
        self.joint_ranges = []
        for name in UPPER_BODY_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"joint '{name}' not found for IK")
            self.joint_qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
            low, high = (float(v) for v in self.model.jnt_range[joint_id])
            if name == "waist_yaw_joint":
                low, high = max(low, -0.15), min(high, 0.15)
            elif name == "waist_roll_joint":
                low, high = max(low, -0.08), min(high, 0.08)
            elif name == "waist_pitch_joint":
                low, high = max(low, -0.10), min(high, 0.14)
            self.joint_ranges.append((low, high))
        self.joint_qpos_ids = np.asarray(self.joint_qpos_ids, dtype=np.int32)
        self.joint_dof_ids = np.asarray(self.joint_dof_ids, dtype=np.int32)
        self.joint_ranges = np.asarray(self.joint_ranges, dtype=np.float64)

    def _sync_live_qpos(self) -> None:
        if load_npy_if_ready is not None:
            try:
                qpos = load_npy_if_ready(self.qpos_path)
            except OSError:
                qpos = None
            if qpos is not None:
                n = min(len(qpos), self.model.nq)
                self.data.qpos[:n] = qpos[:n]
        mujoco.mj_forward(self.model, self.data)

    def _base_to_world(self, point_base: Sequence[float]) -> np.ndarray:
        point_base = np.asarray(point_base, dtype=np.float64)
        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        return base_pos + base_rot @ point_base

    def _solve(
        self,
        left_base: Sequence[float],
        right_base: Sequence[float],
        seed: Sequence[float],
        *,
        max_iters: int,
        damping: float,
        regularization: float,
        step_limit: float,
    ) -> tuple[list[float], float]:
        self._sync_live_qpos()
        q = self.data.qpos.copy()
        seed_arr = np.asarray(seed, dtype=np.float64)
        q[self.joint_qpos_ids] = seed_arr
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        left_world = self._base_to_world(left_base)
        right_world = self._base_to_world(right_base)
        last_err = np.inf

        for _ in range(max(1, int(max_iters))):
            err = np.concatenate(
                [
                    left_world - self.data.xpos[self.left_body_id],
                    right_world - self.data.xpos[self.right_body_id],
                ]
            )
            last_err = float(np.linalg.norm(err.reshape(2, 3), axis=1).max())
            if last_err < 0.012:
                break

            jac_left = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_right = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jac_left, jacr, self.left_body_id)
            mujoco.mj_jacBody(self.model, self.data, jac_right, jacr, self.right_body_id)
            jac = np.vstack([jac_left[:, self.joint_dof_ids], jac_right[:, self.joint_dof_ids]])

            q_sel = self.data.qpos[self.joint_qpos_ids]
            lhs = jac.T @ jac + (float(damping) ** 2 + float(regularization)) * np.eye(len(seed_arr))
            rhs = jac.T @ err - float(regularization) * (q_sel - seed_arr)
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
            if max_abs > step_limit > 0.0:
                dq *= float(step_limit) / max_abs
            q_sel = q_sel + dq
            q_sel = np.minimum(self.joint_ranges[:, 1], np.maximum(self.joint_ranges[:, 0], q_sel))
            self.data.qpos[self.joint_qpos_ids] = q_sel
            mujoco.mj_forward(self.model, self.data)

        return [float(v) for v in self.data.qpos[self.joint_qpos_ids]], last_err

    def _clip_upper_pose(self, pose: Sequence[float]) -> list[float]:
        pose_arr = np.asarray(pose, dtype=np.float64)
        pose_arr = np.minimum(self.joint_ranges[:, 1], np.maximum(self.joint_ranges[:, 0], pose_arr))
        return [float(v) for v in pose_arr]

    def _lift_from_clamp_pose(self, clamp_pose: Sequence[float], args: argparse.Namespace) -> list[float]:
        pose = list(float(v) for v in clamp_pose)
        shoulder_delta = float(args.ik_lift_shoulder_delta)
        elbow_delta = float(args.ik_lift_elbow_delta)
        waist_delta = float(args.ik_lift_waist_delta)
        pose[2] -= waist_delta
        pose[3] -= shoulder_delta
        pose[4] -= shoulder_delta
        pose[9] -= elbow_delta
        pose[10] -= elbow_delta
        return self._clip_upper_pose(pose)

    def _solve_named_pose(
        self,
        name: str,
        *,
        y: float,
        x_delta: float,
        z_delta: float,
        seed: Sequence[float],
        hand_x: float,
        hand_z: float,
        center_y: float,
        args: argparse.Namespace,
    ) -> tuple[list[float], float]:
        left = [hand_x + x_delta, center_y + y, hand_z + z_delta]
        right = [hand_x + x_delta, center_y - y, hand_z + z_delta]
        return self._solve(
            left,
            right,
            seed,
            max_iters=args.ik_iters,
            damping=args.ik_damping,
            regularization=args.ik_regularization,
            step_limit=args.ik_step_limit,
        )

    def solve_grasp_poses(
        self,
        anchor: dict,
        fallback_poses: dict[str, list[float]],
        args: argparse.Namespace,
    ) -> tuple[dict[str, list[float]], dict[str, float]]:
        box_base = np.asarray(anchor.get("box_point_base", [np.nan, np.nan, np.nan]), dtype=np.float64)
        box_size = np.asarray(anchor.get("box_size", [0.24, 0.19, 0.20]), dtype=np.float64)
        if box_base.shape != (3,) or box_size.shape != (3,) or not np.all(np.isfinite(box_base)):
            raise ValueError("bad box anchor for IK")

        grasp = anchor.get("grasp") or {}
        half_y = max(0.05, float(box_size[1]) * 0.5)
        center_y = float(grasp.get("target_y", box_base[1]))
        open_y = max(half_y + float(args.ik_open_margin), float(grasp.get("open_y", half_y + 0.08)))
        clamp_y = max(half_y + float(args.ik_clamp_margin), float(grasp.get("clamp_y", half_y + 0.01)))
        squeeze_y = max(
            half_y + float(args.ik_squeeze_margin),
            clamp_y - float(args.ik_squeeze_inset),
        )
        hand_x = float(box_base[0]) + float(args.ik_x_offset)
        hand_z = float(box_base[2]) + float(args.ik_z_offset)

        poses: dict[str, list[float]] = {}
        errors: dict[str, float] = {}
        previous = fallback_poses["open_ready"]
        for name, y, x_delta, z_delta, seed_name in [
            ("open_ready", open_y, -0.060, float(args.ik_pregrasp_z), "open_ready"),
            ("reach_open", open_y, -0.030, float(args.ik_reach_z), "reach_open"),
            ("clamp_table", clamp_y, 0.000, 0.0, "clamp_table"),
        ]:
            seed = poses.get(seed_name, fallback_poses.get(seed_name, previous))
            solved, err = self._solve_named_pose(
                name,
                y=y,
                x_delta=x_delta,
                z_delta=z_delta,
                seed=seed,
                hand_x=hand_x,
                hand_z=hand_z,
                center_y=center_y,
                args=args,
            )
            poses[name] = solved
            errors[name] = err
            previous = solved
        if "clamp_table" in poses:
            poses["lift_table"] = self._lift_from_clamp_pose(poses["clamp_table"], args)
            errors["lift_table"] = 0.0

            squeeze_seed = poses["lift_table"]
            squeeze, squeeze_err = self._solve_named_pose(
                "squeeze_table",
                y=squeeze_y,
                x_delta=float(args.ik_squeeze_x_delta),
                z_delta=float(args.ik_squeeze_z),
                seed=squeeze_seed,
                hand_x=hand_x,
                hand_z=hand_z,
                center_y=center_y,
                args=args,
            )
            poses["squeeze_table"] = squeeze
            errors["squeeze_table"] = squeeze_err

            carry_seed = squeeze
            carry, carry_err = self._solve_named_pose(
                "carry",
                y=squeeze_y,
                x_delta=float(args.ik_carry_x_delta),
                z_delta=float(args.ik_lift_z) + float(args.ik_carry_z_extra),
                seed=carry_seed,
                hand_x=hand_x,
                hand_z=hand_z,
                center_y=center_y,
                args=args,
            )
            poses["carry"] = carry
            errors["carry"] = carry_err
        return poses, errors


def _upper_body_poses(args: argparse.Namespace) -> dict[str, list[float]]:
    neutral = _named_upper_body()
    open_ready = _symmetric_upper_body(
        shoulder_pitch=args.ready_pitch,
        shoulder_roll=args.open_roll,
        shoulder_yaw=args.ready_yaw,
        elbow=args.ready_elbow,
    )
    reach_open = _symmetric_upper_body(
        waist_pitch=args.waist_pitch,
        shoulder_pitch=args.reach_pitch,
        shoulder_roll=args.open_roll,
        shoulder_yaw=args.reach_yaw,
        elbow=args.reach_elbow,
    )
    clamp_table = _symmetric_upper_body(
        waist_pitch=args.waist_pitch,
        shoulder_pitch=args.reach_pitch,
        shoulder_roll=args.clamp_roll,
        shoulder_yaw=args.clamp_yaw,
        elbow=args.clamp_elbow,
    )
    lift_table = _symmetric_upper_body(
        waist_pitch=args.waist_pitch * 0.35,
        shoulder_pitch=args.clear_pitch,
        shoulder_roll=args.clamp_roll,
        shoulder_yaw=args.clamp_yaw,
        elbow=args.clear_elbow,
    )
    squeeze_table = _symmetric_upper_body(
        waist_pitch=args.waist_pitch * 0.25,
        shoulder_pitch=args.clear_pitch,
        shoulder_roll=max(0.20, args.clamp_roll - 0.03),
        shoulder_yaw=args.clamp_yaw,
        elbow=args.clear_elbow + 0.04,
    )
    carry = _symmetric_upper_body(
        waist_pitch=0.0,
        shoulder_pitch=args.carry_pitch,
        shoulder_roll=args.carry_roll,
        shoulder_yaw=args.carry_yaw,
        elbow=args.carry_elbow,
        wrist_pitch=args.carry_wrist_pitch,
    )
    return {
        "neutral": neutral,
        "open_ready": open_ready,
        "reach_open": reach_open,
        "clamp_table": clamp_table,
        "lift_table": lift_table,
        "squeeze_table": squeeze_table,
        "carry": carry,
    } | getattr(args, "ik_pose_overrides", {})


def make_demo_phases(args: argparse.Namespace) -> list[DemoPhase]:
    poses = _upper_body_poses(args)
    left_grasp, right_grasp = _closed_hands(args.close_ratio)
    left_squeeze, right_squeeze = _closed_hands(args.squeeze_close_ratio)
    stand = _command(mode=LOCO_IDLE, upper_body=poses["neutral"], height=args.stand_height)
    walk = _command(
        mode=LOCO_SLOW_WALK,
        movement=(1.0, 0.0, 0.0),
        speed=args.walk_speed,
        upper_body=poses["neutral"],
        height=args.stand_height,
    )
    ready = _command(mode=LOCO_IDLE, upper_body=poses["open_ready"], height=args.stand_height)
    reach = _command(mode=LOCO_IDLE, upper_body=poses["reach_open"], height=args.stand_height)
    clamp = _command(
        mode=LOCO_IDLE,
        upper_body=poses["clamp_table"],
        height=args.stand_height,
        left_hand=left_grasp,
        right_hand=right_grasp,
    )
    lift = _command(
        mode=LOCO_IDLE,
        upper_body=poses["lift_table"],
        height=args.stand_height,
        left_hand=left_grasp,
        right_hand=right_grasp,
    )
    squeeze = _command(
        mode=LOCO_IDLE,
        upper_body=poses["squeeze_table"],
        height=args.stand_height,
        left_hand=left_squeeze,
        right_hand=right_squeeze,
    )
    carry = _command(
        mode=LOCO_IDLE,
        upper_body=poses["carry"],
        height=args.stand_height,
        left_hand=left_squeeze,
        right_hand=right_squeeze,
    )
    carry_walk = _command(
        mode=LOCO_SLOW_WALK,
        movement=(1.0, 0.0, 0.0),
        speed=args.carry_walk_speed,
        upper_body=poses["carry"],
        height=args.stand_height,
        left_hand=left_squeeze,
        right_hand=right_squeeze,
    )
    return [
        DemoPhase("stand_ready", 1.0, stand, stand),
        DemoPhase("walk_two_steps", args.walk_duration, walk, walk),
        DemoPhase("settle_before_grasp", 0.8, stand, stand),
        DemoPhase("arms_open_table", args.prepare_duration, stand, ready),
        DemoPhase("reach_table_open", args.reach_duration, ready, reach),
        DemoPhase("forearm_clamp_box", args.clamp_duration, reach, clamp),
        DemoPhase("lift_box_from_table", args.clear_duration, clamp, lift),
        DemoPhase("squeeze_box_secure", args.squeeze_duration, lift, squeeze),
        DemoPhase("bring_box_to_chest", args.lift_duration, squeeze, carry),
        DemoPhase("carry_settle", args.carry_lock_duration, carry, carry),
        DemoPhase("carry_walk_forward", args.carry_walk_duration, carry_walk, carry_walk),
        DemoPhase("hold_box_clamped", args.hold_duration, carry, carry),
    ]


def apply_box_anchor(args: argparse.Namespace, anchor: dict, *, update_walk: bool = True) -> str:
    grasp = anchor.get("grasp") or {}
    if update_walk:
        if "walk_speed" in grasp:
            setattr(args, "walk_speed", float(grasp["walk_speed"]))
        if "walk_duration" in grasp:
            walk_duration = float(grasp["walk_duration"]) + float(args.walk_extra_duration)
            setattr(args, "walk_duration", min(float(args.max_approach_duration), walk_duration))

    for arg_name in [
        "assist_x",
        "assist_z",
        "clamp_assist_x",
        "clamp_assist_z",
        "clear_assist_x",
        "clear_assist_z",
    ]:
        if arg_name in grasp:
            setattr(args, arg_name, float(grasp[arg_name]))

    if "target_y" in grasp:
        args.assist_y = float(grasp["target_y"])
    if "approach_target_x" in grasp:
        args.approach_target_x = float(grasp["approach_target_x"])

    open_y = float(grasp.get("open_y", 0.24))
    clamp_y = float(grasp.get("clamp_y", 0.14))
    args.open_roll = _clamp(0.24 + 1.05 * open_y, 0.38, 0.66)
    args.clamp_roll = _clamp(0.17 + 0.78 * clamp_y, 0.22, min(args.open_roll - 0.10, 0.38))
    args.carry_roll = _clamp(args.clamp_roll - 0.01, 0.21, 0.36)

    reach_x = float(grasp.get("reach_x", 0.38))
    reach_z = float(grasp.get("reach_z", -0.02))
    pitch_mag = 0.58 + 0.55 * max(0.0, reach_x - 0.34) + 0.15 * max(0.0, -0.04 - reach_z)
    args.reach_pitch = -_clamp(pitch_mag, 0.58, 0.86)
    args.reach_elbow = _clamp(0.58 - 0.55 * max(0.0, reach_x - 0.38), 0.36, 0.62)
    args.clamp_elbow = _clamp(args.reach_elbow + 0.07, 0.45, 0.68)
    args.clear_pitch = _clamp(args.reach_pitch + 0.06, -0.82, -0.52)
    args.clear_elbow = _clamp(args.clamp_elbow + 0.10, 0.62, 0.84)
    args.carry_pitch = _clamp(args.clear_pitch + 0.06, -0.80, -0.54)

    box_base = anchor.get("box_point_base", [math.nan, math.nan, math.nan])
    box_camera = anchor.get("box_point_camera_depth", [math.nan, math.nan, math.nan])
    stage = "initial" if update_walk else "post_walk"
    return (
        f"{stage} "
        f"box_base=({float(box_base[0]):.2f},{float(box_base[1]):.2f},{float(box_base[2]):.2f}) "
        f"camera_depth=({float(box_camera[0]):.2f},{float(box_camera[1]):.2f},{float(box_camera[2]):.2f}) "
        f"walk={args.walk_duration:.2f}s target_x={args.approach_target_x:.2f} "
        f"open_roll={args.open_roll:.2f} "
        f"clamp_roll={args.clamp_roll:.2f} reach_pitch={args.reach_pitch:.2f} "
        f"clamp=({float(args.clamp_assist_x or 0.0):.2f},{args.assist_y:.2f},{float(args.clamp_assist_z or 0.0):.2f}) "
        f"assist=({args.assist_x:.2f},{args.assist_y:.2f},{args.assist_z:.2f})"
    )


class BoxGraspDemo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("box_grasp_demo")
        self.args = args
        self.args.ik_pose_overrides = {}
        self.ik_solver = None
        if args.ik_upper_body:
            try:
                self.ik_solver = MujocoUpperBodyIK(args.scene, qpos_path=args.qpos_path)
                _log(f"IK upper-body solver ready: scene={args.scene}")
            except Exception as exc:
                if args.require_ik:
                    raise
                self.get_logger().warn(f"upper-body IK disabled: {exc}")
        self.phase_pub = self.create_publisher(String, "/sonic_demo/phase", 10)
        self.latest_anchor: dict | None = None
        self._anchor_wall_time = 0.0
        if args.use_box_anchor:
            qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(String, args.anchor_topic, self._anchor_cb, qos)

        self.zmq_context = zmq.Context.instance()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.endpoint = f"tcp://{args.zmq_bind_host}:{args.zmq_port}"
        self.socket.bind(self.endpoint)
        self.phases = make_demo_phases(args)
        self._last_upper_body: list[float] | None = None
        self._last_send_time: float | None = None
        self._post_walk_anchor_applied = False
        self._last_anchor_retrack = 0.0
        self._box_pre_lift_z: float | None = None
        _log(f"ZMQ publisher bound: {self.endpoint}")
        if self.args.box_attach:
            _log("box attach assist enabled explicitly")
        else:
            _log("box attach assist disabled; using contact/friction grasp only")

    def run(self) -> None:
        time.sleep(float(self.args.zmq_connect_wait))
        self._write_box_attach(False, "warmup")
        time.sleep(float(self.args.warmup))
        initial_anchor_after = time.monotonic()
        self._send_start()

        if self.args.use_box_anchor:
            anchor_ok = self._wait_and_apply_anchor(fresh_after=initial_anchor_after)
            if self.args.require_box_anchor and not anchor_ok:
                raise RuntimeError(
                    f"box anchor is required but no anchor was received on {self.args.anchor_topic}"
                )

        period = 1.0 / max(5.0, float(self.args.rate))
        phase_index = 0
        approach_retries = 0
        while phase_index < len(self.phases):
            phase = self.phases[phase_index]
            if (
                phase.name == "arms_open_table"
                and self.args.use_box_anchor
                and not self._post_walk_anchor_applied
            ):
                self._wait_and_apply_anchor(
                    update_walk=False,
                    timeout=1.2,
                    fresh_after=time.monotonic(),
                )
                self._post_walk_anchor_applied = True
                phase = self.phases[phase_index]
            elif (
                phase.name in {"reach_table_open", "forearm_clamp_box"}
                and self.args.use_box_anchor
                and self.args.retrack_before_grasp
            ):
                self._retrack_anchor_for_grasp()
                phase = self.phases[phase_index]

            _log(f"phase: {phase.name} ({phase.duration:.1f}s)")
            phase_msg = String()
            phase_msg.data = phase.name
            if phase.name == "lift_box_from_table":
                self._mark_lift_reference()
            t0 = time.monotonic()
            while rclpy.ok():
                elapsed = time.monotonic() - t0
                if elapsed >= phase.duration:
                    break
                ratio = elapsed / max(1e-3, phase.duration)
                cmd = _interp_command(phase.start, phase.end, ratio)
                self._publish_planner(cmd)
                self._update_box_attach(phase.name, ratio)
                self.phase_pub.publish(phase_msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                if self._should_finish_walk(phase.name, elapsed):
                    break
                time.sleep(period)
            self._publish_planner(phase.end)
            self._update_box_attach(phase.name, 1.0)
            if phase.name == "lift_box_from_table" and self.args.use_box_anchor:
                self._wait_for_lifted_box(timeout=float(self.args.lift_detect_timeout))
            if phase.name == "walk_two_steps" and self.args.use_box_anchor:
                forward = self._current_box_forward()
                target = float(self.args.approach_target_x) + float(self.args.approach_tolerance)
                if forward is not None and forward > target:
                    if approach_retries < int(self.args.max_approach_retries):
                        approach_retries += 1
                        _log(
                            f"approach still far: box_x={forward:.2f}m target={target:.2f}m; "
                            f"retry {approach_retries}/{self.args.max_approach_retries}"
                        )
                        retry_cmd = phase.end
                        self.phases[phase_index] = DemoPhase(
                            "walk_two_steps",
                            float(self.args.approach_retry_duration),
                            retry_cmd,
                            retry_cmd,
                        )
                        continue
                    soft_target = target + float(self.args.approach_soft_tolerance)
                    if forward <= soft_target:
                        _log(
                            f"approach close enough: box_x={forward:.2f}m target={target:.2f}m "
                            f"soft_limit={soft_target:.2f}m; continuing"
                        )
                    elif self.args.require_box_anchor:
                        raise RuntimeError(
                            f"approach failed: box_x={forward:.2f}m remains beyond soft limit {soft_target:.2f}m"
                        )
            phase_index += 1

        done_msg = String()
        done_msg.data = "done"
        self.phase_pub.publish(done_msg)
        if self.args.hold:
            _log("demo done; holding final box pose until Ctrl+C")
            final_cmd = self.phases[-1].end
            while rclpy.ok():
                self._publish_planner(final_cmd)
                self._update_box_attach("done", 1.0)
                self.phase_pub.publish(done_msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
        else:
            self._write_box_attach(False, "done")
            _log("demo done")

    def close(self) -> None:
        self._write_box_attach(False, "shutdown")
        self.socket.close(0)

    def _anchor_cb(self, msg: String) -> None:
        try:
            self.latest_anchor = json.loads(msg.data)
            self._anchor_wall_time = time.monotonic()
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"bad box anchor JSON: {exc}")

    def _current_box_forward(self) -> float | None:
        box_base = self._current_box_base()
        if box_base is None:
            return None
        return float(box_base[0])

    def _current_box_base(self) -> list[float] | None:
        if self.latest_anchor is None:
            return None
        if time.monotonic() - self._anchor_wall_time > float(self.args.anchor_fresh_age):
            return None
        box_base = self.latest_anchor.get("box_point_base")
        if not isinstance(box_base, list) or len(box_base) < 3:
            return None
        try:
            point = [float(box_base[0]), float(box_base[1]), float(box_base[2])]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in point):
            return None
        return point

    def _should_finish_walk(self, phase_name: str, elapsed: float) -> bool:
        if phase_name != "walk_two_steps" or not self.args.use_box_anchor:
            return False
        if elapsed < float(self.args.min_approach_duration):
            return False
        forward = self._current_box_forward()
        if forward is None:
            return False
        target = float(self.args.approach_target_x) + float(self.args.approach_tolerance)
        if forward <= target:
            _log(
                f"approach reached: box_x={forward:.2f}m "
                f"target={self.args.approach_target_x:.2f}m elapsed={elapsed:.1f}s"
            )
            return True
        return False

    def _wait_and_apply_anchor(
        self,
        *,
        update_walk: bool = True,
        timeout: float | None = None,
        fresh_after: float | None = None,
    ) -> bool:
        timeout = max(0.0, float(self.args.anchor_timeout if timeout is None else timeout))
        deadline = time.monotonic() + timeout
        reported_implausible = False
        while rclpy.ok() and time.monotonic() < deadline:
            has_anchor = self.latest_anchor is not None
            is_fresh = fresh_after is None or self._anchor_wall_time > fresh_after
            is_plausible = (
                has_anchor
                and is_fresh
                and self._anchor_is_plausible(self.latest_anchor, update_walk=update_walk)
            )
            if is_plausible:
                break
            if has_anchor and is_fresh and not reported_implausible:
                box_base = self.latest_anchor.get("box_point_base", [math.nan, math.nan, math.nan])
                try:
                    x, y, z = (float(box_base[i]) for i in range(3))
                except (TypeError, ValueError, IndexError):
                    x, y, z = math.nan, math.nan, math.nan
                _log(
                    "waiting for plausible box anchor; "
                    f"latest box_base=({x:.2f},{y:.2f},{z:.2f})"
                )
                reported_implausible = True
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.latest_anchor is None or (
            fresh_after is not None and self._anchor_wall_time <= fresh_after
        ) or not self._anchor_is_plausible(self.latest_anchor, update_walk=update_walk):
            if reported_implausible:
                _log(f"no plausible box anchor received on {self.args.anchor_topic}; using fixed defaults")
            else:
                _log(f"no box anchor received on {self.args.anchor_topic}; using fixed defaults")
            return False
        try:
            summary = apply_box_anchor(self.args, self.latest_anchor, update_walk=update_walk)
            if not update_walk:
                self._update_ik_poses_from_anchor(self.latest_anchor)
            self.phases = make_demo_phases(self.args)
            _log(f"using box anchor: {summary}")
            return True
        except Exception as exc:
            self.get_logger().warn(f"failed to apply box anchor; using fixed defaults: {exc}")
            return False

    def _anchor_is_plausible(self, anchor: dict | None, *, update_walk: bool) -> bool:
        if anchor is None or not update_walk:
            return True
        box_base = anchor.get("box_point_base")
        if not isinstance(box_base, list) or len(box_base) < 3:
            return False
        try:
            x, y, z = (float(box_base[i]) for i in range(3))
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (x, y, z)):
            return False
        return (
            float(self.args.initial_anchor_min_x) <= x <= float(self.args.initial_anchor_max_x)
            and abs(y) <= float(self.args.initial_anchor_max_abs_y)
            and float(self.args.initial_anchor_min_z) <= z <= float(self.args.initial_anchor_max_z)
        )

    def _retrack_anchor_for_grasp(self) -> None:
        now = time.monotonic()
        if now - self._last_anchor_retrack < float(self.args.retrack_min_interval):
            return
        self._last_anchor_retrack = now
        self._wait_and_apply_anchor(
            update_walk=False,
            timeout=float(self.args.retrack_timeout),
            fresh_after=now - float(self.args.anchor_fresh_age),
        )

    def _mark_lift_reference(self) -> None:
        box_base = self._current_box_base()
        if box_base is None:
            self._box_pre_lift_z = None
            return
        self._box_pre_lift_z = float(box_base[2])
        _log(f"lift reference: box_z={self._box_pre_lift_z:.3f}m")

    def _wait_for_lifted_box(self, *, timeout: float) -> bool:
        if self._box_pre_lift_z is None:
            self._mark_lift_reference()
        if self._box_pre_lift_z is None:
            _log("box lift check skipped: no fresh anchor")
            return False

        threshold = float(self.args.lift_detect_z)
        deadline = time.monotonic() + max(0.0, timeout)
        best_delta = -math.inf
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.03)
            box_base = self._current_box_base()
            if box_base is None:
                continue
            delta = float(box_base[2]) - float(self._box_pre_lift_z)
            best_delta = max(best_delta, delta)
            if delta >= threshold:
                _log(f"box lifted: dz={delta:.3f}m; tightening hands")
                return True
            time.sleep(0.03)
        if math.isfinite(best_delta):
            _log(f"box not lifted yet: best_dz={best_delta:.3f}m; applying gentle squeeze anyway")
        else:
            _log("box lift check had no fresh samples; applying gentle squeeze anyway")
        return False

    def _update_ik_poses_from_anchor(self, anchor: dict) -> None:
        if self.ik_solver is None:
            return
        previous_overrides = getattr(self.args, "ik_pose_overrides", {})
        self.args.ik_pose_overrides = {}
        fallback_poses = _upper_body_poses(self.args)
        self.args.ik_pose_overrides = previous_overrides
        poses, errors = self.ik_solver.solve_grasp_poses(anchor, fallback_poses, self.args)
        critical_errors = {name: err for name, err in errors.items() if name in {"clamp_table"}}
        worst_error = max(critical_errors.values()) if critical_errors else math.inf
        if worst_error > float(self.args.ik_max_error):
            message = " ".join(f"{name}={err:.3f}" for name, err in errors.items())
            if self.args.require_ik:
                raise RuntimeError(f"IK error too large: {message}")
            self.get_logger().warn(f"IK result rejected: {message}")
            return
        self.args.ik_pose_overrides = poses
        message = " ".join(f"{name}={err:.3f}" for name, err in errors.items())
        _log(f"IK upper-body poses applied: {message}")

    def _send_start(self) -> None:
        start_msg = build_command_message(start=True, stop=False, planner=True)
        idle = self.phases[0].start
        for _ in range(max(1, int(self.args.start_bursts))):
            self.socket.send(start_msg)
            self._publish_planner(idle)
            time.sleep(0.12)
        _log("sent start command: planner mode with upper-body WBC targets")

    def _publish_planner(self, cmd: DemoCommand) -> None:
        now = time.monotonic()
        dt = 0.0 if self._last_send_time is None else now - self._last_send_time
        upper_vel = _velocity(
            cmd.upper_body,
            self._last_upper_body,
            dt,
            float(self.args.max_upper_body_velocity),
        )
        msg = build_planner_message(
            cmd.mode,
            cmd.movement,
            cmd.facing,
            speed=cmd.speed,
            height=cmd.height,
            upper_body_position=cmd.upper_body,
            upper_body_velocity=upper_vel,
            left_hand_position=cmd.left_hand,
            right_hand_position=cmd.right_hand,
        )
        self.socket.send(msg)
        self._last_upper_body = list(cmd.upper_body)
        self._last_send_time = now

    def _box_offset_for_phase(self, phase_name: str, ratio: float) -> tuple[float, float, float]:
        clamp_x = self.args.clamp_assist_x
        if clamp_x is None:
            clamp_x = self.args.assist_x + 0.10
        clamp_z = self.args.clamp_assist_z
        if clamp_z is None:
            clamp_z = self.args.assist_z - 0.04

        clear_x = self.args.clear_assist_x
        if clear_x is None:
            clear_x = clamp_x
        clear_z = self.args.clear_assist_z
        if clear_z is None:
            clear_z = self.args.assist_z + 0.05

        if phase_name == "forearm_clamp_box":
            return float(clamp_x), float(self.args.assist_y), float(clamp_z)
        if phase_name == "lift_box_from_table":
            s = _smoothstep(ratio)
            return (
                _lerp(float(clamp_x), float(clear_x), s),
                float(self.args.assist_y),
                _lerp(float(clamp_z), float(clear_z), s),
            )
        if phase_name == "bring_box_to_chest":
            s = _smoothstep(ratio)
            return (
                _lerp(float(clear_x), float(self.args.assist_x), s),
                float(self.args.assist_y),
                _lerp(float(clear_z), float(self.args.assist_z), s),
            )
        return float(self.args.assist_x), float(self.args.assist_y), float(self.args.assist_z)

    def _update_box_attach(self, phase_name: str, ratio: float) -> None:
        active = phase_name in {
            "lift_box_from_table",
            "squeeze_box_secure",
            "bring_box_to_chest",
            "carry_settle",
            "carry_walk_forward",
            "hold_box_clamped",
            "done",
        }
        if phase_name == "forearm_clamp_box":
            active = ratio >= float(self.args.attach_engage_ratio)
        if not active:
            self._write_box_attach(False, phase_name)
            return
        self._write_box_attach(True, phase_name, self._box_offset_for_phase(phase_name, ratio))

    def _write_box_attach(
        self,
        enabled: bool,
        phase_name: str,
        offset: tuple[float, float, float] | None = None,
    ) -> None:
        if offset is None:
            offset = (float(self.args.assist_x), float(self.args.assist_y), float(self.args.assist_z))
        payload = {
            "enabled": bool(enabled and self.args.box_attach),
            "phase": phase_name,
            "stamp": time.time(),
            "box_enabled": bool(enabled and self.args.box_attach),
            "local_offset": [float(v) for v in offset],
            "blend": float(self.args.box_attach_blend),
        }
        tmp_path = f"{GRASP_ASSIST_FILE}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, GRASP_ASSIST_FILE)
        except OSError as exc:
            self.get_logger().warn(f"failed to write box attach file {GRASP_ASSIST_FILE}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 box demo over zmq_manager: lower-body SONIC, upper-body joint targets."
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--zmq-bind-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-connect-wait", type=float, default=1.0)
    parser.add_argument("--start-bursts", type=int, default=8)
    parser.add_argument("--max-upper-body-velocity", type=float, default=1.8)
    parser.add_argument("--scene", default="box_demo")
    parser.add_argument("--qpos-path", default="/tmp/sonic_qpos.npy")

    parser.add_argument("--stand-height", type=float, default=0.78)
    parser.add_argument("--squat-height", type=float, default=0.78)
    parser.add_argument("--walk-speed", type=float, default=0.26)
    parser.add_argument("--walk-duration", type=float, default=2.0)
    parser.add_argument("--walk-extra-duration", type=float, default=3.0)
    parser.add_argument("--max-approach-duration", type=float, default=10.0)
    parser.add_argument("--min-approach-duration", type=float, default=1.0)
    parser.add_argument("--approach-target-x", type=float, default=0.45)
    parser.add_argument("--approach-tolerance", type=float, default=0.02)
    parser.add_argument("--approach-soft-tolerance", type=float, default=0.08)
    parser.add_argument("--max-approach-retries", type=int, default=5)
    parser.add_argument("--approach-retry-duration", type=float, default=1.0)
    parser.add_argument("--carry-walk-speed", type=float, default=0.09)
    parser.add_argument("--carry-walk-duration", type=float, default=2.0)

    parser.add_argument("--prepare-duration", type=float, default=1.2)
    parser.add_argument("--squat-duration", type=float, default=0.0)
    parser.add_argument("--reach-duration", type=float, default=1.6)
    parser.add_argument("--clamp-duration", type=float, default=1.3)
    parser.add_argument("--clear-duration", type=float, default=1.1)
    parser.add_argument("--squeeze-duration", type=float, default=0.8)
    parser.add_argument("--lift-duration", type=float, default=2.0)
    parser.add_argument("--carry-lock-duration", type=float, default=1.4)
    parser.add_argument("--hold-duration", type=float, default=3.0)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--no-hold", dest="hold", action="store_false")

    parser.add_argument("--ready-pitch", type=float, default=-0.18)
    parser.add_argument("--squat-pitch", type=float, default=-0.18)
    parser.add_argument("--reach-pitch", type=float, default=-0.66)
    parser.add_argument("--clear-pitch", type=float, default=-0.60)
    parser.add_argument("--carry-pitch", type=float, default=-0.56)
    parser.add_argument("--open-roll", type=float, default=0.42)
    parser.add_argument("--clamp-roll", type=float, default=0.25)
    parser.add_argument("--carry-roll", type=float, default=0.24)
    parser.add_argument("--ready-yaw", type=float, default=0.04)
    parser.add_argument("--reach-yaw", type=float, default=0.05)
    parser.add_argument("--clamp-yaw", type=float, default=0.01)
    parser.add_argument("--carry-yaw", type=float, default=0.0)
    parser.add_argument("--ready-elbow", type=float, default=0.68)
    parser.add_argument("--reach-elbow", type=float, default=0.46)
    parser.add_argument("--clamp-elbow", type=float, default=0.54)
    parser.add_argument("--clear-elbow", type=float, default=0.66)
    parser.add_argument("--carry-elbow", type=float, default=0.78)
    parser.add_argument("--carry-wrist-pitch", type=float, default=0.0)
    parser.add_argument("--waist-pitch", type=float, default=0.04)
    parser.add_argument("--close-ratio", type=float, default=0.08)
    parser.add_argument("--squeeze-close-ratio", type=float, default=0.18)

    parser.add_argument("--box-attach", action="store_true", default=False)
    parser.add_argument("--no-box-attach", dest="box_attach", action="store_false")
    parser.add_argument("--box-attach-blend", type=float, default=1.0)
    parser.add_argument("--attach-engage-ratio", type=float, default=0.85)
    parser.add_argument("--assist-x", type=float, default=0.30)
    parser.add_argument("--assist-y", type=float, default=0.0)
    parser.add_argument("--assist-z", type=float, default=-0.02)
    parser.add_argument("--clamp-assist-x", type=float, default=None)
    parser.add_argument("--clamp-assist-z", type=float, default=None)
    parser.add_argument("--clear-assist-x", type=float, default=None)
    parser.add_argument("--clear-assist-z", type=float, default=None)

    parser.add_argument("--use-box-anchor", action="store_true")
    parser.add_argument("--require-box-anchor", action="store_true")
    parser.add_argument("--anchor-topic", default="/sonic_demo/box_anchor")
    parser.add_argument("--anchor-timeout", type=float, default=4.0)
    parser.add_argument("--anchor-fresh-age", type=float, default=0.8)
    parser.add_argument("--initial-anchor-min-x", type=float, default=0.25)
    parser.add_argument("--initial-anchor-max-x", type=float, default=2.50)
    parser.add_argument("--initial-anchor-max-abs-y", type=float, default=0.65)
    parser.add_argument("--initial-anchor-min-z", type=float, default=-0.35)
    parser.add_argument("--initial-anchor-max-z", type=float, default=0.30)
    parser.add_argument("--retrack-before-grasp", action="store_true", default=True)
    parser.add_argument("--no-retrack-before-grasp", dest="retrack_before_grasp", action="store_false")
    parser.add_argument("--retrack-timeout", type=float, default=0.35)
    parser.add_argument("--retrack-min-interval", type=float, default=0.45)
    parser.add_argument("--lift-detect-z", type=float, default=0.035)
    parser.add_argument("--lift-detect-timeout", type=float, default=0.8)

    parser.add_argument("--ik-upper-body", action="store_true", default=True)
    parser.add_argument("--no-ik-upper-body", dest="ik_upper_body", action="store_false")
    parser.add_argument("--require-ik", action="store_true")
    parser.add_argument("--ik-open-margin", type=float, default=0.115)
    parser.add_argument("--ik-clamp-margin", type=float, default=0.055)
    parser.add_argument("--ik-squeeze-margin", type=float, default=0.025)
    parser.add_argument("--ik-squeeze-inset", type=float, default=0.030)
    parser.add_argument("--ik-x-offset", type=float, default=-0.160)
    parser.add_argument("--ik-z-offset", type=float, default=0.055)
    parser.add_argument("--ik-pregrasp-z", type=float, default=0.210)
    parser.add_argument("--ik-reach-z", type=float, default=0.095)
    parser.add_argument("--ik-lift-z", type=float, default=0.115)
    parser.add_argument("--ik-squeeze-z", type=float, default=0.090)
    parser.add_argument("--ik-squeeze-x-delta", type=float, default=-0.012)
    parser.add_argument("--ik-carry-x-delta", type=float, default=-0.095)
    parser.add_argument("--ik-carry-z-extra", type=float, default=0.020)
    parser.add_argument("--ik-lift-shoulder-delta", type=float, default=0.24)
    parser.add_argument("--ik-lift-elbow-delta", type=float, default=0.14)
    parser.add_argument("--ik-lift-waist-delta", type=float, default=0.00)
    parser.add_argument("--ik-iters", type=int, default=80)
    parser.add_argument("--ik-damping", type=float, default=0.045)
    parser.add_argument("--ik-regularization", type=float, default=0.010)
    parser.add_argument("--ik-step-limit", type=float, default=0.12)
    parser.add_argument("--ik-max-error", type=float, default=0.140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = BoxGraspDemo(args)
    try:
        node.run()
    except KeyboardInterrupt:
        _log("stopping box grasp demo")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
