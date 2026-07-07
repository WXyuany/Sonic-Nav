#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Sequence

import mujoco
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from box_grasp_demo import (  # noqa: E402
    GRASP_ASSIST_FILE,
    JOINT_LIMITS,
    LOCO_IDLE,
    LOCO_SLOW_WALK,
    NEUTRAL_HAND,
    UPPER_BODY_ORDER,
    DemoCommand,
    DemoPhase,
    _clamp,
    _closed_hands,
    _command,
    _interp_command,
    _lerp,
    _named_upper_body,
    _smoothstep,
    _velocity,
)
from g1_ros2_nav.tmp_io import load_npy_if_ready  # noqa: E402
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
)
from wam_primitives import (  # noqa: E402
    ContactServoConfig,
    ContactServoPolicy,
    GraspQuality,
    WorkspaceAligner,
)


RIGHT_IK_ORDER = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
FINGER_CONTACT_LOCAL = {
    "thumb": np.asarray([0.0, 0.034, 0.0], dtype=np.float64),
    "middle": np.asarray([0.040, 0.0, 0.0], dtype=np.float64),
    "index": np.asarray([0.040, 0.0, 0.0], dtype=np.float64),
}
PALM_CONTACT_LOCAL = np.asarray([0.102, -0.018, 0.0], dtype=np.float64)
FULL_INDEX = {name: i for i, name in enumerate(UPPER_BODY_ORDER)}


def _log(text: str) -> None:
    print(f"[ball_pick_place_demo] {text}", flush=True)


def _finite_vec3(value) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def _apply_ball_scaled_grasp(args: argparse.Namespace, radius: float) -> None:
    if not bool(getattr(args, "auto_ball_grasp", True)):
        return
    r = max(0.015, float(radius))
    args.simple_tip_x_offset = float(args.simple_tip_x_radius_scale) * r
    args.simple_tip_y_offset = float(args.simple_tip_y_radius_scale) * r
    args.simple_tip_z_offset = float(args.simple_tip_z_radius_scale) * r

    close = _clamp(
        float(args.finger_close_intercept) - float(args.finger_close_radius_gain) * r,
        float(args.finger_close_min),
        float(args.finger_close_max),
    )
    args.close_ratio = close
    args.capture_close_ratio = _clamp(
        close - float(args.capture_open_delta),
        float(args.capture_close_min),
        close,
    )
    args.preload_close_ratio = _clamp(
        close + float(args.preload_close_extra),
        close,
        float(args.max_hold_close_ratio),
    )
    args.squeeze_close_ratio = _clamp(
        close + float(args.squeeze_close_extra),
        close,
        float(args.max_hold_close_ratio),
    )
    args.hold_close_ratio = _clamp(
        close + float(args.hold_close_extra),
        close,
        float(args.max_hold_close_ratio),
    )


class MujocoRightHandIK:
    def __init__(
        self,
        scene: str,
        *,
        qpos_path: str,
        hand_body: str,
        wrist_limits: dict[str, tuple[float, float]] | None = None,
    ):
        selection = resolve_scene(scene, repo_root=REPO)
        self.model = mujoco.MjModel.from_xml_path(str(selection.abs_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_path = qpos_path
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        if min(self.base_body_id, self.hand_body_id) < 0:
            raise RuntimeError(f"required bodies not found for IK: pelvis or {hand_body}")
        self.finger_body_ids = {
            "thumb": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link"),
            "middle": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_1_link"),
            "index": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link"),
        }
        if min(self.finger_body_ids.values()) < 0:
            raise RuntimeError("required right-hand finger bodies were not found for IK")
        self.palm_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_wrist_yaw_link",
        )
        if self.palm_body_id < 0:
            raise RuntimeError("required right-hand palm body was not found for IK")
        self.contact_body_ids = {"palm": self.palm_body_id, **self.finger_body_ids}

        self.joint_qpos_ids = []
        self.joint_dof_ids = []
        self.joint_ranges = []
        for name in RIGHT_IK_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"joint '{name}' not found for right-hand IK")
            self.joint_qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
            low, high = (float(v) for v in self.model.jnt_range[joint_id])
            if name == "waist_yaw_joint":
                low, high = max(low, -0.30), min(high, 0.30)
            elif name == "waist_roll_joint":
                low, high = max(low, -0.10), min(high, 0.10)
            elif name == "waist_pitch_joint":
                low, high = max(low, -0.08), min(high, 0.18)
            if wrist_limits is not None and name in wrist_limits:
                wrist_low, wrist_high = wrist_limits[name]
                low, high = max(low, float(wrist_low)), min(high, float(wrist_high))
            self.joint_ranges.append((low, high))
        self.joint_qpos_ids = np.asarray(self.joint_qpos_ids, dtype=np.int32)
        self.joint_dof_ids = np.asarray(self.joint_dof_ids, dtype=np.int32)
        self.joint_ranges = np.asarray(self.joint_ranges, dtype=np.float64)
        self.right_hand_qpos_ids = []
        for name in RIGHT_HAND_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"joint '{name}' not found for right-hand IK")
            self.right_hand_qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
        self.right_hand_qpos_ids = np.asarray(self.right_hand_qpos_ids, dtype=np.int32)

    def _sync_live_qpos(self) -> None:
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

    def _palm_basis_base(self) -> np.ndarray | None:
        qpos = self._live_qpos()
        if qpos is None:
            return None
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        palm_rot = self.data.xmat[self.palm_body_id].reshape(3, 3)
        return base_rot.T @ palm_rot

    def _solve(
        self,
        target_base: Sequence[float],
        seed_full: Sequence[float],
        *,
        max_iters: int,
        damping: float,
        regularization: float,
        step_limit: float,
        hand_pose: Sequence[float] | None = None,
    ) -> tuple[list[float], float]:
        return self._solve_targets(
            [(self.hand_body_id, target_base, 1.0)],
            seed_full,
            max_iters=max_iters,
            damping=damping,
            regularization=regularization,
            step_limit=step_limit,
            hand_pose=hand_pose,
        )

    def _solve_targets(
        self,
        targets_base: Sequence[tuple[int, Sequence[float], float]],
        seed_full: Sequence[float],
        *,
        max_iters: int,
        damping: float,
        regularization: float,
        step_limit: float,
        hand_pose: Sequence[float] | None = None,
    ) -> tuple[list[float], float]:
        self._sync_live_qpos()
        q = self.data.qpos.copy()
        seed_full_arr = np.asarray(seed_full, dtype=np.float64)
        seed_sel = np.asarray([seed_full_arr[FULL_INDEX[name]] for name in RIGHT_IK_ORDER], dtype=np.float64)
        seed_sel = np.minimum(self.joint_ranges[:, 1], np.maximum(self.joint_ranges[:, 0], seed_sel))
        q[self.joint_qpos_ids] = seed_sel
        if hand_pose is not None:
            hand_arr = np.asarray(hand_pose, dtype=np.float64)
            q[self.right_hand_qpos_ids] = hand_arr[: len(self.right_hand_qpos_ids)]
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        targets = []
        for raw_target in targets_base:
            if len(raw_target) == 3:
                body_id, target_base, weight = raw_target
                local_point = np.zeros(3, dtype=np.float64)
            elif len(raw_target) == 4:
                body_id, target_base, weight, local_point = raw_target
                local_point = np.asarray(local_point, dtype=np.float64)
            else:
                raise ValueError(f"bad IK target tuple: {raw_target!r}")
            targets.append(
                (
                    int(body_id),
                    self._base_to_world(target_base),
                    math.sqrt(max(1e-6, float(weight))),
                    local_point,
                )
            )
        last_err = np.inf
        for _ in range(max(1, int(max_iters))):
            err_parts = []
            jac_parts = []
            unweighted_errors = []
            for body_id, target_world, weight, local_point in targets:
                body_rot = self.data.xmat[body_id].reshape(3, 3)
                point_world = self.data.xpos[body_id] + body_rot @ local_point
                err_i = target_world - point_world
                unweighted_errors.append(float(np.linalg.norm(err_i)))
                err_parts.append(weight * err_i)
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jac(self.model, self.data, jacp, jacr, point_world, body_id)
                jac_parts.append(weight * jacp[:, self.joint_dof_ids])
            err = np.concatenate(err_parts)
            jac = np.vstack(jac_parts)
            if unweighted_errors:
                err_values = np.asarray(unweighted_errors, dtype=np.float64)
                last_err = float(math.sqrt(np.mean(err_values * err_values)))
            else:
                last_err = 0.0
            if last_err < 0.010:
                break

            q_sel = self.data.qpos[self.joint_qpos_ids]
            lhs = jac.T @ jac + (float(damping) ** 2 + float(regularization)) * np.eye(len(seed_sel))
            rhs = jac.T @ err - float(regularization) * (q_sel - seed_sel)
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

        solved = list(float(v) for v in seed_full_arr)
        for name, value in zip(RIGHT_IK_ORDER, self.data.qpos[self.joint_qpos_ids]):
            solved[FULL_INDEX[name]] = float(value)
        return solved, last_err

    def _contact_targets(
        self,
        center_base: np.ndarray,
        radius: float,
        *,
        table_contact: bool = False,
        args: argparse.Namespace | None = None,
    ) -> list[tuple[int, np.ndarray, float, np.ndarray]]:
        r = max(0.025, float(radius))
        if table_contact:
            z_offsets = (0.04, -0.04, 0.18)
            palm_z = float(getattr(args, "palm_pocket_table_z_radius", 0.00))
        else:
            z_offsets = (-0.18, -0.24, 0.24)
            palm_z = float(getattr(args, "palm_pocket_lift_z_radius", 0.02))
        palm_x = float(getattr(args, "palm_pocket_x_radius", -1.85))
        palm_y = float(getattr(args, "palm_pocket_y_radius", -0.10))
        palm_weight = float(getattr(args, "palm_contact_weight", 1.05))
        basis = None
        if bool(getattr(args, "palm_frame_contact_targets", True)):
            basis = self._palm_basis_base()

        def contact_offset(local_offset: Sequence[float]) -> np.ndarray:
            local = np.asarray(local_offset, dtype=np.float64)
            if basis is None:
                return local
            return basis @ local

        return [
            (
                self.palm_body_id,
                center_base + contact_offset([palm_x * r, palm_y * r, palm_z * r]),
                palm_weight,
                PALM_CONTACT_LOCAL,
            ),
            (
                self.finger_body_ids["thumb"],
                center_base + contact_offset([-0.12 * r, 1.05 * r, z_offsets[0] * r]),
                1.25,
                FINGER_CONTACT_LOCAL["thumb"],
            ),
            (
                self.finger_body_ids["middle"],
                center_base + contact_offset([0.22 * r, -1.05 * r, z_offsets[1] * r]),
                1.35,
                FINGER_CONTACT_LOCAL["middle"],
            ),
            (
                self.finger_body_ids["index"],
                center_base + contact_offset([0.22 * r, -0.58 * r, z_offsets[2] * r]),
                1.15,
                FINGER_CONTACT_LOCAL["index"],
            ),
        ]

    def _finger_targets(
        self,
        center_base: np.ndarray,
        radius: float,
        *,
        table_contact: bool = False,
    ) -> list[tuple[int, np.ndarray, float, np.ndarray]]:
        return self._contact_targets(center_base, radius, table_contact=table_contact)

    def _live_qpos(self) -> np.ndarray | None:
        try:
            qpos = load_npy_if_ready(self.qpos_path)
        except OSError:
            return None
        if qpos is None:
            return None
        out = np.asarray(qpos, dtype=np.float64)
        if out.size < self.model.nq:
            padded = self.data.qpos.copy()
            padded[: out.size] = out
            return padded
        return out[: self.model.nq].copy()

    def live_upper_body_pose(self, fallback: Sequence[float]) -> list[float]:
        qpos = self._live_qpos()
        if qpos is None:
            return list(float(v) for v in fallback)
        pose = list(float(v) for v in fallback)
        for name in UPPER_BODY_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            pose[FULL_INDEX[name]] = float(qpos[self.model.jnt_qposadr[joint_id]])
        return pose

    def contact_points_base(self) -> dict[str, np.ndarray] | None:
        qpos = self._live_qpos()
        if qpos is None:
            return None
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        points: dict[str, np.ndarray] = {}
        contact_local_points = {"palm": PALM_CONTACT_LOCAL, **FINGER_CONTACT_LOCAL}
        for name, body_id in self.contact_body_ids.items():
            local_point = contact_local_points[name]
            point_world = self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ local_point
            points[name] = base_rot.T @ (point_world - base_pos)
        return points

    def fingertip_points_base(self) -> dict[str, np.ndarray] | None:
        points = self.contact_points_base()
        if points is None:
            return None
        return {name: points[name] for name in self.finger_body_ids if name in points}

    def contact_error_base(
        self,
        center_base: np.ndarray,
        radius: float,
        *,
        table_contact: bool,
        args: argparse.Namespace | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        actual = self.contact_points_base()
        if actual is None:
            return None
        targets = self._contact_targets(center_base, radius, table_contact=table_contact, args=args)
        errors = []
        per_contact: dict[str, np.ndarray] = {}
        id_to_name = {body_id: name for name, body_id in self.contact_body_ids.items()}
        for body_id, target, _weight, _local in targets:
            name = id_to_name.get(int(body_id))
            if name is None or name not in actual:
                continue
            error = actual[name] - np.asarray(target, dtype=np.float64)
            per_contact[name] = error
            errors.append(error)
        if not errors:
            return None
        return np.mean(np.vstack(errors), axis=0), per_contact

    def solve_contact_pose(
        self,
        center_base: np.ndarray,
        radius: float,
        seed_full: Sequence[float],
        args: argparse.Namespace,
        *,
        hand_pose: Sequence[float],
        table_contact: bool,
    ) -> tuple[list[float], float]:
        return self._solve_targets(
            self._contact_targets(center_base, radius, table_contact=table_contact, args=args),
            seed_full,
            max_iters=int(args.servo_ik_iters),
            damping=float(args.ik_damping),
            regularization=float(args.servo_ik_regularization),
            step_limit=float(args.ik_step_limit),
            hand_pose=hand_pose,
        )

    def solve_pick_place_poses(
        self,
        anchor: dict,
        fallback_poses: dict[str, list[float]],
        args: argparse.Namespace,
    ) -> tuple[dict[str, list[float]], dict[str, float]]:
        ball_base = np.asarray(anchor.get("ball_point_base", [np.nan, np.nan, np.nan]), dtype=np.float64)
        place_base = np.asarray(anchor.get("place_point_base", [np.nan, np.nan, np.nan]), dtype=np.float64)
        if ball_base.shape != (3,) or place_base.shape != (3,):
            raise ValueError("bad ball anchor for IK")
        if not np.all(np.isfinite(ball_base)) or not np.all(np.isfinite(place_base)):
            raise ValueError("bad ball/place anchor for IK")

        radius = float(anchor.get("ball_radius", 0.062))
        side_y = float(args.ik_y_offset)
        if args.auto_side_offset:
            side_y = -_clamp(radius * float(args.ik_side_radius_scale), 0.018, 0.045)

        pick = np.asarray(
            [
                float(ball_base[0]) + float(args.ik_x_offset),
                float(ball_base[1]) + side_y,
                float(ball_base[2]) + float(args.ik_z_offset),
            ],
            dtype=np.float64,
        )
        ball_center = np.asarray(ball_base, dtype=np.float64)
        hover = pick + np.asarray([float(args.ik_hover_x_delta), float(args.ik_hover_y_delta), float(args.ik_hover_z)], dtype=np.float64)
        pregrasp = pick + np.asarray([float(args.ik_pregrasp_x_delta), 0.0, float(args.ik_pregrasp_z)], dtype=np.float64)
        lift_center = ball_center + np.asarray([float(args.ik_lift_x_delta), 0.0, float(args.ik_lift_z)], dtype=np.float64)
        lift = pick + np.asarray([float(args.ik_lift_x_delta), 0.0, float(args.ik_lift_z)], dtype=np.float64)
        place = np.asarray(
            [
                float(place_base[0]) + float(args.ik_place_x_offset),
                float(place_base[1]) + float(args.ik_place_y_offset),
                float(place_base[2]) + float(args.ik_place_z_offset),
            ],
            dtype=np.float64,
        )
        place_center = np.asarray(place_base, dtype=np.float64)
        place_hover = place + np.asarray([float(args.ik_place_hover_x_delta), 0.0, float(args.ik_place_hover_z)], dtype=np.float64)
        place_hover_center = place_center + np.asarray([float(args.ik_place_hover_x_delta), 0.0, float(args.ik_place_hover_z)], dtype=np.float64)
        retreat = place + np.asarray([float(args.ik_retreat_x_delta), float(args.ik_retreat_y_delta), float(args.ik_retreat_z)], dtype=np.float64)

        targets = {
            "high_ready": hover,
            "pregrasp": pregrasp,
            "grasp": pick,
            "lift": lift,
            "secure": lift + np.asarray([0.0, 0.0, float(args.ik_secure_z_extra)], dtype=np.float64),
            "place_hover": place_hover,
            "place_low": place,
            "retreat": retreat,
        }
        simple_tip_targets = {}
        contact_targets = {}
        if bool(getattr(args, "simple_clamp_grasp", True)):
            simple_offset = np.asarray(
                [
                    float(args.simple_tip_x_offset),
                    float(args.simple_tip_y_offset),
                    float(args.simple_tip_z_offset),
                ],
                dtype=np.float64,
            )
            simple_tip_targets = {
                "grasp": ball_center + simple_offset,
                "lift": lift_center + simple_offset,
                "secure": lift_center
                + simple_offset
                + np.asarray([0.0, 0.0, float(args.ik_secure_z_extra)], dtype=np.float64),
                "place_hover": place_hover_center + simple_offset,
                "place_low": place_center + simple_offset,
            }
        else:
            contact_targets = {
                "grasp": ball_center,
                "lift": lift_center,
                "secure": lift_center + np.asarray([0.0, 0.0, float(args.ik_secure_z_extra)], dtype=np.float64),
                "place_hover": place_hover_center,
                "place_low": place_center,
            }
        poses: dict[str, list[float]] = {}
        errors: dict[str, float] = {}
        seed = fallback_poses["high_ready"]
        _, right_grasp = _closed_hands(args.close_ratio)
        _, right_squeeze = _closed_hands(args.squeeze_close_ratio)
        hand_poses = {
            "high_ready": NEUTRAL_HAND,
            "pregrasp": NEUTRAL_HAND,
            "grasp": right_grasp,
            "lift": right_squeeze,
            "secure": right_squeeze,
            "place_hover": right_squeeze,
            "place_low": right_squeeze,
            "retreat": NEUTRAL_HAND,
        }
        for name in ["high_ready", "pregrasp", "grasp", "lift", "secure", "place_hover", "place_low", "retreat"]:
            seed = poses.get(name, seed)
            if name in simple_tip_targets:
                solved, err = self._solve_targets(
                    [
                        (
                            self.finger_body_ids["middle"],
                            simple_tip_targets[name],
                            1.0,
                            FINGER_CONTACT_LOCAL["middle"],
                        )
                    ],
                    seed,
                    max_iters=args.ik_iters,
                    damping=args.ik_damping,
                    regularization=args.ik_regularization,
                    step_limit=args.ik_step_limit,
                    hand_pose=hand_poses.get(name),
                )
            elif name in contact_targets:
                solved, err = self._solve_targets(
                    self._contact_targets(
                        contact_targets[name],
                        radius,
                        table_contact=name in {"grasp", "place_low"},
                        args=args,
                    ),
                    seed,
                    max_iters=args.ik_iters,
                    damping=args.ik_damping,
                    regularization=args.ik_regularization,
                    step_limit=args.ik_step_limit,
                    hand_pose=hand_poses.get(name),
                )
            else:
                solved, err = self._solve(
                    targets[name],
                    seed,
                    max_iters=args.ik_iters,
                    damping=args.ik_damping,
                    regularization=args.ik_regularization,
                    step_limit=args.ik_step_limit,
                    hand_pose=hand_poses.get(name),
                )
            poses[name] = solved
            errors[name] = err
            seed = solved
        return poses, errors


def _single_right_pose(
    *,
    waist_yaw: float = 0.0,
    waist_roll: float = 0.0,
    waist_pitch: float = 0.0,
    right_pitch: float,
    right_roll: float,
    right_yaw: float,
    right_elbow: float,
    right_wrist_roll: float = 0.0,
    right_wrist_pitch: float = 0.0,
    right_wrist_yaw: float = 0.0,
) -> list[float]:
    return _named_upper_body(
        waist_yaw_joint=waist_yaw,
        waist_roll_joint=waist_roll,
        waist_pitch_joint=waist_pitch,
        left_shoulder_pitch_joint=0.18,
        left_shoulder_roll_joint=0.24,
        left_shoulder_yaw_joint=-0.02,
        left_elbow_joint=0.66,
        left_wrist_pitch_joint=0.0,
        right_shoulder_pitch_joint=right_pitch,
        right_shoulder_roll_joint=right_roll,
        right_shoulder_yaw_joint=right_yaw,
        right_elbow_joint=right_elbow,
        right_wrist_roll_joint=right_wrist_roll,
        right_wrist_pitch_joint=right_wrist_pitch,
        right_wrist_yaw_joint=right_wrist_yaw,
    )


def _upper_body_poses(args: argparse.Namespace) -> dict[str, list[float]]:
    neutral = _named_upper_body()
    table_clear = _single_right_pose(
        waist_yaw=-0.02,
        waist_pitch=0.0,
        right_pitch=-0.98,
        right_roll=-0.30,
        right_yaw=0.20,
        right_elbow=0.62,
        right_wrist_pitch=0.08,
    )
    high_ready = _single_right_pose(
        waist_yaw=-0.03,
        waist_pitch=0.02,
        right_pitch=args.ready_pitch,
        right_roll=args.ready_roll,
        right_yaw=args.ready_yaw,
        right_elbow=args.ready_elbow,
        right_wrist_pitch=args.ready_wrist_pitch,
    )
    pregrasp = _single_right_pose(
        waist_yaw=-0.05,
        waist_pitch=args.reach_waist_pitch,
        right_pitch=args.reach_pitch,
        right_roll=args.reach_roll,
        right_yaw=args.reach_yaw,
        right_elbow=args.reach_elbow,
        right_wrist_pitch=args.reach_wrist_pitch,
    )
    grasp = _single_right_pose(
        waist_yaw=-0.06,
        waist_pitch=args.reach_waist_pitch,
        right_pitch=args.grasp_pitch,
        right_roll=args.grasp_roll,
        right_yaw=args.grasp_yaw,
        right_elbow=args.grasp_elbow,
        right_wrist_pitch=args.grasp_wrist_pitch,
    )
    lift = _single_right_pose(
        waist_yaw=-0.03,
        waist_pitch=0.02,
        right_pitch=args.lift_pitch,
        right_roll=args.lift_roll,
        right_yaw=args.lift_yaw,
        right_elbow=args.lift_elbow,
        right_wrist_pitch=args.lift_wrist_pitch,
    )
    place_hover = _single_right_pose(
        waist_yaw=0.08,
        waist_pitch=0.02,
        right_pitch=args.place_pitch,
        right_roll=args.place_roll,
        right_yaw=args.place_yaw,
        right_elbow=args.place_elbow,
        right_wrist_pitch=args.place_wrist_pitch,
    )
    place_low = _single_right_pose(
        waist_yaw=0.08,
        waist_pitch=args.reach_waist_pitch,
        right_pitch=args.place_low_pitch,
        right_roll=args.place_low_roll,
        right_yaw=args.place_yaw,
        right_elbow=args.place_low_elbow,
        right_wrist_pitch=args.place_low_wrist_pitch,
    )
    retreat = _single_right_pose(
        waist_yaw=0.02,
        waist_pitch=0.0,
        right_pitch=args.retreat_pitch,
        right_roll=args.retreat_roll,
        right_yaw=args.retreat_yaw,
        right_elbow=args.retreat_elbow,
        right_wrist_pitch=args.retreat_wrist_pitch,
    )
    return {
        "neutral": neutral,
        "table_clear": table_clear,
        "high_ready": high_ready,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
        "secure": lift,
        "place_hover": place_hover,
        "place_low": place_low,
        "retreat": retreat,
    } | getattr(args, "ik_pose_overrides", {})


def make_demo_phases(args: argparse.Namespace) -> list[DemoPhase]:
    poses = _upper_body_poses(args)
    _, right_grasp = _closed_hands(args.close_ratio)
    _, right_squeeze = _closed_hands(args.squeeze_close_ratio)
    stand = _command(mode=LOCO_IDLE, upper_body=poses["neutral"], height=args.stand_height)
    table_clear = _command(mode=LOCO_IDLE, upper_body=poses["table_clear"], height=args.stand_height)
    walk = _command(
        mode=LOCO_SLOW_WALK,
        movement=(1.0, 0.0, 0.0),
        speed=args.walk_speed,
        upper_body=poses["table_clear"],
        height=args.stand_height,
    )
    align_clear = _command(
        mode=(
            LOCO_SLOW_WALK
            if max(abs(float(args.align_movement_x)), abs(float(args.align_movement_y))) > 1e-3
            else LOCO_IDLE
        ),
        movement=(float(args.align_movement_x), float(args.align_movement_y), 0.0),
        speed=args.align_speed,
        upper_body=poses["table_clear"],
        height=args.stand_height,
    )
    align_ready = _command(
        mode=(
            LOCO_SLOW_WALK
            if max(abs(float(args.align_movement_x)), abs(float(args.align_movement_y))) > 1e-3
            else LOCO_IDLE
        ),
        movement=(float(args.align_movement_x), float(args.align_movement_y), 0.0),
        speed=args.align_speed,
        upper_body=poses["high_ready"],
        height=args.stand_height,
    )
    ready = _command(mode=LOCO_IDLE, upper_body=poses["high_ready"], height=args.stand_height)
    pregrasp = _command(mode=LOCO_IDLE, upper_body=poses["pregrasp"], height=args.stand_height)
    grasp_open = _command(mode=LOCO_IDLE, upper_body=poses["grasp"], height=args.stand_height)
    pregrasp_open = _command(mode=LOCO_IDLE, upper_body=poses["pregrasp"], height=args.stand_height)
    _, right_capture = _closed_hands(args.capture_close_ratio)
    capture = _command(
        mode=LOCO_IDLE,
        upper_body=poses["grasp"],
        height=args.stand_height,
        right_hand=right_capture,
    )
    grasp = _command(
        mode=LOCO_IDLE,
        upper_body=poses["grasp"],
        height=args.stand_height,
        right_hand=right_grasp,
    )
    squeeze = _command(
        mode=LOCO_IDLE,
        upper_body=poses["grasp"],
        height=args.stand_height,
        right_hand=right_squeeze,
    )
    lift = _command(
        mode=LOCO_IDLE,
        upper_body=poses["lift"],
        height=args.stand_height,
        right_hand=right_squeeze,
    )
    secure = _command(
        mode=LOCO_IDLE,
        upper_body=poses["secure"],
        height=args.stand_height,
        right_hand=right_squeeze,
    )
    place_hover = _command(
        mode=LOCO_IDLE,
        upper_body=poses["place_hover"],
        height=args.stand_height,
        right_hand=right_squeeze,
    )
    place_low = _command(
        mode=LOCO_IDLE,
        upper_body=poses["place_low"],
        height=args.stand_height,
        right_hand=right_squeeze,
    )
    release = _command(mode=LOCO_IDLE, upper_body=poses["place_low"], height=args.stand_height)
    retreat = _command(mode=LOCO_IDLE, upper_body=poses["retreat"], height=args.stand_height)
    return [
        DemoPhase("stand_ready", 1.0, stand, stand),
        DemoPhase("raise_hand_clear", args.clear_hand_duration, stand, table_clear),
        DemoPhase("walk_to_table", args.walk_duration, walk, walk),
        DemoPhase("fine_align_to_ball", args.align_duration, align_clear, align_clear),
        DemoPhase("settle_before_pick", args.settle_duration, table_clear, table_clear),
        DemoPhase("hand_high_ready", args.prepare_duration, table_clear, ready),
        DemoPhase("fine_align_before_grasp", args.align_duration, align_ready, align_ready),
        DemoPhase("approach_from_above", args.pregrasp_duration, ready, pregrasp),
        DemoPhase("lower_to_ball_open", args.reach_duration, pregrasp_open, grasp_open),
        DemoPhase("capture_ball_contact", args.capture_duration, pregrasp_open, capture),
        DemoPhase("close_on_ball", args.close_duration, capture, grasp),
        DemoPhase("squeeze_ball_secure", args.palm_squeeze_duration, grasp, squeeze),
        DemoPhase("lift_ball", args.lift_duration, squeeze, lift),
        DemoPhase("secure_ball", args.secure_duration, lift, secure),
        DemoPhase("move_to_place", args.transfer_duration, secure, place_hover),
        DemoPhase("lower_to_place", args.place_duration, place_hover, place_low),
        DemoPhase("release_ball", args.release_duration, place_low, release),
        DemoPhase("retreat_hand", args.retreat_duration, release, retreat),
        DemoPhase("hold_done", args.hold_duration, retreat, retreat),
    ]


def apply_ball_anchor(args: argparse.Namespace, anchor: dict, *, update_walk: bool = True) -> str:
    grasp = anchor.get("grasp") or {}
    if update_walk:
        if "walk_speed" in grasp:
            setattr(args, "walk_speed", float(grasp["walk_speed"]))
        if "walk_duration" in grasp:
            walk_duration = float(grasp["walk_duration"]) + float(args.walk_extra_duration)
            setattr(args, "walk_duration", min(float(args.max_approach_duration), walk_duration))
    if "approach_target_x" in grasp:
        args.approach_target_x = float(grasp["approach_target_x"])

    ball_base = _finite_vec3(anchor.get("ball_point_base")) or [math.nan, math.nan, math.nan]
    place_base = _finite_vec3(anchor.get("place_point_base")) or [math.nan, math.nan, math.nan]
    if all(math.isfinite(v) for v in ball_base):
        args.assist_x = float(ball_base[0])
        args.assist_y = float(ball_base[1])
        args.assist_z = float(ball_base[2])
    if all(math.isfinite(v) for v in place_base):
        args.place_assist_x = float(place_base[0])
        args.place_assist_y = float(place_base[1])
        args.place_assist_z = float(place_base[2])
    align_error = math.nan
    align_error_x = math.nan
    if all(math.isfinite(v) for v in ball_base):
        align_error_x = float(ball_base[0]) - float(args.align_target_x)
        if align_error_x > float(args.align_x_tolerance):
            args.align_movement_x = _clamp(
                float(args.align_forward_gain) * align_error_x,
                0.0,
                float(args.align_max_forward),
            )
        else:
            args.align_movement_x = 0.0
        align_plan = WorkspaceAligner(
            target_y=float(args.align_target_y),
            tolerance=float(args.align_y_tolerance),
            lateral_gain=float(args.align_lateral_gain),
            speed=float(args.align_speed),
            min_duration=float(args.align_min_duration),
            max_duration=float(args.align_max_duration),
            duration_gain=float(args.align_duration_gain),
        ).plan(float(ball_base[1]))
        align_error = float(align_plan.error_y)
        args.align_movement_y = float(align_plan.movement_y)
        x_duration = 0.0
        if align_error_x > float(args.align_x_tolerance):
            x_duration = (
                align_error_x
                / max(0.03, float(args.align_speed))
                * max(0.1, float(args.align_duration_gain))
            )
        args.align_duration = float(
            _clamp(
                max(float(align_plan.duration), x_duration),
                float(args.align_min_duration),
                float(args.align_max_duration),
            )
        )

    radius = float(anchor.get("ball_radius", args.ball_radius))
    args.ball_radius = radius
    _apply_ball_scaled_grasp(args, radius)
    reach_x = float(grasp.get("reach_x", ball_base[0] if math.isfinite(ball_base[0]) else 0.46))
    lateral = float(grasp.get("target_y", ball_base[1] if math.isfinite(ball_base[1]) else -0.12))
    args.reach_pitch = -_clamp(0.52 + 0.50 * max(0.0, reach_x - 0.38), 0.50, 0.78)
    args.grasp_pitch = _clamp(args.reach_pitch - 0.04, -0.82, -0.56)
    args.ready_roll = _clamp(-0.34 + 0.20 * min(0.0, lateral), -0.48, -0.26)
    args.reach_roll = _clamp(-0.42 + 0.30 * min(0.0, lateral), -0.58, -0.30)
    args.grasp_roll = _clamp(args.reach_roll - 0.03, -0.60, -0.34)
    args.lift_roll = _clamp(args.grasp_roll + 0.08, -0.50, -0.28)

    ball_camera = anchor.get("ball_point_camera_depth", [math.nan, math.nan, math.nan])
    stage = "initial" if update_walk else "post_walk"
    return (
        f"{stage} "
        f"ball_base=({float(ball_base[0]):.2f},{float(ball_base[1]):.2f},{float(ball_base[2]):.2f}) "
        f"place_base=({float(place_base[0]):.2f},{float(place_base[1]):.2f},{float(place_base[2]):.2f}) "
        f"camera_depth=({float(ball_camera[0]):.2f},{float(ball_camera[1]):.2f},{float(ball_camera[2]):.2f}) "
        f"walk={args.walk_duration:.2f}s target_x={args.approach_target_x:.2f} "
        f"align=({args.align_movement_x:.2f},{args.align_movement_y:.2f})/{args.align_duration:.2f}s "
        f"err=({align_error_x:.2f},{align_error:.2f}) "
        f"radius={radius:.3f} close={args.close_ratio:.2f} "
        f"tip=({args.simple_tip_x_offset:.3f},{args.simple_tip_y_offset:.3f},{args.simple_tip_z_offset:.3f}) "
        f"reach_pitch={args.reach_pitch:.2f}"
    )


class BallPickPlaceDemo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ball_pick_place_demo")
        self.args = args
        self.args.ik_pose_overrides = {}
        self.ik_solver = None
        if args.ik_upper_body:
            try:
                self.ik_solver = MujocoRightHandIK(
                    args.scene,
                    qpos_path=args.qpos_path,
                    hand_body=args.ik_hand_body,
                    wrist_limits={
                        "right_wrist_roll_joint": (
                            -float(args.ik_wrist_roll_limit),
                            float(args.ik_wrist_roll_limit),
                        ),
                        "right_wrist_pitch_joint": (
                            float(args.ik_wrist_pitch_min),
                            float(args.ik_wrist_pitch_max),
                        ),
                        "right_wrist_yaw_joint": (
                            -float(args.ik_wrist_yaw_limit),
                            float(args.ik_wrist_yaw_limit),
                        ),
                    },
                )
                _log(f"IK right-hand solver ready: scene={args.scene} hand_body={args.ik_hand_body}")
            except Exception as exc:
                if args.require_ik:
                    raise
                self.get_logger().warn(f"right-hand IK disabled: {exc}")

        self.phase_pub = self.create_publisher(String, "/sonic_demo/phase", 10)
        self.latest_anchor: dict | None = None
        self._anchor_wall_time = 0.0
        if args.use_ball_anchor:
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
        self._ball_pre_lift_z: float | None = None
        self._last_servo_time = 0.0
        self._last_servo_log = 0.0
        self._servo_cache_phase: str | None = None
        self._servo_cache_cmd: DemoCommand | None = None
        self._contact_center_bias = np.zeros(3, dtype=np.float64)
        self._last_contact_error_norm: float | None = None
        self._last_contact_max_error: float | None = None
        self._last_contact_ready = False
        self._contact_capture_retries = 0
        self._contact_squeeze_retries = 0
        self._lift_retries = 0
        self._workspace_align_retries: dict[str, int] = {}
        self._last_workspace_servo_log = 0.0
        self._lift_start_ball_map_z: float | None = None
        self._ball_lifted_seen = False
        _log(f"ZMQ publisher bound: {self.endpoint}")
        if self.args.ball_attach:
            _log("ball contact-lock assist explicitly enabled")
        else:
            _log("ball contact-lock assist disabled; using contact/friction grasp only")
        if self.args.contact_servo:
            _log("contact servo enabled")
        else:
            _log("contact servo disabled; using direct first-reach clamp")

    def run(self) -> None:
        time.sleep(float(self.args.zmq_connect_wait))
        self._write_ball_attach(False, "warmup")
        time.sleep(float(self.args.warmup))
        initial_anchor_ok = True
        if self.args.use_ball_anchor:
            initial_anchor_ok = self._wait_and_apply_anchor()
            if self.args.require_ball_anchor and not initial_anchor_ok:
                raise RuntimeError(
                    f"ball anchor is required but no anchor was received on {self.args.anchor_topic}"
                )
        self._send_start()

        period = 1.0 / max(5.0, float(self.args.rate))
        phase_index = 0
        approach_retries = 0
        while phase_index < len(self.phases):
            phase = self.phases[phase_index]
            if (
                phase.name == "hand_high_ready"
                and self.args.use_ball_anchor
                and not self._post_walk_anchor_applied
            ):
                self._wait_and_apply_anchor(
                    update_walk=False,
                    timeout=1.2,
                    fresh_after=time.monotonic(),
                )
                self._post_walk_anchor_applied = True
                phase = self.phases[phase_index]
            elif phase.name.startswith("fine_align") and self.args.use_ball_anchor:
                self._wait_and_apply_anchor(
                    update_walk=False,
                    timeout=1.0,
                    fresh_after=time.monotonic(),
                )
                phase = self.phases[phase_index]
            elif (
                phase.name == "approach_from_above"
                and self.args.use_ball_anchor
                and self.args.retrack_before_pick
            ):
                self._retrack_anchor_for_pick()
                phase = self.phases[phase_index]

            _log(f"phase: {phase.name} ({phase.duration:.1f}s)")
            phase_msg = String()
            phase_msg.data = phase.name
            if phase.name == "lift_ball":
                self._mark_lift_reference()
            t0 = time.monotonic()
            while rclpy.ok():
                elapsed = time.monotonic() - t0
                if elapsed >= phase.duration:
                    break
                ratio = elapsed / max(1e-3, phase.duration)
                cmd = _interp_command(phase.start, phase.end, ratio)
                cmd = self._workspace_servo_command(phase.name, cmd)
                cmd = self._contact_servo_command(phase.name, ratio, cmd)
                self._publish_planner(cmd)
                self._update_ball_attach(phase.name, ratio)
                self.phase_pub.publish(phase_msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                if self._should_finish_walk(phase.name, elapsed):
                    break
                if self._should_finish_align(phase.name, elapsed):
                    break
                time.sleep(period)
            end_cmd = self._contact_servo_command(phase.name, 1.0, phase.end)
            self._publish_planner(end_cmd)
            self._update_ball_attach(phase.name, 1.0)
            if phase.name == "lift_ball" and self.args.use_ball_anchor:
                lifted = self._wait_for_lifted_ball(timeout=float(self.args.lift_detect_timeout))
                if not lifted and self._lift_retries < int(self.args.lift_max_retries):
                    self._lift_retries += 1
                    _log(
                        f"lift not stable: retry {self._lift_retries}/{self.args.lift_max_retries}"
                    )
                    self.phases[phase_index] = DemoPhase(
                        "lift_ball",
                        float(self.args.lift_retry_duration),
                        phase.end,
                        phase.end,
                    )
                    continue
            if phase.name == "release_ball":
                self._write_ball_attach(False, "release_ball")
            if phase.name.startswith("fine_align") and self.args.use_ball_anchor:
                self._wait_and_apply_anchor(
                    update_walk=False,
                    timeout=float(self.args.retrack_timeout),
                    fresh_after=t0,
                )
                workspace_error = self._workspace_error_xy()
                if workspace_error is not None:
                    error_x, error_y = workspace_error
                    x_ready = (
                        error_x <= float(self.args.align_x_tolerance)
                        and error_x >= -float(self.args.align_close_x_tolerance)
                    )
                    needs_retry = (
                        not x_ready
                        or abs(error_y) > float(self.args.align_y_tolerance)
                    )
                    if needs_retry:
                        retries = self._workspace_align_retries.get(phase.name, 0)
                        if retries < int(self.args.align_max_retries):
                            self._workspace_align_retries[phase.name] = retries + 1
                            _log(
                                f"workspace alignment residual: error=({error_x:.3f},{error_y:.3f})m "
                                f"retry {retries + 1}/{self.args.align_max_retries}"
                            )
                            phase = self.phases[phase_index]
                            self.phases[phase_index] = DemoPhase(
                                phase.name,
                                float(self.args.align_duration),
                                phase.start,
                                phase.end,
                            )
                            continue
                        x_soft_bad = (
                            error_x > float(self.args.align_x_soft_tolerance)
                            or error_x < -float(self.args.align_close_x_tolerance)
                        )
                        soft_bad = x_soft_bad or abs(error_y) > float(self.args.align_soft_tolerance)
                        if soft_bad:
                            _log(
                                f"workspace still offset after alignment: error=({error_x:.3f},{error_y:.3f})m; "
                                "continuing with contact servo residual compensation"
                            )
            if phase.name == "capture_ball_contact":
                err = self._last_contact_error_norm
                if (
                    self.args.contact_servo
                    and err is not None
                    and not self._last_contact_ready
                    and self._contact_capture_retries < int(self.args.capture_max_retries)
                ):
                    self._contact_capture_retries += 1
                    _log(
                        f"capture not ready: rms={err:.3f}m max={self._last_contact_max_error or math.nan:.3f}m "
                        f"retry {self._contact_capture_retries}/{self.args.capture_max_retries}"
                    )
                    retry_cmd = phase.end
                    self.phases[phase_index] = DemoPhase(
                        "capture_ball_contact",
                        float(self.args.capture_retry_duration),
                        retry_cmd,
                        retry_cmd,
                    )
                    continue
            if phase.name == "squeeze_ball_secure":
                err = self._last_contact_error_norm
                if (
                    self.args.contact_servo
                    and err is not None
                    and not self._last_contact_ready
                    and self._contact_squeeze_retries < int(self.args.squeeze_max_retries)
                ):
                    self._contact_squeeze_retries += 1
                    _log(
                        f"palm pocket not ready: rms={err:.3f}m max={self._last_contact_max_error or math.nan:.3f}m "
                        f"retry {self._contact_squeeze_retries}/{self.args.squeeze_max_retries}"
                    )
                    retry_cmd = phase.end
                    self.phases[phase_index] = DemoPhase(
                        "squeeze_ball_secure",
                        float(self.args.squeeze_retry_duration),
                        retry_cmd,
                        retry_cmd,
                    )
                    continue
            if phase.name == "walk_to_table" and self.args.use_ball_anchor:
                forward = self._current_ball_forward()
                target = float(self.args.approach_target_x) + float(self.args.approach_tolerance)
                if forward is not None and forward > target:
                    if approach_retries < int(self.args.max_approach_retries):
                        approach_retries += 1
                        _log(
                            f"approach still far: ball_x={forward:.2f}m target={target:.2f}m; "
                            f"retry {approach_retries}/{self.args.max_approach_retries}"
                        )
                        retry_cmd = phase.end
                        self.phases[phase_index] = DemoPhase(
                            "walk_to_table",
                            float(self.args.approach_retry_duration),
                            retry_cmd,
                            retry_cmd,
                        )
                        continue
                    soft_target = target + float(self.args.approach_soft_tolerance)
                    if forward <= soft_target:
                        _log(
                            f"approach close enough: ball_x={forward:.2f}m target={target:.2f}m "
                            f"soft_limit={soft_target:.2f}m; continuing"
                        )
                    elif self.args.require_ball_anchor:
                        raise RuntimeError(
                            f"approach failed: ball_x={forward:.2f}m remains beyond soft limit {soft_target:.2f}m"
                        )
            phase_index += 1

        done_msg = String()
        done_msg.data = "done"
        self.phase_pub.publish(done_msg)
        self._write_ball_attach(False, "done")
        if self.args.hold:
            _log("demo done; holding final single-hand place pose until Ctrl+C")
            final_cmd = self.phases[-1].end
            while rclpy.ok():
                self._publish_planner(final_cmd)
                self.phase_pub.publish(done_msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
        else:
            _log("demo done")

    def close(self) -> None:
        self._write_ball_attach(False, "shutdown")
        self.socket.close(0)

    def _anchor_cb(self, msg: String) -> None:
        try:
            self.latest_anchor = json.loads(msg.data)
            self._anchor_wall_time = time.monotonic()
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"bad ball anchor JSON: {exc}")

    def _current_ball_forward(self) -> float | None:
        ball_base = self._current_ball_base()
        if ball_base is None:
            return None
        return float(ball_base[0])

    def _current_ball_base(self) -> list[float] | None:
        if self.latest_anchor is None:
            return None
        if time.monotonic() - self._anchor_wall_time > float(self.args.anchor_fresh_age):
            return None
        return _finite_vec3(self.latest_anchor.get("ball_point_base"))

    def _current_ball_map(self) -> list[float] | None:
        if self.latest_anchor is None:
            return None
        if time.monotonic() - self._anchor_wall_time > float(self.args.anchor_fresh_age):
            return None
        return _finite_vec3(self.latest_anchor.get("ball_center_map"))

    def _workspace_error_y(self) -> float | None:
        ball_base = self._current_ball_base()
        if ball_base is None:
            return None
        return float(ball_base[1]) - float(self.args.align_target_y)

    def _workspace_error_xy(self) -> tuple[float, float] | None:
        ball_base = self._current_ball_base()
        if ball_base is None:
            return None
        return (
            float(ball_base[0]) - float(self.args.align_target_x),
            float(ball_base[1]) - float(self.args.align_target_y),
        )

    def _workspace_servo_command(self, phase_name: str, cmd: DemoCommand) -> DemoCommand:
        if not phase_name.startswith("fine_align") or not self.args.use_ball_anchor:
            return cmd
        error = self._workspace_error_xy()
        if error is None:
            return cmd
        error_x, error_y = error
        move_x = 0.0
        move_y = 0.0
        if error_x > float(self.args.align_x_tolerance):
            move_x = _clamp(
                float(self.args.align_forward_gain) * error_x,
                0.0,
                float(self.args.align_max_forward),
            )
        if abs(error_y) > float(self.args.align_y_tolerance):
            move_y = _clamp(
                -float(self.args.align_lateral_gain) * error_y,
                -float(self.args.align_max_lateral),
                float(self.args.align_max_lateral),
            )
        mode = LOCO_SLOW_WALK if max(abs(move_x), abs(move_y)) > 1e-3 else LOCO_IDLE
        now = time.monotonic()
        if now - self._last_workspace_servo_log > float(self.args.align_log_period):
            _log(
                f"workspace servo {phase_name}: "
                f"err=({error_x:.3f},{error_y:.3f}) "
                f"move=({move_x:.2f},{move_y:.2f}) "
                f"target=({self.args.align_target_x:.2f},{self.args.align_target_y:.2f})"
            )
            self._last_workspace_servo_log = now
        return DemoCommand(
            mode=mode,
            movement=(move_x, move_y, 0.0),
            facing=cmd.facing,
            speed=float(self.args.align_speed),
            height=cmd.height,
            upper_body=cmd.upper_body,
            left_hand=cmd.left_hand,
            right_hand=cmd.right_hand,
        )

    def _should_finish_walk(self, phase_name: str, elapsed: float) -> bool:
        if phase_name != "walk_to_table" or not self.args.use_ball_anchor:
            return False
        if elapsed < float(self.args.min_approach_duration):
            return False
        forward = self._current_ball_forward()
        if forward is None:
            return False
        target = float(self.args.approach_target_x) + float(self.args.approach_tolerance)
        if forward <= target:
            _log(
                f"approach reached: ball_x={forward:.2f}m "
                f"target={self.args.approach_target_x:.2f}m elapsed={elapsed:.1f}s"
            )
            return True
        return False

    def _should_finish_align(self, phase_name: str, elapsed: float) -> bool:
        if not phase_name.startswith("fine_align") or not self.args.use_ball_anchor:
            return False
        if elapsed < float(self.args.align_min_duration):
            return False
        error = self._workspace_error_xy()
        if error is None:
            return False
        error_x, error_y = error
        x_ready = (
            error_x <= float(self.args.align_x_tolerance)
            and error_x >= -float(self.args.align_close_x_tolerance)
        )
        if x_ready and abs(error_y) <= float(self.args.align_y_tolerance):
            _log(
                f"workspace aligned: error=({error_x:.3f},{error_y:.3f})m "
                f"target=({self.args.align_target_x:.2f},{self.args.align_target_y:.2f})m "
                f"elapsed={elapsed:.1f}s"
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
                ball_base = self.latest_anchor.get("ball_point_base", [math.nan, math.nan, math.nan])
                try:
                    x, y, z = (float(ball_base[i]) for i in range(3))
                except (TypeError, ValueError, IndexError):
                    x, y, z = math.nan, math.nan, math.nan
                _log(
                    "waiting for plausible ball anchor; "
                    f"latest ball_base=({x:.2f},{y:.2f},{z:.2f})"
                )
                reported_implausible = True
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.latest_anchor is None or (
            fresh_after is not None and self._anchor_wall_time <= fresh_after
        ) or not self._anchor_is_plausible(self.latest_anchor, update_walk=update_walk):
            if reported_implausible:
                _log(f"no plausible ball anchor received on {self.args.anchor_topic}; using fixed defaults")
            else:
                _log(f"no ball anchor received on {self.args.anchor_topic}; using fixed defaults")
            return False
        try:
            summary = apply_ball_anchor(self.args, self.latest_anchor, update_walk=update_walk)
            if not update_walk:
                self._update_ik_poses_from_anchor(self.latest_anchor)
            self.phases = make_demo_phases(self.args)
            self._servo_cache_cmd = None
            self._servo_cache_phase = None
            _log(f"using ball anchor: {summary}")
            return True
        except Exception as exc:
            self.get_logger().warn(f"failed to apply ball anchor; using fixed defaults: {exc}")
            return False

    def _anchor_is_plausible(self, anchor: dict | None, *, update_walk: bool) -> bool:
        if anchor is None or not update_walk:
            return True
        ball_base = _finite_vec3(anchor.get("ball_point_base"))
        if ball_base is None:
            return False
        x, y, z = ball_base
        return (
            float(self.args.initial_anchor_min_x) <= x <= float(self.args.initial_anchor_max_x)
            and abs(y) <= float(self.args.initial_anchor_max_abs_y)
            and float(self.args.initial_anchor_min_z) <= z <= float(self.args.initial_anchor_max_z)
        )

    def _retrack_anchor_for_pick(self) -> None:
        now = time.monotonic()
        if now - self._last_anchor_retrack < float(self.args.retrack_min_interval):
            return
        self._last_anchor_retrack = now
        self._wait_and_apply_anchor(
            update_walk=False,
            timeout=float(self.args.retrack_timeout),
            fresh_after=now - float(self.args.anchor_fresh_age),
        )

    def _adaptive_close_ratio(self, phase_name: str, ratio: float) -> float:
        err = self._last_contact_error_norm
        ready_err = float(self.args.servo_contact_ready_error)
        good_contact = err is not None and err <= ready_err
        if phase_name == "lower_to_ball_open":
            return 0.0
        if phase_name == "capture_ball_contact":
            base = _lerp(0.0, float(self.args.capture_close_ratio), _smoothstep(ratio))
            if good_contact:
                base = max(base, float(self.args.capture_close_ratio))
            return _clamp(base, 0.0, float(self.args.close_ratio))
        if phase_name == "close_on_ball":
            start = float(self.args.capture_close_ratio)
            target = float(self.args.close_ratio)
            if good_contact:
                target = max(target, float(self.args.preload_close_ratio))
            return _clamp(_lerp(start, target, _smoothstep(ratio)), 0.0, float(self.args.hold_close_ratio))
        if phase_name == "squeeze_ball_secure":
            start = float(self.args.preload_close_ratio if good_contact else self.args.close_ratio)
            target = float(self.args.hold_close_ratio if good_contact else self.args.preload_close_ratio)
            return _clamp(
                _lerp(start, target, _smoothstep(ratio)),
                0.0,
                float(self.args.max_hold_close_ratio),
            )
        if phase_name == "lift_ball":
            target = float(self.args.hold_close_ratio if self._ball_lifted_seen else self.args.preload_close_ratio)
            return _clamp(
                _lerp(float(self.args.preload_close_ratio), target, _smoothstep(ratio)),
                0.0,
                float(self.args.max_hold_close_ratio),
            )
        if phase_name in {"secure_ball", "move_to_place", "lower_to_place"}:
            return _clamp(float(self.args.hold_close_ratio), 0.0, float(self.args.max_hold_close_ratio))
        return _clamp(float(self.args.close_ratio), 0.0, float(self.args.max_hold_close_ratio))

    def _contact_servo_command(self, phase_name: str, ratio: float, cmd: DemoCommand) -> DemoCommand:
        active_phases = {
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
            "lift_ball",
            "secure_ball",
            "move_to_place",
            "lower_to_place",
        }
        if (
            not self.args.contact_servo
            or self.ik_solver is None
            or phase_name not in active_phases
        ):
            return cmd

        now = time.monotonic()
        if (
            self._servo_cache_cmd is not None
            and self._servo_cache_phase == phase_name
            and now - self._last_servo_time < 1.0 / max(1.0, float(self.args.servo_rate))
        ):
            return DemoCommand(
                mode=cmd.mode,
                movement=cmd.movement,
                facing=cmd.facing,
                speed=cmd.speed,
                height=cmd.height,
                upper_body=self._servo_cache_cmd.upper_body,
                left_hand=cmd.left_hand,
                right_hand=self._servo_cache_cmd.right_hand,
            )

        ball_base = self._current_ball_base()
        if ball_base is None or self.latest_anchor is None:
            return cmd
        radius = float(self.latest_anchor.get("ball_radius", self.args.ball_radius))
        ball = np.asarray(ball_base, dtype=np.float64)
        table_phase = phase_name in {
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
        }
        policy = ContactServoPolicy(
            ContactServoConfig(
                error_gain=float(self.args.servo_error_gain),
                hold_error_gain=float(self.args.servo_hold_error_gain),
                max_x_comp=float(self.args.servo_max_x_comp),
                max_y_comp=float(self.args.servo_max_y_comp),
                max_z_comp=float(self.args.servo_max_z_comp),
                table_down_radius=float(self.args.servo_table_down_radius),
                contact_ready_error=float(self.args.servo_contact_ready_error),
                capture_close_ratio=float(self.args.capture_close_ratio),
                close_ratio=float(self.args.close_ratio),
                preload_close_ratio=float(self.args.preload_close_ratio),
                hold_close_ratio=float(self.args.hold_close_ratio),
                max_hold_close_ratio=float(self.args.max_hold_close_ratio),
                lift_detect_z=float(self.args.lift_detect_z),
                lift_x_lead=float(self.args.servo_lift_x_lead),
                lift_z_lead=float(self.args.servo_lift_z_lead),
                lift_z_max_lead=float(self.args.servo_lift_z_max_lead),
                lift_ramp_start=float(self.args.servo_lift_ramp_start),
                hold_z_lead=float(self.args.servo_hold_z_lead),
                transfer_lead=float(self.args.servo_transfer_lead),
                place_lead=float(self.args.servo_place_lead),
                place_down_lead=float(self.args.servo_place_down_lead),
            )
        )

        effective_radius = radius
        center = ball.copy()
        if phase_name == "lower_to_ball_open":
            effective_radius = radius + float(self.args.open_shell_radius)
            center = center + np.asarray([0.0, 0.0, float(self.args.open_shell_z)], dtype=np.float64)
        elif phase_name == "capture_ball_contact":
            shell = 1.0 - _smoothstep(ratio)
            effective_radius = radius + float(self.args.capture_shell_radius) * shell
            center = center + np.asarray([0.0, 0.0, float(self.args.capture_shell_z) * shell], dtype=np.float64)
        mean_error = None
        palm_error = None
        quality: GraspQuality | None = None
        error_info = self.ik_solver.contact_error_base(
            center,
            effective_radius,
            table_contact=table_phase,
            args=self.args,
        )
        if error_info is not None:
            mean_error, _per_contact = error_info
            palm_error = _per_contact.get("palm")
            finger_contact_errors = {
                name: error
                for name, error in _per_contact.items()
                if name in {"thumb", "middle", "index"}
            }
            quality = GraspQuality.from_contact_errors(
                finger_contact_errors,
                ready_error=float(self.args.servo_contact_ready_error),
                max_ready_error=float(self.args.servo_contact_max_error),
                required_names=("thumb", "middle", "index"),
            )
            if quality is not None:
                self._last_contact_error_norm = float(quality.rms_error)
                self._last_contact_max_error = float(quality.max_error)
                self._last_contact_ready = bool(quality.ready)
        servo_bias = policy.contact_bias(quality, radius=effective_radius, table_contact=table_phase)
        servo_bias[0] = max(servo_bias[0], -float(self.args.servo_max_backward_x_comp))
        if table_phase:
            self._contact_center_bias = servo_bias

        if phase_name in {"lift_ball", "secure_ball"}:
            ball_map = self._current_ball_map()
            lift_delta = 0.0
            if ball_map is not None and self._lift_start_ball_map_z is not None:
                lift_delta = float(ball_map[2]) - float(self._lift_start_ball_map_z)
            lifted = lift_delta >= float(self.args.lift_detect_z)
            self._ball_lifted_seen = self._ball_lifted_seen or lifted
            center = ball + policy.lift_lead(ratio, lifted=self._ball_lifted_seen)
            center = center + self._contact_center_bias * max(0.0, 1.0 - _smoothstep(ratio))
            center = center + servo_bias
        elif phase_name in {"move_to_place", "lower_to_place"}:
            place_base = _finite_vec3(self.latest_anchor.get("place_point_base"))
            if place_base is not None:
                lead = policy.transport_lead(
                    ball,
                    np.asarray(place_base, dtype=np.float64),
                    lowering=phase_name == "lower_to_place",
                )
            else:
                lead = np.zeros(3, dtype=np.float64)
            center = ball + lead + servo_bias
        else:
            center = center + servo_bias

        adaptive_close_ratio = policy.close_ratio(
            phase_name,
            ratio,
            quality,
            lifted=self._ball_lifted_seen,
        )
        _, adaptive_right_hand = _closed_hands(adaptive_close_ratio)

        seed = self.ik_solver.live_upper_body_pose(cmd.upper_body)
        try:
            solved, ik_error = self.ik_solver.solve_contact_pose(
                center,
                effective_radius,
                seed,
                self.args,
                hand_pose=adaptive_right_hand,
                table_contact=table_phase,
            )
        except Exception as exc:
            if now - self._last_servo_log > float(self.args.servo_log_period):
                self.get_logger().warn(f"contact servo failed in {phase_name}: {exc}")
                self._last_servo_log = now
            return cmd

        if ik_error > float(self.args.servo_ik_max_error):
            if now - self._last_servo_log > float(self.args.servo_log_period):
                self.get_logger().warn(
                    f"contact servo rejected in {phase_name}: ik_error={ik_error:.3f}"
                )
                self._last_servo_log = now
            return cmd

        out = DemoCommand(
            mode=cmd.mode,
            movement=cmd.movement,
            facing=cmd.facing,
            speed=cmd.speed,
            height=cmd.height,
            upper_body=solved,
            left_hand=cmd.left_hand,
            right_hand=adaptive_right_hand,
        )
        self._servo_cache_cmd = out
        self._servo_cache_phase = phase_name
        self._last_servo_time = now

        if now - self._last_servo_log > float(self.args.servo_log_period):
            if mean_error is None:
                err_text = "err=(nan,nan,nan)"
            else:
                err_text = f"err=({mean_error[0]:.3f},{mean_error[1]:.3f},{mean_error[2]:.3f})"
            if palm_error is None:
                palm_text = "palm=(nan,nan,nan)"
            else:
                palm_text = f"palm=({palm_error[0]:.3f},{palm_error[1]:.3f},{palm_error[2]:.3f})"
            if quality is None:
                quality_text = "rms=nan max=nan ready=0"
            else:
                quality_text = (
                    f"rms={quality.rms_error:.3f} max={quality.max_error:.3f} "
                    f"ready={int(quality.ready)}"
                )
            wrist_target = (
                solved[FULL_INDEX["right_wrist_roll_joint"]],
                solved[FULL_INDEX["right_wrist_pitch_joint"]],
                solved[FULL_INDEX["right_wrist_yaw_joint"]],
            )
            _log(
                f"contact servo {phase_name}: {err_text} {palm_text} {quality_text} "
                f"center_delta=({center[0] - ball[0]:.3f},{center[1] - ball[1]:.3f},{center[2] - ball[2]:.3f}) "
                f"close={adaptive_close_ratio:.2f} "
                f"ik={ik_error:.3f} "
                f"wrist_target=({wrist_target[0]:.2f},{wrist_target[1]:.2f},{wrist_target[2]:.2f})"
            )
            self._last_servo_log = now
        return out

    def _mark_lift_reference(self) -> None:
        ball_map = self._current_ball_map()
        if ball_map is None:
            self._ball_pre_lift_z = None
            self._lift_start_ball_map_z = None
            return
        self._ball_pre_lift_z = float(ball_map[2])
        self._lift_start_ball_map_z = float(ball_map[2])
        self._ball_lifted_seen = False
        _log(f"lift reference: ball_map_z={self._ball_pre_lift_z:.3f}m")
        self._log_grasp_geometry()

    def _log_grasp_geometry(self) -> None:
        if self.ik_solver is None:
            return
        ball_base = self._current_ball_base()
        if ball_base is None:
            return
        try:
            qpos = load_npy_if_ready(self.args.qpos_path)
        except OSError:
            qpos = None
        if qpos is None:
            return

        model = self.ik_solver.model
        data = self.ik_solver.data
        n = min(len(qpos), model.nq)
        data.qpos[:n] = qpos[:n]
        mujoco.mj_forward(model, data)
        base_pos = data.xpos[self.ik_solver.base_body_id]
        base_rot = data.xmat[self.ik_solver.base_body_id].reshape(3, 3)
        ball = np.asarray(ball_base, dtype=np.float64)
        samples = [
            ("palm", self.ik_solver.palm_body_id, PALM_CONTACT_LOCAL),
            ("thumb", self.ik_solver.finger_body_ids["thumb"], FINGER_CONTACT_LOCAL["thumb"]),
            ("middle", self.ik_solver.finger_body_ids["middle"], FINGER_CONTACT_LOCAL["middle"]),
            ("index", self.ik_solver.finger_body_ids["index"], FINGER_CONTACT_LOCAL["index"]),
        ]
        pieces = []
        for name, body_id, local_point in samples:
            point_world = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point
            point_base = base_rot.T @ (point_world - base_pos)
            rel = point_base - ball
            pieces.append(f"{name}=({rel[0]:.3f},{rel[1]:.3f},{rel[2]:.3f})")
        hand_values = []
        for joint_name in RIGHT_HAND_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            hand_values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        wrist_values = []
        for joint_name in [
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            wrist_values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        _log(
            "grasp geometry rel_to_ball_base: "
            + " ".join(pieces)
            + " hand_q="
            + ",".join(f"{value:.2f}" for value in hand_values)
            + " wrist_q="
            + ",".join(f"{value:.2f}" for value in wrist_values)
        )

    def _wait_for_lifted_ball(self, *, timeout: float) -> bool:
        if self._ball_pre_lift_z is None:
            self._mark_lift_reference()
        if self._ball_pre_lift_z is None:
            _log("ball lift check skipped: no fresh anchor")
            return False

        threshold = float(self.args.lift_detect_z)
        deadline = time.monotonic() + max(0.0, timeout)
        best_delta = -math.inf
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.03)
            ball_map = self._current_ball_map()
            if ball_map is None:
                continue
            delta = float(ball_map[2]) - float(self._ball_pre_lift_z)
            best_delta = max(best_delta, delta)
            if delta >= threshold:
                _log(f"ball lifted: dz={delta:.3f}m")
                return True
            time.sleep(0.03)
        if math.isfinite(best_delta):
            _log(f"ball not lifted yet: best_dz={best_delta:.3f}m")
        else:
            _log("ball lift check had no fresh samples")
        return False

    def _update_ik_poses_from_anchor(self, anchor: dict) -> None:
        if self.ik_solver is None:
            return
        previous_overrides = getattr(self.args, "ik_pose_overrides", {})
        self.args.ik_pose_overrides = {}
        fallback_poses = _upper_body_poses(self.args)
        self.args.ik_pose_overrides = previous_overrides
        poses, errors = self.ik_solver.solve_pick_place_poses(anchor, fallback_poses, self.args)
        critical_errors = {
            name: err for name, err in errors.items() if name in {"grasp", "lift", "secure"}
        }
        worst_error = max(critical_errors.values()) if critical_errors else math.inf
        if worst_error > float(self.args.ik_max_error):
            message = " ".join(f"{name}={err:.3f}" for name, err in errors.items())
            if self.args.require_ik:
                raise RuntimeError(f"IK error too large: {message}")
            self.get_logger().warn(f"IK result rejected: {message}")
            return
        self.args.ik_pose_overrides = poses
        message = " ".join(f"{name}={err:.3f}" for name, err in errors.items())
        _log(f"IK right-hand poses applied: {message}")

    def _send_start(self) -> None:
        start_msg = build_command_message(start=True, stop=False, planner=True)
        idle = self.phases[0].start
        for _ in range(max(1, int(self.args.start_bursts))):
            self.socket.send(start_msg)
            self._publish_planner(idle)
            time.sleep(0.12)
        _log("sent start command: planner mode with right-hand WBC targets")

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

    def _ball_offset_for_phase(self, phase_name: str, ratio: float) -> tuple[float, float, float]:
        pick = (float(self.args.assist_x), float(self.args.assist_y), float(self.args.assist_z))
        lift = (
            float(self.args.assist_x) + float(self.args.attach_lift_x_delta),
            float(self.args.assist_y),
            float(self.args.assist_z) + float(self.args.attach_lift_z),
        )
        place_low = (
            float(self.args.place_assist_x),
            float(self.args.place_assist_y),
            float(self.args.place_assist_z),
        )
        place_high = (
            float(self.args.place_assist_x) + float(self.args.attach_place_x_delta),
            float(self.args.place_assist_y),
            float(self.args.place_assist_z) + float(self.args.attach_lift_z),
        )
        if phase_name == "close_on_ball":
            return pick
        if phase_name == "lift_ball":
            s = _smoothstep(ratio)
            return tuple(_lerp(a, b, s) for a, b in zip(pick, lift))
        if phase_name == "secure_ball":
            return lift
        if phase_name == "move_to_place":
            s = _smoothstep(ratio)
            return tuple(_lerp(a, b, s) for a, b in zip(lift, place_high))
        if phase_name == "lower_to_place":
            s = _smoothstep(ratio)
            return tuple(_lerp(a, b, s) for a, b in zip(place_high, place_low))
        return lift

    def _update_ball_attach(self, phase_name: str, ratio: float) -> None:
        active = phase_name in {
            "lift_ball",
            "secure_ball",
            "move_to_place",
            "lower_to_place",
        }
        if phase_name == "close_on_ball":
            active = ratio >= float(self.args.attach_engage_ratio)
        if not active:
            self._write_ball_attach(False, phase_name)
            return
        self._write_ball_attach(True, phase_name, self._ball_offset_for_phase(phase_name, ratio))

    def _write_ball_attach(
        self,
        enabled: bool,
        phase_name: str,
        offset: tuple[float, float, float] | None = None,
    ) -> None:
        if offset is None:
            offset = (float(self.args.assist_x), float(self.args.assist_y), float(self.args.assist_z))
        active = bool(enabled and self.args.ball_attach)
        payload = {
            "enabled": active,
            "phase": phase_name,
            "stamp": time.time(),
            "box_enabled": active,
            "object_enabled": active,
            "object_joint": "demo_ball_freejoint",
            "object_body": "demo_ball",
            "object_geom": "demo_ball_visual",
            "local_offset": [float(v) for v in offset],
            "blend": float(self.args.ball_attach_blend),
        }
        tmp_path = f"{GRASP_ASSIST_FILE}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, GRASP_ASSIST_FILE)
        except OSError as exc:
            self.get_logger().warn(f"failed to write ball attach file {GRASP_ASSIST_FILE}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 tabletop ball pick-place demo over zmq_manager: lower-body SONIC, right-hand WBC targets."
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--zmq-bind-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-connect-wait", type=float, default=1.0)
    parser.add_argument("--start-bursts", type=int, default=8)
    parser.add_argument("--max-upper-body-velocity", type=float, default=1.8)
    parser.add_argument("--scene", default="ball_demo")
    parser.add_argument("--qpos-path", default="/tmp/sonic_qpos.npy")

    parser.add_argument("--stand-height", type=float, default=0.78)
    parser.add_argument("--walk-speed", type=float, default=0.24)
    parser.add_argument("--walk-duration", type=float, default=2.0)
    parser.add_argument("--walk-extra-duration", type=float, default=2.8)
    parser.add_argument("--max-approach-duration", type=float, default=9.0)
    parser.add_argument("--min-approach-duration", type=float, default=1.0)
    parser.add_argument("--approach-target-x", type=float, default=0.56)
    parser.add_argument("--approach-tolerance", type=float, default=0.035)
    parser.add_argument("--approach-soft-tolerance", type=float, default=0.08)
    parser.add_argument("--max-approach-retries", type=int, default=8)
    parser.add_argument("--approach-retry-duration", type=float, default=0.65)
    parser.add_argument("--align-target-x", type=float, default=0.54)
    parser.add_argument("--align-x-tolerance", type=float, default=0.045)
    parser.add_argument("--align-close-x-tolerance", type=float, default=0.16)
    parser.add_argument("--align-x-soft-tolerance", type=float, default=0.10)
    parser.add_argument("--align-forward-gain", type=float, default=4.2)
    parser.add_argument("--align-max-forward", type=float, default=0.90)
    parser.add_argument("--align-target-y", type=float, default=-0.24)
    parser.add_argument("--align-y-tolerance", type=float, default=0.035)
    parser.add_argument("--align-lateral-gain", type=float, default=5.0)
    parser.add_argument("--align-max-lateral", type=float, default=0.85)
    parser.add_argument("--align-speed", type=float, default=0.10)
    parser.add_argument("--align-min-duration", type=float, default=0.15)
    parser.add_argument("--align-max-duration", type=float, default=5.2)
    parser.add_argument("--align-duration-gain", type=float, default=2.2)
    parser.add_argument("--align-duration", type=float, default=0.15)
    parser.add_argument("--align-movement-x", type=float, default=0.0)
    parser.add_argument("--align-movement-y", type=float, default=0.0)
    parser.add_argument("--align-soft-tolerance", type=float, default=0.075)
    parser.add_argument("--align-max-retries", type=int, default=4)
    parser.add_argument("--align-log-period", type=float, default=0.6)

    parser.add_argument("--settle-duration", type=float, default=0.8)
    parser.add_argument("--clear-hand-duration", type=float, default=1.0)
    parser.add_argument("--prepare-duration", type=float, default=1.2)
    parser.add_argument("--pregrasp-duration", type=float, default=1.0)
    parser.add_argument("--reach-duration", type=float, default=1.2)
    parser.add_argument("--capture-duration", type=float, default=1.4)
    parser.add_argument("--close-duration", type=float, default=1.5)
    parser.add_argument("--palm-squeeze-duration", type=float, default=0.8)
    parser.add_argument("--lift-duration", type=float, default=1.6)
    parser.add_argument("--lift-retry-duration", type=float, default=1.2)
    parser.add_argument("--lift-max-retries", type=int, default=2)
    parser.add_argument("--secure-duration", type=float, default=1.0)
    parser.add_argument("--transfer-duration", type=float, default=1.8)
    parser.add_argument("--place-duration", type=float, default=1.0)
    parser.add_argument("--release-duration", type=float, default=0.7)
    parser.add_argument("--retreat-duration", type=float, default=1.0)
    parser.add_argument("--hold-duration", type=float, default=2.0)
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--no-hold", dest="hold", action="store_false")

    parser.add_argument("--ready-pitch", type=float, default=-0.88)
    parser.add_argument("--ready-roll", type=float, default=-0.32)
    parser.add_argument("--ready-yaw", type=float, default=0.16)
    parser.add_argument("--ready-elbow", type=float, default=0.58)
    parser.add_argument("--ready-wrist-pitch", type=float, default=0.08)
    parser.add_argument("--reach-waist-pitch", type=float, default=0.05)
    parser.add_argument("--reach-pitch", type=float, default=-0.62)
    parser.add_argument("--reach-roll", type=float, default=-0.45)
    parser.add_argument("--reach-yaw", type=float, default=0.12)
    parser.add_argument("--reach-elbow", type=float, default=0.54)
    parser.add_argument("--reach-wrist-pitch", type=float, default=-0.18)
    parser.add_argument("--grasp-pitch", type=float, default=-0.66)
    parser.add_argument("--grasp-roll", type=float, default=-0.48)
    parser.add_argument("--grasp-yaw", type=float, default=0.10)
    parser.add_argument("--grasp-elbow", type=float, default=0.50)
    parser.add_argument("--grasp-wrist-pitch", type=float, default=-0.24)
    parser.add_argument("--lift-pitch", type=float, default=-0.50)
    parser.add_argument("--lift-roll", type=float, default=-0.38)
    parser.add_argument("--lift-yaw", type=float, default=0.08)
    parser.add_argument("--lift-elbow", type=float, default=0.66)
    parser.add_argument("--lift-wrist-pitch", type=float, default=-0.08)
    parser.add_argument("--place-pitch", type=float, default=-0.54)
    parser.add_argument("--place-roll", type=float, default=-0.34)
    parser.add_argument("--place-yaw", type=float, default=0.16)
    parser.add_argument("--place-elbow", type=float, default=0.64)
    parser.add_argument("--place-wrist-pitch", type=float, default=-0.12)
    parser.add_argument("--place-low-pitch", type=float, default=-0.62)
    parser.add_argument("--place-low-roll", type=float, default=-0.38)
    parser.add_argument("--place-low-elbow", type=float, default=0.56)
    parser.add_argument("--place-low-wrist-pitch", type=float, default=-0.22)
    parser.add_argument("--retreat-pitch", type=float, default=-0.28)
    parser.add_argument("--retreat-roll", type=float, default=-0.32)
    parser.add_argument("--retreat-yaw", type=float, default=0.08)
    parser.add_argument("--retreat-elbow", type=float, default=0.72)
    parser.add_argument("--retreat-wrist-pitch", type=float, default=-0.08)
    parser.add_argument("--capture-close-ratio", type=float, default=0.28)
    parser.add_argument("--close-ratio", type=float, default=0.50)
    parser.add_argument("--preload-close-ratio", type=float, default=0.70)
    parser.add_argument("--squeeze-close-ratio", type=float, default=0.65)
    parser.add_argument("--hold-close-ratio", type=float, default=0.76)
    parser.add_argument("--max-hold-close-ratio", type=float, default=0.85)
    parser.add_argument("--auto-ball-grasp", action="store_true", default=True)
    parser.add_argument("--no-auto-ball-grasp", dest="auto_ball_grasp", action="store_false")
    parser.add_argument("--finger-close-intercept", type=float, default=0.94)
    parser.add_argument("--finger-close-radius-gain", type=float, default=8.0)
    parser.add_argument("--finger-close-min", type=float, default=0.38)
    parser.add_argument("--finger-close-max", type=float, default=0.74)
    parser.add_argument("--capture-open-delta", type=float, default=0.18)
    parser.add_argument("--capture-close-min", type=float, default=0.18)
    parser.add_argument("--preload-close-extra", type=float, default=0.10)
    parser.add_argument("--squeeze-close-extra", type=float, default=0.17)
    parser.add_argument("--hold-close-extra", type=float, default=0.22)

    parser.add_argument("--ball-attach", action="store_true", default=False)
    parser.add_argument("--no-ball-attach", dest="ball_attach", action="store_false")
    parser.add_argument("--ball-attach-blend", type=float, default=1.0)
    parser.add_argument("--attach-engage-ratio", type=float, default=0.62)
    parser.add_argument("--assist-x", type=float, default=0.46)
    parser.add_argument("--assist-y", type=float, default=-0.14)
    parser.add_argument("--assist-z", type=float, default=0.04)
    parser.add_argument("--place-assist-x", type=float, default=0.46)
    parser.add_argument("--place-assist-y", type=float, default=0.04)
    parser.add_argument("--place-assist-z", type=float, default=0.04)
    parser.add_argument("--attach-lift-x-delta", type=float, default=-0.02)
    parser.add_argument("--attach-lift-z", type=float, default=0.18)
    parser.add_argument("--attach-place-x-delta", type=float, default=-0.02)

    parser.add_argument("--use-ball-anchor", action="store_true")
    parser.add_argument("--require-ball-anchor", action="store_true")
    parser.add_argument("--anchor-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--anchor-timeout", type=float, default=8.0)
    parser.add_argument("--anchor-fresh-age", type=float, default=0.8)
    parser.add_argument("--initial-anchor-min-x", type=float, default=0.30)
    parser.add_argument("--initial-anchor-max-x", type=float, default=2.50)
    parser.add_argument("--initial-anchor-max-abs-y", type=float, default=0.75)
    parser.add_argument("--initial-anchor-min-z", type=float, default=-0.35)
    parser.add_argument("--initial-anchor-max-z", type=float, default=0.22)
    parser.add_argument("--retrack-before-pick", action="store_true", default=True)
    parser.add_argument("--no-retrack-before-pick", dest="retrack_before_pick", action="store_false")
    parser.add_argument("--retrack-timeout", type=float, default=0.35)
    parser.add_argument("--retrack-min-interval", type=float, default=0.45)
    parser.add_argument("--lift-detect-z", type=float, default=0.030)
    parser.add_argument("--lift-detect-timeout", type=float, default=0.8)

    parser.add_argument("--ik-upper-body", action="store_true", default=True)
    parser.add_argument("--no-ik-upper-body", dest="ik_upper_body", action="store_false")
    parser.add_argument("--require-ik", action="store_true")
    parser.add_argument("--simple-clamp-grasp", action="store_true", default=False)
    parser.add_argument("--contact-template-grasp", dest="simple_clamp_grasp", action="store_false")
    parser.add_argument("--simple-tip-x-offset", type=float, default=0.005)
    parser.add_argument("--simple-tip-y-offset", type=float, default=-0.010)
    parser.add_argument("--simple-tip-z-offset", type=float, default=-0.005)
    parser.add_argument("--simple-tip-x-radius-scale", type=float, default=0.11)
    parser.add_argument("--simple-tip-y-radius-scale", type=float, default=-0.22)
    parser.add_argument("--simple-tip-z-radius-scale", type=float, default=-0.75)
    parser.add_argument("--ik-hand-body", default="right_hand_middle_1_link")
    parser.add_argument("--ik-x-offset", type=float, default=0.050)
    parser.add_argument("--ik-y-offset", type=float, default=-0.030)
    parser.add_argument("--auto-side-offset", action="store_true", default=True)
    parser.add_argument("--no-auto-side-offset", dest="auto_side_offset", action="store_false")
    parser.add_argument("--ik-side-radius-scale", type=float, default=0.45)
    parser.add_argument("--ik-z-offset", type=float, default=-0.115)
    parser.add_argument("--ik-hover-x-delta", type=float, default=-0.12)
    parser.add_argument("--ik-hover-y-delta", type=float, default=-0.06)
    parser.add_argument("--ik-hover-z", type=float, default=0.30)
    parser.add_argument("--ik-pregrasp-x-delta", type=float, default=-0.055)
    parser.add_argument("--ik-pregrasp-z", type=float, default=0.18)
    parser.add_argument("--ik-lift-x-delta", type=float, default=0.015)
    parser.add_argument("--ik-lift-z", type=float, default=0.18)
    parser.add_argument("--ik-secure-z-extra", type=float, default=0.015)
    parser.add_argument("--ik-place-x-offset", type=float, default=0.002)
    parser.add_argument("--ik-place-y-offset", type=float, default=-0.028)
    parser.add_argument("--ik-place-z-offset", type=float, default=-0.035)
    parser.add_argument("--ik-place-hover-x-delta", type=float, default=-0.03)
    parser.add_argument("--ik-place-hover-z", type=float, default=0.17)
    parser.add_argument("--ik-retreat-x-delta", type=float, default=-0.16)
    parser.add_argument("--ik-retreat-y-delta", type=float, default=-0.08)
    parser.add_argument("--ik-retreat-z", type=float, default=0.20)
    parser.add_argument("--ik-iters", type=int, default=90)
    parser.add_argument("--ik-damping", type=float, default=0.045)
    parser.add_argument("--ik-regularization", type=float, default=0.012)
    parser.add_argument("--ik-step-limit", type=float, default=0.12)
    parser.add_argument("--ik-max-error", type=float, default=0.220)
    parser.add_argument("--ik-wrist-roll-limit", type=float, default=0.24)
    parser.add_argument("--ik-wrist-pitch-min", type=float, default=-0.28)
    parser.add_argument("--ik-wrist-pitch-max", type=float, default=0.16)
    parser.add_argument("--ik-wrist-yaw-limit", type=float, default=0.16)
    parser.add_argument("--ball-radius", type=float, default=0.045)
    parser.add_argument("--contact-servo", action="store_true", default=True)
    parser.add_argument("--no-contact-servo", dest="contact_servo", action="store_false")
    parser.add_argument("--servo-rate", type=float, default=8.0)
    parser.add_argument("--servo-error-gain", type=float, default=0.85)
    parser.add_argument("--servo-hold-error-gain", type=float, default=0.60)
    parser.add_argument("--servo-max-x-comp", type=float, default=0.09)
    parser.add_argument("--servo-max-backward-x-comp", type=float, default=0.025)
    parser.add_argument("--servo-max-y-comp", type=float, default=0.14)
    parser.add_argument("--servo-max-z-comp", type=float, default=0.07)
    parser.add_argument("--servo-table-down-radius", type=float, default=0.35)
    parser.add_argument("--servo-lift-x-lead", type=float, default=0.025)
    parser.add_argument("--servo-lift-z-lead", type=float, default=0.050)
    parser.add_argument("--servo-lift-z-max-lead", type=float, default=0.115)
    parser.add_argument("--servo-lift-ramp-start", type=float, default=0.25)
    parser.add_argument("--servo-hold-z-lead", type=float, default=0.016)
    parser.add_argument("--servo-transfer-lead", type=float, default=0.045)
    parser.add_argument("--servo-place-lead", type=float, default=0.035)
    parser.add_argument("--servo-place-down-lead", type=float, default=0.018)
    parser.add_argument("--servo-contact-ready-error", type=float, default=0.035)
    parser.add_argument("--servo-ik-iters", type=int, default=55)
    parser.add_argument("--servo-ik-regularization", type=float, default=0.006)
    parser.add_argument("--servo-ik-max-error", type=float, default=0.360)
    parser.add_argument("--servo-log-period", type=float, default=0.6)
    parser.add_argument("--open-shell-radius", type=float, default=0.035)
    parser.add_argument("--open-shell-z", type=float, default=0.070)
    parser.add_argument("--capture-shell-radius", type=float, default=0.025)
    parser.add_argument("--capture-shell-z", type=float, default=0.040)
    parser.add_argument("--capture-max-retries", type=int, default=2)
    parser.add_argument("--capture-retry-duration", type=float, default=0.8)
    parser.add_argument("--squeeze-max-retries", type=int, default=2)
    parser.add_argument("--squeeze-retry-duration", type=float, default=0.65)
    parser.add_argument("--servo-contact-max-error", type=float, default=0.055)
    parser.add_argument("--palm-pocket-x-radius", type=float, default=-1.85)
    parser.add_argument("--palm-pocket-y-radius", type=float, default=-0.10)
    parser.add_argument("--palm-pocket-table-z-radius", type=float, default=0.00)
    parser.add_argument("--palm-pocket-lift-z-radius", type=float, default=0.02)
    parser.add_argument("--palm-contact-weight", type=float, default=0.35)
    parser.add_argument("--palm-frame-contact-targets", action="store_true", default=True)
    parser.add_argument("--base-frame-contact-targets", dest="palm_frame_contact_targets", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_ball_scaled_grasp(args, float(args.ball_radius))
    rclpy.init()
    node = BallPickPlaceDemo(args)
    try:
        node.run()
    except KeyboardInterrupt:
        _log("stopping ball pick-place demo")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
