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
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = os.path.dirname(SCRIPTS_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

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
from sonic_world.planners import TaskPlanner, TaskRequest  # noqa: E402
from sonic_world.rollout_logging import add_rollout_log_args, logger_from_args  # noqa: E402
from sonic_world.skills import runtime_plan_for_graph, skill_summary  # noqa: E402
from sonic_world.world_model import anchor_to_world  # noqa: E402


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


def _anchor_stamp_wall_time(anchor: dict) -> float:
    stamp = anchor.get("stamp") if isinstance(anchor, dict) else None
    if not isinstance(stamp, dict):
        return time.time()
    try:
        sec = float(stamp.get("sec", 0.0))
        nanosec = float(stamp.get("nanosec", 0.0))
    except (TypeError, ValueError):
        return time.time()
    if sec <= 0.0:
        return time.time()
    return sec + nanosec * 1e-9


def _signed_axis_command(error: float, *, tolerance: float, gain: float, max_abs: float, response_sign: float) -> float:
    if abs(float(error)) <= float(tolerance):
        return 0.0
    sign = -1.0 if float(response_sign) < 0.0 else 1.0
    return _clamp(-sign * float(gain) * float(error), -float(max_abs), float(max_abs))


def _load_policy_action(raw: str | None, *, task_id: str | None = None) -> dict | None:
    if not raw:
        return None
    text = raw
    path = raw if isinstance(raw, str) else ""
    if path and os.path.exists(os.path.expanduser(path)):
        with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
            text = handle.read()
    candidates: list[dict] = []
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        if isinstance(payload, list):
            candidates.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            candidates.append(payload)
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                candidates.append(payload)
    if not candidates:
        return None

    scored: list[tuple[int, dict]] = []
    for candidate in candidates:
        for action, base_score in _candidate_policy_actions(candidate, task_id=task_id):
            intent = action.get("task_intent") if isinstance(action.get("task_intent"), dict) else {}
            score = int(base_score)
            if task_id and str(action.get("task_id") or "") == str(task_id):
                score += 100
            if str(intent.get("demo_kind") or "") == "ball":
                score += 20
            if "single_hand_pinch" in [str(item) for item in action.get("skill_selection") or []]:
                score += 5
            scored.append((score, action))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_policy_action(payload: dict) -> dict | None:
    if isinstance(payload.get("action"), dict):
        return payload["action"]
    if isinstance(payload.get("policy_action"), dict):
        action = payload["policy_action"]
        if isinstance(action.get("action"), dict):
            return action["action"]
        return action
    if isinstance(payload.get("task_intent"), dict) and "skill_selection" in payload:
        return payload
    return None


def _candidate_policy_actions(payload: dict, *, task_id: str | None = None) -> list[tuple[dict, int]]:
    if payload.get("schema") == "sonic_task_policy_memory_v0":
        return _policy_memory_actions(payload, task_id=task_id)
    action = _extract_policy_action(payload)
    return [(action, 0)] if action is not None else []


def _policy_memory_actions(model: dict, *, task_id: str | None = None) -> list[tuple[dict, int]]:
    exact = model.get("exact_task_policy") if isinstance(model.get("exact_task_policy"), dict) else {}
    out: list[tuple[dict, int]] = []
    if task_id and isinstance(exact.get(str(task_id)), dict):
        action = exact[str(task_id)].get("recommended_action")
        if isinstance(action, dict):
            return [(action, 160)]
    for entry in exact.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("demo_kind") or "") != "ball":
            continue
        action = entry.get("recommended_action")
        if isinstance(action, dict):
            score = 20 + int(float(entry.get("best_dense_score") or 0.0) * 20.0)
            out.append((action, score))
    fallback = model.get("fallback_policy") if isinstance(model.get("fallback_policy"), dict) else {}
    for key, entry in fallback.items():
        if not isinstance(entry, dict) or not str(key).startswith("ball|"):
            continue
        action = entry.get("recommended_action")
        if isinstance(action, dict):
            score = 10 + int(float(entry.get("best_dense_score") or 0.0) * 10.0)
            out.append((action, score))
    return out


def _policy_feedback_deltas(action: dict | None) -> dict[str, float]:
    if not isinstance(action, dict):
        return {}
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    feedback = metadata.get("feedback_policy") if isinstance(metadata.get("feedback_policy"), dict) else {}
    modes = feedback.get("applied_modes") if isinstance(feedback.get("applied_modes"), list) else []
    deltas: dict[str, float] = {}
    for mode in modes:
        if not isinstance(mode, dict):
            continue
        for item in mode.get("fields") or []:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "")
            try:
                old = float(item.get("old"))
                new = float(item.get("new"))
            except (TypeError, ValueError):
                continue
            deltas[field] = deltas.get(field, 0.0) + (new - old)
    return deltas


def _apply_policy_navigation(args: argparse.Namespace) -> list[str]:
    action = getattr(args, "policy_action_payload", None)
    if not isinstance(action, dict) or args.policy_action_apply == "off":
        return []
    if args.policy_action_apply == "safe" and not bool(args.policy_action_safe_standoff):
        return []
    notes: list[str] = []
    base_goal = action.get("base_goal") if isinstance(action.get("base_goal"), dict) else {}
    standoff = base_goal.get("standoff")
    if standoff is None:
        return notes
    try:
        raw_target = float(standoff)
    except (TypeError, ValueError):
        return notes
    max_delta = max(0.0, float(args.policy_action_max_standoff_delta))
    old = float(args.approach_target_x)
    target = _clamp(raw_target, old - max_delta, old + max_delta)
    target = _clamp(target, 0.38, 0.70)
    args.approach_target_x = target
    args.align_target_x = target
    notes.append(f"standoff {old:.3f}->{target:.3f}")
    return notes


def _apply_policy_manipulation(args: argparse.Namespace, anchor: dict) -> list[str]:
    action = getattr(args, "policy_action_payload", None)
    if not isinstance(action, dict) or args.policy_action_apply == "off":
        return []
    notes: list[str] = []
    deltas = _policy_feedback_deltas(action)
    max_close = max(0.0, float(args.policy_action_max_close_delta))
    close_delta = _clamp(
        float(deltas.get("grasp_close_ratio.close_ratio", 0.0)),
        -max_close,
        max_close,
    )
    if abs(close_delta) > 1e-5:
        old = float(args.close_ratio)
        args.close_ratio = _clamp(old + close_delta, 0.18, float(args.max_hold_close_ratio))
        args.capture_close_ratio = _clamp(
            float(args.capture_close_ratio) + 0.5 * close_delta,
            float(args.capture_close_min),
            args.close_ratio,
        )
        args.preload_close_ratio = _clamp(
            float(args.preload_close_ratio) + close_delta,
            args.close_ratio,
            float(args.max_hold_close_ratio),
        )
        args.squeeze_close_ratio = _clamp(
            float(args.squeeze_close_ratio) + close_delta,
            args.close_ratio,
            float(args.max_hold_close_ratio),
        )
        args.hold_close_ratio = _clamp(
            float(args.hold_close_ratio) + close_delta,
            args.close_ratio,
            float(args.max_hold_close_ratio),
        )
        notes.append(f"close {old:.3f}->{args.close_ratio:.3f}")

    if args.policy_action_apply != "full":
        return notes

    radius = max(0.015, float(anchor.get("ball_radius", args.ball_radius)))
    max_contact = max(0.0, float(args.policy_action_max_contact_delta))
    contact_x = _clamp(
        float(deltas.get("hand_pose_target.contact.palm.0", 0.0)),
        -max_contact,
        max_contact,
    )
    contact_z = _clamp(
        float(deltas.get("hand_pose_target.contact.palm.2", 0.0)),
        -max_contact,
        max_contact,
    )
    if abs(contact_x) > 1e-5:
        old = float(args.palm_pocket_x_radius)
        args.palm_pocket_x_radius = _clamp(old + contact_x / radius, -2.60, -0.65)
        notes.append(f"palm_x_radius {old:.2f}->{args.palm_pocket_x_radius:.2f}")
    if abs(contact_z) > 1e-5:
        old = float(args.palm_pocket_table_z_radius)
        args.palm_pocket_table_z_radius = _clamp(old + contact_z / radius, -0.60, 0.45)
        notes.append(f"palm_table_z_radius {old:.2f}->{args.palm_pocket_table_z_radius:.2f}")

    max_wrist = max(0.0, float(args.policy_action_max_wrist_delta))
    wrist_delta = _clamp(
        float(deltas.get("wrist_target.pitch", 0.0)),
        -max_wrist,
        max_wrist,
    )
    if abs(wrist_delta) > 1e-5:
        old = float(args.grasp_wrist_pitch)
        args.grasp_wrist_pitch = _clamp(old + wrist_delta, -0.42, 0.10)
        args.reach_wrist_pitch = _clamp(float(args.reach_wrist_pitch) + 0.5 * wrist_delta, -0.36, 0.10)
        args.ik_wrist_pitch_min = min(float(args.ik_wrist_pitch_min), args.grasp_wrist_pitch - 0.02)
        notes.append(f"grasp_wrist_pitch {old:.3f}->{args.grasp_wrist_pitch:.3f}")
    return notes


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
        movement=(float(args.approach_movement_x), 0.0, 0.0),
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
    pregrasp_align = align_ready if bool(args.pregrasp_align_base) else ready
    pregrasp_align_duration = float(args.pregrasp_align_duration)
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
        DemoPhase("fine_align_before_grasp", pregrasp_align_duration, pregrasp_align, pregrasp_align),
        DemoPhase("approach_from_above", args.pregrasp_duration, ready, pregrasp),
        DemoPhase("lower_to_ball_open", args.reach_duration, pregrasp_open, grasp_open),
        DemoPhase("capture_ball_contact", args.capture_duration, pregrasp_open, capture),
        DemoPhase("close_on_ball", args.close_duration, capture, grasp),
        DemoPhase("squeeze_ball_secure", args.palm_squeeze_duration, grasp, squeeze),
        # Give the compliant hand time to settle around the object before
        # applying vertical motion.  The world-model lift primitive consumes
        # this as a separate, parameterized low-height hold phase.
        DemoPhase("low_hold_ball", args.low_hold_duration, squeeze, squeeze),
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
    map_fallback_note = ""
    if (
        update_walk
        and bool(getattr(args, "initial_map_anchor_fallback", False))
        and all(math.isfinite(v) for v in ball_base)
        and ball_base[0] < float(args.initial_map_anchor_min_x)
    ):
        ball_map = _finite_vec3(anchor.get("ball_center_map"))
        place_map = _finite_vec3(anchor.get("place_center_map"))
        if ball_map is not None:
            start_x = float(args.initial_map_base_x)
            start_y = float(args.initial_map_base_y)
            start_z = float(args.initial_map_base_z)
            ball_base = [
                float(ball_map[0]) - start_x,
                float(ball_map[1]) - start_y,
                float(ball_map[2]) - start_z,
            ]
            if place_map is not None:
                place_base = [
                    float(place_map[0]) - start_x,
                    float(place_map[1]) - start_y,
                    float(place_map[2]) - start_z,
                ]
            walk_distance = max(0.0, float(ball_base[0]) - float(args.approach_target_x))
            walk_duration = walk_distance / max(0.05, float(args.walk_speed)) + float(args.walk_extra_duration)
            args.walk_duration = min(float(args.max_approach_duration), max(0.0, walk_duration))
            map_fallback_note = " map_fallback=start_pose"
    if all(math.isfinite(v) for v in ball_base):
        args.assist_x = float(ball_base[0])
        args.assist_y = float(ball_base[1])
        args.assist_z = float(ball_base[2])
    if all(math.isfinite(v) for v in place_base):
        args.place_assist_x = float(place_base[0])
        args.place_assist_y = float(place_base[1])
        args.place_assist_z = float(place_base[2])
    policy_notes = _apply_policy_navigation(args)
    align_error = math.nan
    align_error_x = math.nan
    if all(math.isfinite(v) for v in ball_base):
        align_error_x = float(ball_base[0]) - float(args.align_target_x)
        args.align_movement_x = _signed_axis_command(
            align_error_x,
            tolerance=float(args.align_x_tolerance),
            gain=float(args.align_forward_gain),
            max_abs=float(args.align_max_forward),
            response_sign=float(args.align_forward_response_sign),
        )
        align_plan = WorkspaceAligner(
            target_y=float(args.align_target_y),
            tolerance=float(args.align_y_tolerance),
            lateral_gain=float(args.align_lateral_gain),
            speed=float(args.align_speed),
            min_duration=float(args.align_min_duration),
            max_duration=float(args.align_max_duration),
            duration_gain=float(args.align_duration_gain),
            response_sign=float(args.align_lateral_response_sign),
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
    policy_notes.extend(_apply_policy_manipulation(args, anchor))
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
        f"reach_pitch={args.reach_pitch:.2f} "
        f"policy={'|'.join(policy_notes) if policy_notes else 'none'}{map_fallback_note}"
    )


class BallPickPlaceDemo(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ball_pick_place_demo")
        self.args = args
        self.args.ik_pose_overrides = {}
        self.rollout = logger_from_args(
            args,
            demo_kind="ball",
            task_id=str(args.task_id),
            scene=str(args.scene),
            metadata={
                "anchor_topic": args.anchor_topic,
                "use_anchor": bool(args.use_ball_anchor),
                "require_anchor": bool(args.require_ball_anchor),
            },
        )
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

        graph_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.phase_pub = self.create_publisher(String, "/sonic_demo/phase", 10)
        self.skill_graph_pub = self.create_publisher(String, "/sonic_demo/skill_graph", graph_qos)
        self.runtime_plan_pub = self.create_publisher(String, "/sonic_demo/runtime_plan", graph_qos)
        self.latest_anchor: dict | None = None
        self._anchor_wall_time = 0.0
        self._anchor_stamp_wall_time = 0.0
        self._anchor_min_stamp_wall_time = time.time() - 0.10
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
        self._lift_success_seen = False
        self._task_failure_reason: str | None = None
        self._task_failure_metrics: dict[str, float | int | str | None] = {}
        self._anchor_failure_reason: str | None = None
        self._anchor_failure_metrics: dict[str, float | int | str | None] = {}
        self._workspace_align_retries: dict[str, int] = {}
        self._last_workspace_servo_log = 0.0
        self._workspace_response_sign = np.asarray(
            [
                -1.0 if float(args.align_forward_response_sign) < 0.0 else 1.0,
                -1.0 if float(args.align_lateral_response_sign) < 0.0 else 1.0,
            ],
            dtype=np.float64,
        )
        self._workspace_prev_error: np.ndarray | None = None
        self._workspace_prev_cmd: np.ndarray | None = None
        self._workspace_response_ref_error: np.ndarray | None = None
        self._workspace_response_ref_cmd: np.ndarray | None = None
        self._workspace_response_votes = np.zeros(2, dtype=np.int32)
        self._approach_prev_forward: float | None = None
        self._approach_response_flips = 0
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
        self.rollout.log_event(
            "task_start",
            status="running",
            metadata={
                "phase_count": len(self.phases),
                "phases": [phase.name for phase in self.phases],
                "rollout_log": str(self.rollout.path),
            },
        )
        _log(f"rollout log: {self.rollout.path}")

    def run(self) -> None:
        time.sleep(float(self.args.zmq_connect_wait))
        self._write_ball_attach(False, "warmup")
        time.sleep(float(self.args.warmup))
        initial_anchor_after = time.monotonic()
        self._send_start()
        self._hold_start_pose(float(self.args.post_start_anchor_delay))

        if self.args.use_ball_anchor:
            initial_anchor_ok = self._wait_and_apply_anchor(fresh_after=initial_anchor_after)
            if self.args.require_ball_anchor and not initial_anchor_ok:
                raise RuntimeError(
                    f"ball anchor is required but no anchor was received on {self.args.anchor_topic}"
                )

        period = 1.0 / max(5.0, float(self.args.rate))
        phase_index = 0
        approach_retries = 0
        while phase_index < len(self.phases):
            phase = self.phases[phase_index]
            if phase.name == "walk_to_table" and self._approach_prev_forward is None:
                self._approach_prev_forward = self._current_ball_forward()
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
            self.rollout.phase_start(phase.name, duration=phase.duration)
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
            phase_elapsed = time.monotonic() - t0
            end_cmd = self._contact_servo_command(phase.name, 1.0, phase.end)
            self._publish_planner(end_cmd)
            self._update_ball_attach(phase.name, 1.0)
            self.rollout.phase_end(phase.name, elapsed=phase_elapsed)
            if phase.name == "lift_ball" and self.args.use_ball_anchor:
                lifted = self._wait_for_lifted_ball(timeout=float(self.args.lift_detect_timeout))
                if not lifted and self._lift_retries < int(self.args.lift_max_retries):
                    self._lift_retries += 1
                    reason = "lift_not_stable"
                    _log(
                        f"lift not stable: retry {self._lift_retries}/{self.args.lift_max_retries}"
                    )
                    self.rollout.log_event(
                        "retry",
                        phase=phase.name,
                        status="retry",
                        reason=reason,
                        metrics={
                            "attempt": self._lift_retries,
                            "max_attempts": int(self.args.lift_max_retries),
                        },
                    )
                    if self.args.lift_retry_regrasp:
                        anchor_ok = self._wait_and_apply_anchor(
                            update_walk=False,
                            timeout=float(self.args.retrack_timeout),
                            fresh_after=t0,
                        )
                        if not anchor_ok and self._anchor_failure_is_terminal():
                            self._mark_task_failed(
                                self._anchor_failure_reason or "missing_or_implausible_anchor",
                                phase=phase.name,
                                metrics=self._anchor_failure_metrics,
                            )
                            break
                        recovery = self._lift_regrasp_phases(phase)
                        self.phases[phase_index : phase_index + 1] = recovery
                    else:
                        self.phases[phase_index] = DemoPhase(
                            "lift_ball",
                            float(self.args.lift_retry_duration),
                            phase.end,
                            phase.end,
                        )
                    continue
                if not lifted:
                    self._mark_task_failed(
                        "lift_delta_below_threshold",
                        phase=phase.name,
                        metrics={"attempts": self._lift_retries + 1},
                    )
            if phase.name == "release_ball":
                self._write_ball_attach(False, "release_ball")
            if phase.name.startswith("fine_align") and self.args.use_ball_anchor:
                anchor_ok = self._wait_and_apply_anchor(
                    update_walk=False,
                    timeout=float(self.args.retrack_timeout),
                    fresh_after=t0,
                )
                if not anchor_ok and self._anchor_failure_is_terminal():
                    self._mark_task_failed(
                        self._anchor_failure_reason or "missing_or_implausible_anchor",
                        phase=phase.name,
                        metrics=self._anchor_failure_metrics,
                    )
                    break
                workspace_error = self._workspace_error_xy()
                if workspace_error is not None:
                    error_x, error_y = workspace_error
                    if phase.name == "fine_align_before_grasp" and not self.args.pregrasp_align_base:
                        if abs(error_y) > float(self.args.align_y_tolerance) or error_x < -float(self.args.align_close_x_tolerance):
                            _log(
                                f"pregrasp base align disabled: error=({error_x:.3f},{error_y:.3f})m; "
                                "using hand contact servo"
                            )
                            self.rollout.log_event(
                                "phase_observation",
                                phase=phase.name,
                                status="review",
                                reason="pregrasp_base_align_disabled",
                                metrics={"error_x": error_x, "error_y": error_y},
                            )
                        phase_index += 1
                        continue
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
                            reason = "workspace_alignment_residual"
                            _log(
                                f"workspace alignment residual: error=({error_x:.3f},{error_y:.3f})m "
                                f"retry {retries + 1}/{self.args.align_max_retries}"
                            )
                            self.rollout.log_event(
                                "retry",
                                phase=phase.name,
                                status="retry",
                                reason=reason,
                                metrics={
                                    "attempt": retries + 1,
                                    "max_attempts": int(self.args.align_max_retries),
                                    "error_x": error_x,
                                    "error_y": error_y,
                                },
                            )
                            phase = self.phases[phase_index]
                            retry_duration = (
                                float(self.args.pregrasp_align_duration)
                                if phase.name == "fine_align_before_grasp"
                                else float(self.args.align_duration)
                            )
                            self.phases[phase_index] = DemoPhase(
                                phase.name,
                                retry_duration,
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
                            self.rollout.log_event(
                                "phase_observation",
                                phase=phase.name,
                                status="review",
                                reason="workspace_soft_continue",
                                metrics={"error_x": error_x, "error_y": error_y},
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
                    reason = "capture_contact_not_ready"
                    _log(
                        f"capture not ready: rms={err:.3f}m max={self._last_contact_max_error or math.nan:.3f}m "
                        f"retry {self._contact_capture_retries}/{self.args.capture_max_retries}"
                    )
                    self.rollout.log_event(
                        "retry",
                        phase=phase.name,
                        status="retry",
                        reason=reason,
                        metrics={
                            "attempt": self._contact_capture_retries,
                            "max_attempts": int(self.args.capture_max_retries),
                            "rms_error": err,
                            "max_error": self._last_contact_max_error,
                        },
                    )
                    retry_cmd = phase.end
                    self.phases[phase_index] = DemoPhase(
                        "capture_ball_contact",
                        float(self.args.capture_retry_duration),
                        retry_cmd,
                        retry_cmd,
                    )
                    continue
                if self.args.contact_servo and err is not None and not self._last_contact_ready:
                    self._mark_task_failed(
                        "capture_contact_not_ready",
                        phase=phase.name,
                        metrics={"rms_error": err, "max_error": self._last_contact_max_error},
                    )
            if phase.name == "squeeze_ball_secure":
                err = self._last_contact_error_norm
                if (
                    self.args.contact_servo
                    and err is not None
                    and not self._last_contact_ready
                    and self._contact_squeeze_retries < int(self.args.squeeze_max_retries)
                ):
                    self._contact_squeeze_retries += 1
                    reason = "palm_pocket_not_ready"
                    _log(
                        f"palm pocket not ready: rms={err:.3f}m max={self._last_contact_max_error or math.nan:.3f}m "
                        f"retry {self._contact_squeeze_retries}/{self.args.squeeze_max_retries}"
                    )
                    self.rollout.log_event(
                        "retry",
                        phase=phase.name,
                        status="retry",
                        reason=reason,
                        metrics={
                            "attempt": self._contact_squeeze_retries,
                            "max_attempts": int(self.args.squeeze_max_retries),
                            "rms_error": err,
                            "max_error": self._last_contact_max_error,
                        },
                    )
                    retry_cmd = phase.end
                    self.phases[phase_index] = DemoPhase(
                        "squeeze_ball_secure",
                        float(self.args.squeeze_retry_duration),
                        retry_cmd,
                        retry_cmd,
                    )
                    continue
                if self.args.contact_servo and err is not None and not self._last_contact_ready:
                    self._mark_task_failed(
                        "palm_pocket_not_ready",
                        phase=phase.name,
                        metrics={"rms_error": err, "max_error": self._last_contact_max_error},
                    )
            if phase.name == "walk_to_table" and self.args.use_ball_anchor:
                forward = self._current_ball_forward()
                target = float(self.args.approach_target_x) + float(self.args.approach_tolerance)
                if forward is not None and forward > target:
                    soft_target = target + float(self.args.approach_soft_tolerance)
                    if forward <= soft_target:
                        _log(
                            f"approach close enough: ball_x={forward:.2f}m target={target:.2f}m "
                            f"soft_limit={soft_target:.2f}m; continuing"
                        )
                        self.rollout.log_event(
                            "phase_observation",
                            phase=phase.name,
                            status="review",
                            reason="approach_soft_continue",
                            metrics={"ball_x": forward, "target_x": target, "soft_limit": soft_target},
                        )
                    elif approach_retries < int(self.args.max_approach_retries):
                        approach_retries += 1
                        reason = "approach_still_far"
                        retry_duration = self._approach_retry_duration(forward, target)
                        _log(
                            f"approach still far: ball_x={forward:.2f}m target={target:.2f}m; "
                            f"retry {approach_retries}/{self.args.max_approach_retries} "
                            f"for {retry_duration:.2f}s"
                        )
                        self.rollout.log_event(
                            "retry",
                            phase=phase.name,
                            status="retry",
                            reason=reason,
                            metrics={
                                "attempt": approach_retries,
                                "max_attempts": int(self.args.max_approach_retries),
                                "ball_x": forward,
                                "target_x": target,
                                "retry_duration": retry_duration,
                            },
                        )
                        retry_cmd = self._approach_retry_command(phase.end, forward)
                        self.phases[phase_index] = DemoPhase(
                            "walk_to_table",
                            retry_duration,
                            retry_cmd,
                            retry_cmd,
                        )
                        continue
                    elif self.args.require_ball_anchor:
                        self.rollout.log_event(
                            "task_end",
                            phase=phase.name,
                            status="failed",
                            reason="approach_failed",
                            metrics={"ball_x": forward, "soft_limit": soft_target},
                        )
                        raise RuntimeError(
                            f"approach failed: ball_x={forward:.2f}m remains beyond soft limit {soft_target:.2f}m"
                        )
            phase_index += 1

        done_msg = String()
        done_msg.data = "done"
        self.phase_pub.publish(done_msg)
        self._write_ball_attach(False, "done")
        if self.args.use_ball_anchor and not self._lift_success_seen:
            self._mark_task_failed("lift_not_confirmed", phase="lift_ball")
        final_status = "failed" if self._task_failure_reason else "success"
        self.rollout.log_event(
            "task_end",
            phase="done",
            status=final_status,
            reason=self._task_failure_reason,
            metrics=self._task_failure_metrics,
        )
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
        self.rollout.close()
        self.socket.close(0)

    def _lift_regrasp_phases(self, failed_lift: DemoPhase) -> list[DemoPhase]:
        return [
            DemoPhase(
                "capture_ball_contact",
                float(self.args.lift_regrasp_capture_duration),
                failed_lift.start,
                failed_lift.start,
            ),
            DemoPhase(
                "close_on_ball",
                float(self.args.lift_regrasp_close_duration),
                failed_lift.start,
                failed_lift.start,
            ),
            DemoPhase(
                "squeeze_ball_secure",
                float(self.args.lift_regrasp_squeeze_duration),
                failed_lift.start,
                failed_lift.start,
            ),
            DemoPhase(
                "lift_ball",
                float(self.args.lift_retry_duration),
                failed_lift.start,
                failed_lift.end,
            ),
        ]

    def _clear_recoverable_failure(self) -> None:
        if self._task_failure_reason in {
            "capture_contact_not_ready",
            "palm_pocket_not_ready",
            "lift_delta_below_threshold",
            "lift_not_confirmed",
        }:
            _log(f"task failure cleared after lift success: {self._task_failure_reason}")
            self._task_failure_reason = None
            self._task_failure_metrics = {}

    def _mark_task_failed(
        self,
        reason: str,
        *,
        phase: str | None = None,
        metrics: dict[str, float | int | str | None] | None = None,
    ) -> None:
        if self._task_failure_reason is not None and not (
            self._failure_reason_is_terminal(reason)
            and not self._failure_reason_is_terminal(self._task_failure_reason)
        ):
            return
        self._task_failure_reason = reason
        self._task_failure_metrics = dict(metrics or {})
        _log(f"task marked failed: {reason}")
        self.rollout.log_event(
            "phase_observation",
            phase=phase,
            status="failed",
            reason=reason,
            metrics=self._task_failure_metrics,
        )

    def _anchor_cb(self, msg: String) -> None:
        try:
            anchor = json.loads(msg.data)
            stamp_wall = _anchor_stamp_wall_time(anchor)
            if stamp_wall + 0.05 < self._anchor_min_stamp_wall_time:
                return
            self.latest_anchor = anchor
            self._anchor_wall_time = time.monotonic()
            self._anchor_stamp_wall_time = stamp_wall
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"bad ball anchor JSON: {exc}")

    def _failure_reason_is_terminal(self, reason: str | None) -> bool:
        return reason in {
            "object_out_of_workspace_z",
            "object_out_of_workspace_y",
            "missing_or_implausible_anchor",
        }

    def _anchor_failure_is_terminal(self) -> bool:
        return self._failure_reason_is_terminal(self._anchor_failure_reason)

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
        if phase_name == "fine_align_before_grasp" and not self.args.pregrasp_align_base:
            return DemoCommand(
                mode=LOCO_IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=cmd.facing,
                speed=cmd.speed,
                height=cmd.height,
                upper_body=cmd.upper_body,
                left_hand=cmd.left_hand,
                right_hand=cmd.right_hand,
            )
        error = self._workspace_error_xy()
        if error is None:
            return cmd
        error_vec = np.asarray(error, dtype=np.float64)
        self._update_workspace_response(error_vec)
        error_x, error_y = (float(error_vec[0]), float(error_vec[1]))
        move_x = _signed_axis_command(
            error_x,
            tolerance=float(self.args.align_x_tolerance),
            gain=float(self.args.align_forward_gain),
            max_abs=float(self.args.align_max_forward),
            response_sign=float(self._workspace_response_sign[0]),
        )
        move_y = _signed_axis_command(
            error_y,
            tolerance=float(self.args.align_y_tolerance),
            gain=float(self.args.align_lateral_gain),
            max_abs=float(self.args.align_max_lateral),
            response_sign=float(self._workspace_response_sign[1]),
        )
        cmd_vec = np.asarray([move_x, move_y], dtype=np.float64)
        self._workspace_prev_error = error_vec
        self._workspace_prev_cmd = cmd_vec
        self._seed_workspace_response_reference(error_vec, cmd_vec)
        mode = LOCO_SLOW_WALK if max(abs(move_x), abs(move_y)) > 1e-3 else LOCO_IDLE
        now = time.monotonic()
        if now - self._last_workspace_servo_log > float(self.args.align_log_period):
            _log(
                f"workspace servo {phase_name}: "
                f"err=({error_x:.3f},{error_y:.3f}) "
                f"move=({move_x:.2f},{move_y:.2f}) "
                f"resp=({self._workspace_response_sign[0]:+.0f},{self._workspace_response_sign[1]:+.0f}) "
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

    def _update_workspace_response(self, error_vec: np.ndarray) -> None:
        if not bool(self.args.align_response_adapt):
            return
        if self._workspace_response_ref_error is None or self._workspace_response_ref_cmd is None:
            return
        min_cmd = float(self.args.align_response_min_cmd)
        min_delta = float(self.args.align_response_min_delta)
        for axis in range(2):
            cmd = float(self._workspace_response_ref_cmd[axis])
            if abs(cmd) < min_cmd:
                continue
            delta = float(error_vec[axis] - self._workspace_response_ref_error[axis])
            if abs(delta) < min_delta:
                continue
            estimated = 1.0 if delta * cmd > 0.0 else -1.0
            if estimated != float(self._workspace_response_sign[axis]):
                self._workspace_response_votes[axis] += 1
                required = max(1, int(self.args.align_response_flip_votes))
                if int(self._workspace_response_votes[axis]) >= required:
                    self._workspace_response_sign[axis] = estimated
                    self._workspace_response_votes[axis] = 0
                    _log(
                        "workspace response sign update: "
                        f"axis={'x' if axis == 0 else 'y'} sign={estimated:+.0f} "
                        f"delta_error={delta:+.3f} cmd={cmd:+.3f} votes={required}"
                    )
            else:
                self._workspace_response_votes[axis] = 0
            self._workspace_response_ref_error[axis] = float(error_vec[axis])
            if self._workspace_prev_cmd is not None:
                self._workspace_response_ref_cmd[axis] = float(self._workspace_prev_cmd[axis])

    def _seed_workspace_response_reference(self, error_vec: np.ndarray, cmd_vec: np.ndarray) -> None:
        if not bool(self.args.align_response_adapt):
            return
        min_cmd = float(self.args.align_response_min_cmd)
        if self._workspace_response_ref_error is None or self._workspace_response_ref_cmd is None:
            self._workspace_response_ref_error = error_vec.copy()
            self._workspace_response_ref_cmd = cmd_vec.copy()
            return
        for axis in range(2):
            cmd = float(cmd_vec[axis])
            ref_cmd = float(self._workspace_response_ref_cmd[axis])
            if abs(cmd) < min_cmd:
                self._workspace_response_ref_error[axis] = float(error_vec[axis])
                self._workspace_response_ref_cmd[axis] = 0.0
                continue
            if abs(ref_cmd) < min_cmd or cmd * ref_cmd < 0.0:
                self._workspace_response_ref_error[axis] = float(error_vec[axis])
                self._workspace_response_ref_cmd[axis] = cmd

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
            self.rollout.log_event(
                "phase_observation",
                phase=phase_name,
                status="success",
                reason="approach_reached",
                metrics={"ball_x": forward, "target_x": float(self.args.approach_target_x), "elapsed": elapsed},
            )
            return True
        return False

    def _approach_retry_duration(self, forward: float, target: float) -> float:
        residual = max(0.0, float(forward) - float(target))
        nominal_speed = max(0.08, float(self.args.walk_speed) * 0.55)
        duration = float(self.args.approach_retry_duration) + (
            residual / nominal_speed * float(self.args.approach_retry_gain)
        )
        return _clamp(
            duration,
            float(self.args.approach_retry_duration),
            float(self.args.approach_retry_max_duration),
        )

    def _approach_retry_command(self, cmd: DemoCommand, forward: float) -> DemoCommand:
        movement = tuple(float(v) for v in cmd.movement)
        movement_x = float(movement[0])
        if (
            bool(self.args.approach_response_adapt)
            and self._approach_prev_forward is not None
            and self._approach_response_flips < int(self.args.approach_response_max_flips)
            and abs(movement_x) > 1e-3
        ):
            delta = float(forward) - float(self._approach_prev_forward)
            if delta > float(self.args.approach_response_min_delta):
                old = movement_x
                movement_x = -movement_x
                self.args.approach_movement_x = movement_x
                self._approach_response_flips += 1
                _log(
                    "approach response flip: "
                    f"ball_x_delta={delta:+.3f} movement_x={old:+.1f}->{movement_x:+.1f}"
                )
                self.rollout.log_event(
                    "phase_observation",
                    phase="walk_to_table",
                    status="review",
                    reason="approach_response_flip",
                    metrics={
                        "ball_x_delta": delta,
                        "previous_ball_x": float(self._approach_prev_forward),
                        "ball_x": float(forward),
                        "old_movement_x": old,
                        "new_movement_x": movement_x,
                    },
                )
        self._approach_prev_forward = float(forward)
        return DemoCommand(
            mode=cmd.mode,
            movement=(movement_x, movement[1], movement[2]),
            facing=cmd.facing,
            speed=cmd.speed,
            height=cmd.height,
            upper_body=cmd.upper_body,
            left_hand=cmd.left_hand,
            right_hand=cmd.right_hand,
        )

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
            self.rollout.log_event(
                "phase_observation",
                phase=phase_name,
                status="success",
                reason="workspace_aligned",
                metrics={
                    "error_x": error_x,
                    "error_y": error_y,
                    "target_x": float(self.args.align_target_x),
                    "target_y": float(self.args.align_target_y),
                    "elapsed": elapsed,
                },
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
            reason = self._anchor_implausible_reason(self.latest_anchor, update_walk=update_walk) if has_anchor else "missing_or_implausible_anchor"
            is_plausible = (
                has_anchor
                and is_fresh
                and reason is None
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
                    f"latest ball_base=({x:.2f},{y:.2f},{z:.2f}) reason={reason}"
                )
                reported_implausible = True
            rclpy.spin_once(self, timeout_sec=0.05)
        stale = fresh_after is not None and self._anchor_wall_time <= fresh_after
        failure_reason = self._anchor_implausible_reason(self.latest_anchor, update_walk=update_walk)
        if self.latest_anchor is None or stale or failure_reason is not None:
            ball_base = (
                _finite_vec3(self.latest_anchor.get("ball_point_base"))
                if self.latest_anchor is not None
                else None
            )
            metrics: dict[str, float | int | str | None] = {
                "update_walk": int(bool(update_walk)),
                "timeout": timeout,
            }
            if ball_base is not None:
                metrics.update(
                    {"ball_x": ball_base[0], "ball_y": ball_base[1], "ball_z": ball_base[2]}
                )
            reason = failure_reason or "missing_or_implausible_anchor"
            self._anchor_failure_reason = reason
            self._anchor_failure_metrics = metrics
            if reported_implausible:
                _log(
                    f"no plausible ball anchor received on {self.args.anchor_topic}: "
                    f"{reason}; using fixed defaults"
                )
            else:
                _log(f"no ball anchor received on {self.args.anchor_topic}; using fixed defaults")
            self.rollout.log_event(
                "anchor_update",
                status="failed",
                reason=reason,
                metrics=metrics,
            )
            return False
        try:
            summary = apply_ball_anchor(self.args, self.latest_anchor, update_walk=update_walk)
            if not update_walk:
                self._update_ik_poses_from_anchor(self.latest_anchor)
            self.phases = make_demo_phases(self.args)
            self._servo_cache_cmd = None
            self._servo_cache_phase = None
            self._publish_skill_graph(self.latest_anchor)
            _log(f"using ball anchor: {summary}")
            self.rollout.log_event(
                "anchor_update",
                status="success",
                reason="ball_anchor_applied",
                metrics={"update_walk": bool(update_walk)},
                metadata={"summary": summary},
            )
            self._anchor_failure_reason = None
            self._anchor_failure_metrics = {}
            return True
        except Exception as exc:
            self.get_logger().warn(f"failed to apply ball anchor; using fixed defaults: {exc}")
            self.rollout.log_event(
                "anchor_update",
                status="failed",
                reason="apply_anchor_exception",
                metadata={"error": str(exc)},
            )
            return False

    def _publish_skill_graph(self, anchor: dict) -> None:
        try:
            world = anchor_to_world(anchor)
            request = TaskRequest(
                verb="pick_place",
                object_id=str(anchor.get("ball_name", "ball")),
                target_id="place_target" if "place_center_map" in anchor else None,
            )
            graph = TaskPlanner().plan(world, request)
            runtime = runtime_plan_for_graph(graph, demo_kind="ball")
        except Exception as exc:
            self.get_logger().warn(f"failed to publish ball skill graph: {exc}")
            return
        skill_msg = String()
        skill_msg.data = json.dumps(graph.to_dict(), separators=(",", ":"))
        runtime_msg = String()
        runtime_msg.data = json.dumps(runtime.to_dict(), separators=(",", ":"))
        self.skill_graph_pub.publish(skill_msg)
        self.runtime_plan_pub.publish(runtime_msg)
        _log(f"skill graph: {skill_summary(graph)}")

    def _anchor_is_plausible(self, anchor: dict | None, *, update_walk: bool) -> bool:
        return self._anchor_implausible_reason(anchor, update_walk=update_walk) is None

    def _anchor_implausible_reason(self, anchor: dict | None, *, update_walk: bool) -> str | None:
        if anchor is None:
            return "missing_or_implausible_anchor"
        ball_base = _finite_vec3(anchor.get("ball_point_base"))
        if ball_base is None:
            return "missing_or_implausible_anchor"
        x, y, z = ball_base
        if z < float(self.args.runtime_anchor_min_z):
            return "object_out_of_workspace_z"
        if z > float(self.args.runtime_anchor_max_z):
            return "object_out_of_workspace_z"
        if abs(y) > float(self.args.runtime_anchor_max_abs_y):
            return "object_out_of_workspace_y"
        if not update_walk:
            return None
        if not (
            float(self.args.initial_anchor_min_x) <= x <= float(self.args.initial_anchor_max_x)
            and abs(y) <= float(self.args.initial_anchor_max_abs_y)
            and float(self.args.initial_anchor_min_z) <= z <= float(self.args.initial_anchor_max_z)
        ):
            return "missing_or_implausible_anchor"
        return None

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
        self.rollout.log_event(
            "lift_reference",
            phase="lift_ball",
            status="sampled",
            metrics={"ball_map_z": self._ball_pre_lift_z},
        )
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
        contact_metrics: dict[str, list[float]] = {}
        for name, body_id, local_point in samples:
            point_world = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point
            point_base = base_rot.T @ (point_world - base_pos)
            rel = point_base - ball
            contact_metrics[name] = [float(value) for value in rel]
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
        self.rollout.log_event(
            "grasp_geometry",
            phase="lift_ball",
            status="sampled",
            metrics={
                "contacts_rel_to_ball_base": contact_metrics,
                "hand_q": hand_values,
                "wrist_q": wrist_values,
            },
        )

    def _wait_for_lifted_ball(self, *, timeout: float) -> bool:
        if self._ball_pre_lift_z is None:
            self._mark_lift_reference()
        if self._ball_pre_lift_z is None:
            _log("ball lift check skipped: no fresh anchor")
            self.rollout.log_event(
                "lift_check",
                phase="lift_ball",
                status="skipped",
                reason="missing_fresh_anchor",
            )
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
                self._lift_success_seen = True
                self._clear_recoverable_failure()
                self.rollout.log_event(
                    "lift_check",
                    phase="lift_ball",
                    status="success",
                    metrics={"delta_z": delta, "threshold": threshold},
                )
                return True
            time.sleep(0.03)
        if math.isfinite(best_delta):
            _log(f"ball not lifted yet: best_dz={best_delta:.3f}m")
        else:
            _log("ball lift check had no fresh samples")
        self.rollout.log_event(
            "lift_check",
            phase="lift_ball",
            status="failed",
            reason="lift_delta_below_threshold",
            metrics={
                "best_delta_z": best_delta if math.isfinite(best_delta) else None,
                "threshold": threshold,
            },
        )
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

    def _hold_start_pose(self, duration: float) -> None:
        duration = max(0.0, float(duration))
        if duration <= 0.0:
            return
        cmd = self.phases[0].start
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self._publish_planner(cmd)
            time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))

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
        # Teacher data may engage the simulator-only contact lock as soon as
        # capture starts. Production rollouts keep this disabled and must earn
        # contact through the physical hand/object interaction.
        if bool(self.args.teacher_pregrasp_attach) and phase_name == "capture_ball_contact":
            active = True
        # This late variant retains the real side-grasp effect oracle and only
        # assists the lift trajectory after that primitive has completed.
        if bool(self.args.teacher_lift_attach) and phase_name == "low_hold_ball":
            active = True
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
    parser.add_argument("--post-start-anchor-delay", type=float, default=0.0)
    parser.add_argument("--zmq-bind-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-connect-wait", type=float, default=1.0)
    parser.add_argument("--start-bursts", type=int, default=8)
    parser.add_argument("--max-upper-body-velocity", type=float, default=1.8)
    parser.add_argument("--scene", default="ball_demo")
    parser.add_argument("--qpos-path", default="/tmp/sonic_qpos.npy")
    parser.add_argument("--task-id", default="ball_demo")
    add_rollout_log_args(parser)

    parser.add_argument("--stand-height", type=float, default=0.78)
    parser.add_argument("--walk-speed", type=float, default=0.24)
    parser.add_argument("--walk-duration", type=float, default=2.0)
    parser.add_argument("--walk-extra-duration", type=float, default=3.8)
    parser.add_argument("--max-approach-duration", type=float, default=10.5)
    parser.add_argument("--min-approach-duration", type=float, default=1.0)
    parser.add_argument("--approach-target-x", type=float, default=0.56)
    parser.add_argument("--approach-tolerance", type=float, default=0.08)
    parser.add_argument("--approach-soft-tolerance", type=float, default=0.08)
    parser.add_argument("--max-approach-retries", type=int, default=8)
    parser.add_argument("--approach-retry-duration", type=float, default=0.65)
    parser.add_argument("--approach-retry-gain", type=float, default=0.65)
    parser.add_argument("--approach-retry-max-duration", type=float, default=1.8)
    parser.add_argument("--approach-movement-x", type=float, default=1.0)
    parser.add_argument("--approach-response-adapt", action="store_true", default=False)
    parser.add_argument("--no-approach-response-adapt", dest="approach_response_adapt", action="store_false")
    parser.add_argument("--approach-response-min-delta", type=float, default=0.03)
    parser.add_argument("--approach-response-max-flips", type=int, default=1)
    parser.add_argument("--align-target-x", type=float, default=0.54)
    parser.add_argument("--align-x-tolerance", type=float, default=0.045)
    parser.add_argument("--align-close-x-tolerance", type=float, default=0.16)
    parser.add_argument("--align-x-soft-tolerance", type=float, default=0.10)
    parser.add_argument("--align-forward-gain", type=float, default=4.2)
    parser.add_argument("--align-max-forward", type=float, default=0.90)
    parser.add_argument("--align-forward-response-sign", type=float, default=1.0)
    parser.add_argument("--align-target-y", type=float, default=-0.24)
    parser.add_argument("--align-y-tolerance", type=float, default=0.035)
    parser.add_argument("--align-lateral-gain", type=float, default=5.0)
    parser.add_argument("--align-max-lateral", type=float, default=0.85)
    parser.add_argument("--align-lateral-response-sign", type=float, default=-1.0)
    parser.add_argument("--align-response-adapt", action="store_true", default=False)
    parser.add_argument("--no-align-response-adapt", dest="align_response_adapt", action="store_false")
    parser.add_argument("--align-response-min-cmd", type=float, default=0.05)
    parser.add_argument("--align-response-min-delta", type=float, default=0.008)
    parser.add_argument("--align-response-flip-votes", type=int, default=4)
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
    parser.add_argument("--pregrasp-align-duration", type=float, default=0.20)
    parser.add_argument("--pregrasp-align-base", action="store_true", default=False)

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
    parser.add_argument("--lift-retry-regrasp", action="store_true", default=True)
    parser.add_argument("--no-lift-retry-regrasp", dest="lift_retry_regrasp", action="store_false")
    parser.add_argument("--lift-regrasp-capture-duration", type=float, default=0.75)
    parser.add_argument("--lift-regrasp-close-duration", type=float, default=0.70)
    parser.add_argument("--lift-regrasp-squeeze-duration", type=float, default=0.70)
    parser.add_argument("--secure-duration", type=float, default=1.0)
    parser.add_argument("--low-hold-duration", type=float, default=0.8)
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
    parser.add_argument(
        "--teacher-pregrasp-attach",
        action="store_true",
        default=False,
        help="Simulator-only teacher mode: engage ball attach during capture_ball_contact.",
    )
    parser.add_argument(
        "--teacher-lift-attach",
        action="store_true",
        default=False,
        help="Simulator-only teacher mode: engage ball attach only for low_hold_ball and later lift phases.",
    )
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
    parser.add_argument("--initial-anchor-max-z", type=float, default=0.95)
    parser.add_argument("--initial-map-anchor-fallback", action="store_true")
    parser.add_argument("--initial-map-anchor-min-x", type=float, default=0.45)
    parser.add_argument("--initial-map-base-x", type=float, default=0.0)
    parser.add_argument("--initial-map-base-y", type=float, default=0.0)
    parser.add_argument("--initial-map-base-z", type=float, default=0.793)
    parser.add_argument("--runtime-anchor-min-z", type=float, default=-0.25)
    parser.add_argument("--runtime-anchor-max-z", type=float, default=1.35)
    parser.add_argument("--runtime-anchor-max-abs-y", type=float, default=1.00)
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
    parser.add_argument("--servo-contact-ready-error", type=float, default=0.060)
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
    parser.add_argument("--servo-contact-max-error", type=float, default=0.085)
    parser.add_argument("--palm-pocket-x-radius", type=float, default=-1.85)
    parser.add_argument("--palm-pocket-y-radius", type=float, default=-0.10)
    parser.add_argument("--palm-pocket-table-z-radius", type=float, default=0.00)
    parser.add_argument("--palm-pocket-lift-z-radius", type=float, default=0.02)
    parser.add_argument("--palm-contact-weight", type=float, default=0.35)
    parser.add_argument("--palm-frame-contact-targets", action="store_true", default=True)
    parser.add_argument("--base-frame-contact-targets", dest="palm_frame_contact_targets", action="store_false")
    parser.add_argument(
        "--policy-action-json",
        help="Raw policy-action JSON, policy-sample JSONL, or path to one. Applies task/skill-level parameters only.",
    )
    parser.add_argument("--policy-action-task-id", help="Preferred policy sample task id when reading a JSONL file.")
    parser.add_argument("--policy-action-apply", choices=["off", "safe", "full"], default="safe")
    parser.add_argument(
        "--policy-action-safe-standoff",
        action="store_true",
        help="Allow safe mode to change approach standoff. By default safe mode changes grasp closure only.",
    )
    parser.add_argument("--policy-action-max-standoff-delta", type=float, default=0.03)
    parser.add_argument("--policy-action-max-close-delta", type=float, default=0.10)
    parser.add_argument("--policy-action-max-contact-delta", type=float, default=0.04)
    parser.add_argument("--policy-action-max-wrist-delta", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.policy_action_payload = _load_policy_action(
        args.policy_action_json,
        task_id=args.policy_action_task_id or args.task_id,
    )
    if args.policy_action_json and args.policy_action_payload is None:
        raise RuntimeError(f"no usable ball policy action found in {args.policy_action_json!r}")
    if args.policy_action_payload is not None:
        _log(
            "loaded policy action: "
            f"policy={args.policy_action_payload.get('policy_id')} "
            f"task={args.policy_action_payload.get('task_id')} "
            f"mode={args.policy_action_apply}"
        )
    _apply_ball_scaled_grasp(args, float(args.ball_radius))
    rclpy.init()
    node = BallPickPlaceDemo(args)
    try:
        node.run()
    except KeyboardInterrupt:
        _log("stopping ball pick-place demo")
        node.rollout.log_event("task_end", status="interrupted", reason="keyboard_interrupt")
    except Exception as exc:
        node.rollout.log_event("task_end", status="failed", reason="exception", metadata={"error": str(exc)})
        raise
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
