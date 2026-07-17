#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
MANIPULATION_DIR = SCRIPTS_DIR / "manipulation"
G1_NAV_DIR = REPO / "g1_ros2_nav"
for path in (str(SCRIPTS_DIR), str(REPO), str(MANIPULATION_DIR), str(G1_NAV_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sonic_world.skills import MujocoQposEffectObserver, apply_profile_to_namespace
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene


BALL_PHASE_FALLBACKS = {
    "navigate.approach_object": ("walk_to_table", "fine_align_to_ball"),
    "manip.align_workspace": ("settle_before_pick", "hand_high_ready", "fine_align_before_grasp"),
    "manip.single_hand_pinch": (
        "approach_from_above",
        "lower_to_ball_open",
        "capture_ball_contact",
        "close_on_ball",
        "squeeze_ball_secure",
    ),
    "manip.side_grasp": (
        "approach_from_above",
        "lower_to_ball_open",
        "capture_ball_contact",
        "close_on_ball",
        "squeeze_ball_secure",
    ),
    "manip.top_grasp": (
        "approach_from_above",
        "lower_to_ball_open",
        "capture_ball_contact",
        "close_on_ball",
        "squeeze_ball_secure",
    ),
    "manip.lift_object": ("low_hold_ball", "lift_ball", "secure_ball"),
    "manip.transport_object": ("move_to_place",),
    "manip.place_object": ("lower_to_place",),
    "manip.release": ("release_ball", "retreat_hand", "hold_done"),
}

BOX_PHASE_FALLBACKS = {
    "navigate.approach_object": ("walk_two_steps",),
    "manip.align_workspace": ("settle_before_grasp", "arms_open_table"),
    "manip.bimanual_clamp": ("reach_table_open", "forearm_clamp_box"),
    "manip.lift_object": ("lift_box_from_table", "squeeze_box_secure", "bring_box_to_chest", "carry_settle"),
    "manip.transport_object": ("carry_walk_forward",),
}


class PrimitiveBackend(Protocol):
    backend_name: str

    def execute(self, command: dict[str, Any], *, phases: tuple[str, ...]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume /sonic_world/primitive_command and run skill primitive backends."
    )
    parser.add_argument("--primitive-command-topic", default="/sonic_world/primitive_command")
    parser.add_argument("--primitive-status-topic", default="/sonic_world/primitive_status")
    parser.add_argument("--ball-anchor-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--box-anchor-topic", default="/sonic_demo/box_anchor")
    parser.add_argument("--object-anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--prefer-object-anchor", action="store_true", help="Prefer normalized generic anchors, required for multi-stage episodes.")
    parser.add_argument(
        "--backend",
        choices=["status_only", "contract_test", "zmq_phase"],
        default="status_only",
        help="status_only validates and reports; zmq_phase sends WBC planner phase commands.",
    )
    parser.add_argument("--demo-kind", choices=["auto", "ball", "box"], default="auto")
    parser.add_argument("--scene", help="Scene name/XML passed to the demo phase generator.")
    parser.add_argument("--qpos-path", default="/tmp/sonic_qpos.npy")
    parser.add_argument("--qpos-meta-path", default="/tmp/sonic_qpos_meta.json")
    parser.add_argument(
        "--teacher-assisted",
        action="store_true",
        help="Mark primitive metrics as assisted teacher data; not valid physical benchmark evidence.",
    )
    parser.add_argument(
        "--teacher-assist-skill",
        action="append",
        default=[],
        help="Skill whose success evidence may be replaced by teacher evidence; repeatable. Empty keeps legacy all-skill behavior.",
    )
    parser.add_argument(
        "--effect-observer",
        choices=["auto", "none", "mujoco_qpos"],
        default="auto",
        help="Verify declared effects from live MuJoCo state after primitive actuation.",
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--zmq-bind-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--start-bursts", type=int, default=4)
    parser.add_argument("--max-upper-body-velocity", type=float, default=1.8)
    parser.add_argument("--apply-anchor", action="store_true", default=True)
    parser.add_argument("--no-apply-anchor", dest="apply_anchor", action="store_false")
    parser.add_argument(
        "--demo-arg",
        action="append",
        default=[],
        help="Extra argument passed to the imported demo parse_args for zmq_phase mode. Repeat for pairs.",
    )
    return parser.parse_args()


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def _live_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=4,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


class WorldModelPrimitiveRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_primitive_runner")
        self.args = args
        self.status_pub = self.create_publisher(String, args.primitive_status_topic, 10)
        self.phase_pub = self.create_publisher(String, "/sonic_demo/phase", 10)
        self.latest_ball_anchor: dict[str, Any] | None = None
        self.latest_box_anchor: dict[str, Any] | None = None
        self.latest_object_anchor: dict[str, Any] | None = None
        self.backends: dict[str, PrimitiveBackend] = {}
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.active_action_id: str | None = None
        self.effect_observer = self._build_effect_observer()
        self.create_subscription(String, args.primitive_command_topic, self._command_cb, 10)
        self.create_subscription(String, args.ball_anchor_topic, self._ball_anchor_cb, _live_qos())
        self.create_subscription(String, args.box_anchor_topic, self._box_anchor_cb, _live_qos())
        self.create_subscription(String, args.object_anchor_topic, self._object_anchor_cb, _live_qos())
        self.get_logger().info(
            "Primitive runner listening: "
            f"command={args.primitive_command_topic} status={args.primitive_status_topic} "
            f"backend={args.backend}"
        )

    def _ball_anchor_cb(self, msg: String) -> None:
        self.latest_ball_anchor = _parse_json_or_none(msg.data)

    def _box_anchor_cb(self, msg: String) -> None:
        self.latest_box_anchor = _parse_json_or_none(msg.data)

    def _object_anchor_cb(self, msg: String) -> None:
        self.latest_object_anchor = _parse_json_or_none(msg.data)

    def _command_cb(self, msg: String) -> None:
        if self.worker is not None and self.worker.is_alive():
            command = _parse_json_or_none(msg.data) or {}
            action_id = str(command.get("action_id") or "")
            if action_id and action_id == self.active_action_id:
                # Continuous plan publication must not turn an in-flight identical primitive into a failure.
                self._publish_status(
                    command,
                    "running",
                    demo_kind=str(command.get("demo_kind") or self.args.demo_kind),
                    phases=tuple(),
                    detail="duplicate command acknowledged while primitive remains in flight",
                )
                return
            self._publish_status(
                command,
                "failed",
                demo_kind=str(command.get("demo_kind") or self.args.demo_kind),
                phases=tuple(),
                detail=f"primitive runner busy with action {self.active_action_id}",
            )
            return
        command = _parse_json_or_none(msg.data) or {}
        self.active_action_id = str(command.get("action_id") or "")
        self.cancel_event.clear()
        self.worker = threading.Thread(target=self._execute_command, args=(msg.data,), daemon=True)
        self.worker.start()

    def _execute_command(self, raw: str) -> None:
        started = time.monotonic()
        try:
            command = json.loads(raw)
            if not isinstance(command, dict):
                raise ValueError("primitive command payload must be an object")
            if command.get("schema") != "sonic_skill_primitive_command_v0":
                raise ValueError(f"unsupported primitive command schema {command.get('schema')!r}")
            demo_kind = self._demo_kind(command)
            phases = _phase_names(command, demo_kind=demo_kind)
            self._publish_status(command, "accepted", demo_kind=demo_kind, phases=phases)
            backend = self._backend(demo_kind, command)
            before = self.effect_observer.snapshot(command) if self.effect_observer is not None else None
            result = backend.execute(command, phases=phases)
            if (
                bool(self.args.teacher_assisted)
                and result.get("status") == "success"
                and self._teacher_assists_skill(str(command.get("skill_name") or ""))
            ):
                result["effect_evidence"] = _teacher_assist_evidence(command)
            elif self.effect_observer is not None and before is not None and result.get("status") == "success":
                after = self.effect_observer.snapshot(command)
                result["effect_evidence"] = self.effect_observer.evaluate(command, before, after)
            elapsed = time.monotonic() - started
            self._publish_status(
                command,
                result.get("status", "success"),
                demo_kind=demo_kind,
                phases=phases,
                metrics={"elapsed_s": round(elapsed, 4), **dict(result.get("metrics") or {})},
                backend=result.get("backend") or backend.backend_name,
                detail=str(result.get("detail") or ""),
                effect_evidence=result.get("effect_evidence")
                if isinstance(result.get("effect_evidence"), dict)
                else None,
            )
        except Exception as exc:
            command = _parse_json_or_none(raw) or {}
            if rclpy.ok():
                self._publish_status(
                    command,
                    "cancelled" if self.cancel_event.is_set() else "failed",
                    demo_kind=str(command.get("demo_kind") or self.args.demo_kind),
                    phases=tuple(),
                    detail=str(exc),
                )
        finally:
            self.active_action_id = None

    def _teacher_assists_skill(self, skill_name: str) -> bool:
        selected = {str(item) for item in self.args.teacher_assist_skill if str(item)}
        return not selected or skill_name in selected

    def _build_effect_observer(self) -> MujocoQposEffectObserver | None:
        if self.args.effect_observer == "none":
            return None
        if not self.args.scene:
            if self.args.effect_observer == "mujoco_qpos":
                raise ValueError("--effect-observer mujoco_qpos requires --scene")
            return None
        try:
            scene = resolve_scene(str(self.args.scene), repo_root=REPO).abs_path
        except (FileNotFoundError, ValueError):
            if self.args.effect_observer == "mujoco_qpos":
                raise
            return None
        self.args.scene = str(scene)
        try:
            return MujocoQposEffectObserver(
                scene,
                qpos_path=self.args.qpos_path,
                qpos_meta_path=self.args.qpos_meta_path,
            )
        except Exception:
            if self.args.effect_observer == "mujoco_qpos":
                raise
            return None

    def _demo_kind(self, command: dict[str, Any]) -> str:
        if self.args.demo_kind != "auto":
            return str(self.args.demo_kind)
        raw = command.get("demo_kind")
        if raw in {"ball", "box"}:
            return str(raw)
        skill_name = str(command.get("skill_name") or "")
        if skill_name == "manip.bimanual_clamp":
            return "box"
        if skill_name in BALL_PHASE_FALLBACKS:
            return "ball"
        target_id = str(command.get("target_id") or "")
        if "box" in target_id:
            return "box"
        return "ball"

    def _backend(self, demo_kind: str, command: dict[str, Any]) -> PrimitiveBackend:
        key = f"{self.args.backend}:{demo_kind}"
        if key not in self.backends:
            if self.args.backend == "zmq_phase":
                self.backends[key] = ZmqPhaseBackend(
                    demo_kind=demo_kind,
                    node=self,
                    args=self.args,
                )
            elif self.args.backend == "contract_test":
                self.backends[key] = ContractTestBackend(demo_kind=demo_kind)
            else:
                self.backends[key] = StatusOnlyBackend(demo_kind=demo_kind)
        backend = self.backends[key]
        if isinstance(backend, ZmqPhaseBackend):
            backend.set_anchor(self._anchor_for_demo(demo_kind, target_id=str(command.get("target_id") or "")))
        return backend

    def _anchor_for_demo(self, demo_kind: str, *, target_id: str = "") -> dict[str, Any] | None:
        generic = _generic_anchor_to_demo(self.latest_object_anchor, demo_kind=demo_kind, target_id=target_id)
        if bool(getattr(self.args, "prefer_object_anchor", False)) and generic is not None:
            return generic
        if demo_kind == "box":
            return self.latest_box_anchor or generic
        return self.latest_ball_anchor or generic

    def _publish_status(
        self,
        command: dict[str, Any],
        status: str,
        *,
        demo_kind: str,
        phases: tuple[str, ...],
        metrics: dict[str, Any] | None = None,
        backend: str | None = None,
        detail: str = "",
        effect_evidence: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema": "sonic_skill_primitive_status_v0",
            "event": "primitive_status",
            "status": status,
            "backend": backend or self.args.backend,
            "task_id": command.get("task_id"),
            "action_id": command.get("action_id"),
            "skill_name": command.get("skill_name"),
            "target_id": command.get("target_id"),
            "demo_kind": demo_kind,
            "handler": command.get("handler"),
            "capability": command.get("capability"),
            "phases": list(phases),
            "metrics": metrics or {},
            "detail": detail,
            "effect_evidence": effect_evidence,
            "stamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.status_pub.publish(msg)

    def close(self) -> None:
        self.cancel_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        for backend in self.backends.values():
            backend.close()


class StatusOnlyBackend:
    def __init__(self, *, demo_kind: str):
        self.demo_kind = demo_kind
        self.backend_name = "status_only"

    def execute(self, command: dict[str, Any], *, phases: tuple[str, ...]) -> dict[str, Any]:
        if not phases:
            return {"status": "failed", "detail": "no phases selected", "backend": self.backend_name}
        return {
            "status": "skipped",
            "backend": self.backend_name,
            "metrics": {"phase_count": len(phases)},
            "detail": "validated primitive command without low-level actuation; effects are unverified",
            "effect_evidence": {
                "schema": "sonic_skill_effect_evidence_v0",
                "passed": False,
                "source": self.backend_name,
                "reason": "status_only does not actuate or observe physics",
                "effects": {},
            },
        }

    def close(self) -> None:
        return


class ContractTestBackend:
    """Deterministic backend for transport/sequence integration tests only."""

    def __init__(self, *, demo_kind: str):
        self.demo_kind = demo_kind
        self.backend_name = "contract_test"

    def execute(self, command: dict[str, Any], *, phases: tuple[str, ...]) -> dict[str, Any]:
        effects = {
            str(effect): {"passed": True, "evidence": "contract_test"}
            for effect in command.get("effects", [])
        }
        return {
            "status": "success",
            "backend": self.backend_name,
            "metrics": {"phase_count": len(phases)},
            "effect_evidence": {
                "schema": "sonic_skill_effect_evidence_v0",
                "passed": bool(effects),
                "source": self.backend_name,
                "reason": "deterministic integration-test evidence",
                "effects": effects,
            },
        }

    def close(self) -> None:
        return


class ZmqPhaseBackend:
    def __init__(self, *, demo_kind: str, node: WorldModelPrimitiveRunner, args: argparse.Namespace):
        self.demo_kind = demo_kind
        self.node = node
        self.runner_args = args
        self.backend_name = "zmq_phase"
        self.demo = _load_demo_module(demo_kind)
        self.demo_args = _demo_args(demo_kind, args)
        self.demo_args.zmq_bind_host = args.zmq_bind_host
        self.demo_args.zmq_port = int(args.zmq_port)
        self.demo_args.rate = float(args.rate)
        self.demo_args.start_bursts = int(args.start_bursts)
        self.demo_args.max_upper_body_velocity = float(args.max_upper_body_velocity)
        self.demo_args.warmup = float(args.warmup)
        if self.demo_kind == "ball":
            self.demo_args.pregrasp_align_base = True
            self.demo_args.pregrasp_align_duration = max(
                3.0,
                float(self.demo_args.pregrasp_align_duration),
            )
            self.demo_args.align_response_adapt = True
        if args.scene:
            self.demo_args.scene = str(args.scene)
        self.demo_args.qpos_path = str(args.qpos_path)
        self.demo_args.ik_pose_overrides = {}
        self.ik_solver = self._build_ball_ik_solver()
        self.ik_metrics: dict[str, Any] = {}
        self.last_contact_command: Any | None = None
        self.phases = self.demo.make_demo_phases(self.demo_args)
        self.last_upper_body: list[float] | None = None
        self.last_send_time: float | None = None
        self.anchor: dict[str, Any] | None = None
        self.workspace_response_sign = [
            -1.0 if float(self.demo_args.align_forward_response_sign) < 0.0 else 1.0,
            -1.0 if float(self.demo_args.align_lateral_response_sign) < 0.0 else 1.0,
        ] if self.demo_kind == "ball" else [1.0, 1.0]
        self.workspace_response_votes = [0, 0]
        self.workspace_response_ref_error: list[float] | None = None
        self.workspace_response_ref_cmd: list[float] | None = None
        self.zmq_context = None
        self.socket = None
        self.started = False

    def set_anchor(self, anchor: dict[str, Any] | None) -> None:
        self.anchor = anchor

    def execute(self, command: dict[str, Any], *, phases: tuple[str, ...]) -> dict[str, Any]:
        self._ensure_started()
        payload = command.get("command") if isinstance(command.get("command"), dict) else {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        applied = apply_profile_to_namespace(str(command.get("skill_name") or ""), params, self.demo_args)
        self._apply_anchor()
        selected = self._selected_phases(phases)
        if not selected:
            return {"status": "failed", "backend": self.backend_name, "detail": f"no matching phases for {phases}"}
        low_hold_snapshot: dict[str, Any] | None = None
        for phase in selected:
            self._publish_phase_status(command, phase.name, "phase_start")
            self._play_phase(phase, command=command)
            self._publish_phase_status(command, phase.name, "phase_end")
            if phase.name == "low_hold_ball" and str(command.get("skill_name") or "") == "manip.lift_object":
                low_hold_snapshot = self._low_hold_snapshot(command)
                payload = command.get("command") if isinstance(command.get("command"), dict) else {}
                params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
                required_contacts = int(_bounded_param(params.get("low_hold_min_contacts", 1.0), 1.0, 8.0))
                contacts = int(low_hold_snapshot.get("target_contact_count") or 0) if low_hold_snapshot else 0
                if contacts < required_contacts:
                    return {
                        "status": "failed",
                        "backend": self.backend_name,
                        "detail": f"low_hold_contact_lost: contacts={contacts} required={required_contacts}",
                        "metrics": {
                            "phase_count": len(selected),
                            "low_hold_snapshot": low_hold_snapshot or {},
                            "runtime_overrides": applied,
                            "command_params": {str(key): value for key, value in params.items() if isinstance(value, (int, float, bool))},
                            "ik": dict(self.ik_metrics),
                            "teacher_assisted": bool(self.runner_args.teacher_assisted),
                        },
                    }
        return {
            "status": "success",
            "backend": self.backend_name,
            "metrics": {
                "phase_count": len(selected),
                "duration_s": round(sum(float(p.duration) for p in selected), 4),
                "runtime_overrides": applied,
                "command_params": {str(key): value for key, value in params.items() if isinstance(value, (int, float, bool))},
                "ik": dict(self.ik_metrics),
                "low_hold_snapshot": low_hold_snapshot or {},
                "teacher_assisted": bool(self.runner_args.teacher_assisted),
            },
            "effect_evidence": {
                "schema": "sonic_skill_effect_evidence_v0",
                "passed": False,
                "source": self.backend_name,
                "reason": "phase actuation completed but no physics observer verified effects",
                "effects": {},
            },
        }

    def _low_hold_snapshot(self, command: dict[str, Any]) -> dict[str, Any] | None:
        observer = self.node.effect_observer
        if observer is None:
            return None
        try:
            return observer.snapshot(command)
        except Exception as exc:
            self.ik_metrics["low_hold_observer_error"] = str(exc)
            return None

    def _ensure_started(self) -> None:
        if self.started:
            return
        import zmq

        self.zmq_context = zmq.Context.instance()
        self.socket = self.zmq_context.socket(zmq.PUB)
        endpoint = f"tcp://{self.demo_args.zmq_bind_host}:{int(self.demo_args.zmq_port)}"
        self.socket.bind(endpoint)
        time.sleep(max(0.0, float(self.demo_args.warmup)))
        start_msg = self.demo.build_command_message(start=True, stop=False, planner=True)
        idle = self.phases[0].start
        for _ in range(max(1, int(self.demo_args.start_bursts))):
            self.socket.send(start_msg)
            self._publish_planner(idle)
            time.sleep(0.12)
        self.started = True

    def _apply_anchor(self) -> None:
        if not self.runner_args.apply_anchor or not self.anchor:
            return
        if self.demo_kind == "box":
            self.demo.apply_box_anchor(self.demo_args, self.anchor, update_walk=True)
        else:
            self.demo.apply_ball_anchor(self.demo_args, self.anchor, update_walk=True)
            self._update_ball_ik_poses()
        self.phases = self.demo.make_demo_phases(self.demo_args)

    def _selected_phases(self, phases: tuple[str, ...]) -> list[Any]:
        wanted = set(phases)
        return [phase for phase in self.phases if phase.name in wanted]

    def _play_phase(self, phase: Any, *, command: dict[str, Any]) -> None:
        period = 1.0 / max(5.0, float(self.demo_args.rate))
        phase_msg = String()
        phase_msg.data = str(phase.name)
        t0 = time.monotonic()
        while rclpy.ok():
            if self.node.cancel_event.is_set():
                raise RuntimeError("primitive execution cancelled")
            elapsed = time.monotonic() - t0
            if elapsed >= float(phase.duration):
                break
            ratio = elapsed / max(1e-3, float(phase.duration))
            cmd = self.demo._interp_command(phase.start, phase.end, ratio)
            cmd = self._ball_workspace_servo(phase.name, cmd, command=command)
            cmd = self._ball_contact_servo(phase.name, ratio, cmd, command=command)
            self._publish_planner(cmd)
            self.node.phase_pub.publish(phase_msg)
            time.sleep(period)
        end_cmd = self._ball_workspace_servo(phase.name, phase.end, command=command)
        end_cmd = self._ball_contact_servo(phase.name, 1.0, end_cmd, command=command)
        self._publish_planner(end_cmd)
        self.node.phase_pub.publish(phase_msg)

    def _ball_workspace_servo(self, phase_name: str, cmd: Any, *, command: dict[str, Any]) -> Any:
        if self.demo_kind != "ball" or not phase_name.startswith("fine_align"):
            return cmd
        if phase_name == "fine_align_before_grasp" and not bool(self.demo_args.pregrasp_align_base):
            return self.demo.DemoCommand(
                mode=self.demo.LOCO_IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=cmd.facing,
                speed=cmd.speed,
                height=cmd.height,
                upper_body=cmd.upper_body,
                left_hand=cmd.left_hand,
                right_hand=cmd.right_hand,
            )
        point = self._live_target_point_base(command)
        if point is None:
            return cmd
        error_x = float(point[0]) - float(self.demo_args.align_target_x)
        error_y = float(point[1]) - float(self.demo_args.align_target_y)
        self._update_workspace_response([error_x, error_y])
        move_x = self.demo._signed_axis_command(
            error_x,
            tolerance=float(self.demo_args.align_x_tolerance),
            gain=float(self.demo_args.align_forward_gain),
            max_abs=min(float(self.demo_args.align_max_forward), 0.35),
            response_sign=self.workspace_response_sign[0],
        )
        move_y = self.demo._signed_axis_command(
            error_y,
            tolerance=float(self.demo_args.align_y_tolerance),
            gain=float(self.demo_args.align_lateral_gain),
            max_abs=min(float(self.demo_args.align_max_lateral), 0.35),
            response_sign=self.workspace_response_sign[1],
        )
        self.ik_metrics["workspace_error_xy"] = [round(error_x, 5), round(error_y, 5)]
        self.ik_metrics["workspace_command_xy"] = [round(move_x, 5), round(move_y, 5)]
        self.ik_metrics["workspace_response_sign"] = list(self.workspace_response_sign)
        if self.workspace_response_ref_error is None:
            self.workspace_response_ref_error = [error_x, error_y]
            self.workspace_response_ref_cmd = [move_x, move_y]
        mode = self.demo.LOCO_SLOW_WALK if max(abs(move_x), abs(move_y)) > 1e-3 else self.demo.LOCO_IDLE
        return self.demo.DemoCommand(
            mode=mode,
            movement=(move_x, move_y, 0.0),
            facing=cmd.facing,
            speed=float(self.demo_args.align_speed),
            height=cmd.height,
            upper_body=cmd.upper_body,
            left_hand=cmd.left_hand,
            right_hand=cmd.right_hand,
        )

    def _update_workspace_response(self, error: list[float]) -> None:
        if (
            self.workspace_response_ref_error is None
            or self.workspace_response_ref_cmd is None
        ):
            return
        min_cmd = float(self.demo_args.align_response_min_cmd)
        min_delta = float(self.demo_args.align_response_min_delta)
        sampled = False
        for axis in range(2):
            reference_cmd = float(self.workspace_response_ref_cmd[axis])
            delta = float(error[axis]) - float(self.workspace_response_ref_error[axis])
            if abs(reference_cmd) < min_cmd or abs(delta) < min_delta:
                continue
            sampled = True
            estimated = 1.0 if delta * reference_cmd > 0.0 else -1.0
            if estimated != self.workspace_response_sign[axis]:
                self.workspace_response_votes[axis] += 1
                if self.workspace_response_votes[axis] >= max(1, int(self.demo_args.align_response_flip_votes)):
                    self.workspace_response_sign[axis] = estimated
                    self.workspace_response_votes[axis] = 0
            else:
                self.workspace_response_votes[axis] = 0
        if sampled:
            self.workspace_response_ref_error = None
            self.workspace_response_ref_cmd = None

    def _build_ball_ik_solver(self) -> Any | None:
        if self.demo_kind != "ball" or not bool(getattr(self.demo_args, "ik_upper_body", False)):
            return None
        try:
            return self.demo.MujocoRightHandIK(
                self.demo_args.scene,
                qpos_path=str(self.demo_args.qpos_path),
                hand_body=str(self.demo_args.ik_hand_body),
                wrist_limits={
                    "right_wrist_roll_joint": (
                        -float(self.demo_args.ik_wrist_roll_limit),
                        float(self.demo_args.ik_wrist_roll_limit),
                    ),
                    "right_wrist_pitch_joint": (
                        float(self.demo_args.ik_wrist_pitch_min),
                        float(self.demo_args.ik_wrist_pitch_max),
                    ),
                    "right_wrist_yaw_joint": (
                        -float(self.demo_args.ik_wrist_yaw_limit),
                        float(self.demo_args.ik_wrist_yaw_limit),
                    ),
                },
            )
        except Exception:
            if bool(getattr(self.demo_args, "require_ik", False)):
                raise
            return None

    def _update_ball_ik_poses(self) -> None:
        if self.ik_solver is None or not isinstance(self.anchor, dict):
            return
        previous = getattr(self.demo_args, "ik_pose_overrides", {})
        self.demo_args.ik_pose_overrides = {}
        fallback = self.demo._upper_body_poses(self.demo_args)
        self.demo_args.ik_pose_overrides = previous
        poses, errors = self.ik_solver.solve_pick_place_poses(self.anchor, fallback, self.demo_args)
        critical = [float(errors[name]) for name in ("grasp", "lift", "secure") if name in errors]
        worst = max(critical) if critical else float("inf")
        accepted = worst <= float(self.demo_args.ik_max_error)
        self.ik_metrics = {
            "pose_errors": {name: round(float(value), 5) for name, value in errors.items()},
            "critical_max_error": round(worst, 5),
            "poses_applied": accepted,
        }
        if accepted:
            self.demo_args.ik_pose_overrides = poses
        elif bool(getattr(self.demo_args, "require_ik", False)):
            raise RuntimeError(f"ball IK critical error {worst:.4f} exceeds {self.demo_args.ik_max_error:.4f}")

    def _ball_contact_servo(
        self,
        phase_name: str,
        ratio: float,
        cmd: Any,
        *,
        command: dict[str, Any],
    ) -> Any:
        active = {
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
            "low_hold_ball",
            "lift_ball",
            "secure_ball",
            "move_to_place",
            "lower_to_place",
        }
        if (
            self.demo_kind != "ball"
            or self.ik_solver is None
            or not bool(getattr(self.demo_args, "contact_servo", False))
            or phase_name not in active
        ):
            return cmd
        center = self._live_target_point_base(command)
        if center is None and isinstance(self.anchor, dict):
            center = self.anchor.get("ball_point_base")
        if not isinstance(center, (list, tuple)) or len(center) < 3:
            return cmd
        center = [float(center[0]), float(center[1]), float(center[2])]
        payload = command.get("command") if isinstance(command.get("command"), dict) else {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        # The learned residual is local to the contact pose and cannot move the
        # world anchor or exceed the profile's safe capture envelope.
        center[0] += _bounded_param(params.get("contact_x_delta_m"), -0.025, 0.025)
        center[2] += _bounded_param(params.get("contact_z_delta_m"), -0.015, 0.015)
        radius = float((self.anchor or {}).get("ball_radius", self.demo_args.ball_radius))
        table_phase = phase_name in {
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
        }
        effective_radius = radius
        if phase_name == "lower_to_ball_open":
            effective_radius += float(self.demo_args.open_shell_radius)
            center[2] += float(self.demo_args.open_shell_z)
        elif phase_name == "capture_ball_contact":
            shell = 1.0 - float(self.demo._smoothstep(ratio))
            effective_radius += float(self.demo_args.capture_shell_radius) * shell
            center[2] += float(self.demo_args.capture_shell_z) * shell
        elif phase_name == "lift_ball":
            ramp_start = float(self.demo_args.servo_lift_ramp_start)
            ramp = float(self.demo._smoothstep((ratio - ramp_start) / max(1e-3, 1.0 - ramp_start)))
            z_min = float(self.demo_args.servo_lift_z_lead)
            z_max = float(self.demo_args.servo_lift_z_max_lead)
            center[0] += float(self.demo_args.servo_lift_x_lead)
            center[2] += z_min + (z_max - z_min) * ramp
        elif phase_name == "secure_ball":
            center[2] += float(self.demo_args.servo_hold_z_lead)
        elif phase_name in {"move_to_place", "lower_to_place"}:
            destination = self._live_entity_point_base(command, "destination_ref")
            if destination is not None:
                direction = [float(destination[index]) - center[index] for index in range(3)]
                max_lead = float(
                    self.demo_args.servo_place_lead
                    if phase_name == "lower_to_place"
                    else self.demo_args.servo_transfer_lead
                )
                if phase_name == "lower_to_place":
                    direction[2] = min(direction[2], -float(self.demo_args.servo_place_down_lead))
                else:
                    direction[2] = max(direction[2], float(self.demo_args.servo_hold_z_lead))
                norm = math.sqrt(sum(value * value for value in direction))
                scale = min(1.0, max_lead / norm) if norm > 1e-6 else 0.0
                center = [center[index] + direction[index] * scale for index in range(3)]
        seed = self.ik_solver.live_upper_body_pose(cmd.upper_body)
        try:
            solved, error = self.ik_solver.solve_contact_pose(
                center,
                effective_radius,
                seed,
                self.demo_args,
                hand_pose=cmd.right_hand,
                table_contact=table_phase,
            )
        except Exception as exc:
            self.ik_metrics["servo_last_error"] = str(exc)
            self.ik_metrics["servo_fallback_cached"] = self.last_contact_command is not None
            return self.last_contact_command if self.last_contact_command is not None else cmd
        self.ik_metrics["servo_phase"] = phase_name
        self.ik_metrics["servo_ik_error"] = round(float(error), 5)
        if float(error) > float(self.demo_args.servo_ik_max_error):
            self.ik_metrics["servo_rejected"] = True
            self.ik_metrics["servo_fallback_cached"] = self.last_contact_command is not None
            return self.last_contact_command if self.last_contact_command is not None else cmd
        self.ik_metrics["servo_rejected"] = False
        contact_command = self.demo.DemoCommand(
            mode=cmd.mode,
            movement=cmd.movement,
            facing=cmd.facing,
            speed=cmd.speed,
            height=cmd.height,
            upper_body=solved,
            left_hand=cmd.left_hand,
            right_hand=cmd.right_hand,
        )
        self.last_contact_command = contact_command
        return contact_command

    def _live_target_point_base(self, command: dict[str, Any]) -> list[float] | None:
        return self._live_entity_point_base(command, "target_ref")

    def _live_entity_point_base(self, command: dict[str, Any], ref_key: str) -> list[float] | None:
        if self.ik_solver is None:
            return None
        payload = command.get("command") if isinstance(command.get("command"), dict) else {}
        ref = payload.get(ref_key) if isinstance(payload.get(ref_key), dict) else {}
        qpos = self.ik_solver._live_qpos()
        if qpos is None:
            return None
        self.ik_solver.data.qpos[:] = qpos
        self.demo.mujoco.mj_forward(self.ik_solver.model, self.ik_solver.data)
        point_world = None
        for key, object_type, positions in (
            ("site_name", self.demo.mujoco.mjtObj.mjOBJ_SITE, self.ik_solver.data.site_xpos),
            ("body_name", self.demo.mujoco.mjtObj.mjOBJ_BODY, self.ik_solver.data.xpos),
            ("geom_name", self.demo.mujoco.mjtObj.mjOBJ_GEOM, self.ik_solver.data.geom_xpos),
        ):
            name = str(ref.get(key) or "")
            if not name:
                continue
            object_id = self.demo.mujoco.mj_name2id(self.ik_solver.model, object_type, name)
            if object_id >= 0:
                point_world = positions[object_id]
                break
        if point_world is None:
            return None
        base_id = int(self.ik_solver.base_body_id)
        base_pos = self.ik_solver.data.xpos[base_id]
        base_rot = self.ik_solver.data.xmat[base_id].reshape(3, 3)
        point = base_rot.T @ (point_world - base_pos)
        return [float(value) for value in point]

    def _publish_planner(self, cmd: Any) -> None:
        now = time.monotonic()
        dt = 0.0 if self.last_send_time is None else now - self.last_send_time
        upper_vel = self.demo._velocity(
            cmd.upper_body,
            self.last_upper_body,
            dt,
            float(self.demo_args.max_upper_body_velocity),
        )
        msg = self.demo.build_planner_message(
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
        if self.socket is None:
            raise RuntimeError("ZMQ socket not initialized")
        self.socket.send(msg)
        self.last_upper_body = list(cmd.upper_body)
        self.last_send_time = now

    def _publish_phase_status(self, command: dict[str, Any], phase_name: str, status: str) -> None:
        payload = {
            "schema": "sonic_skill_primitive_status_v0",
            "event": "primitive_phase",
            "status": status,
            "backend": self.backend_name,
            "task_id": command.get("task_id"),
            "action_id": command.get("action_id"),
            "skill_name": command.get("skill_name"),
            "target_id": command.get("target_id"),
            "demo_kind": self.demo_kind,
            "phase": phase_name,
            "stamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.node.status_pub.publish(msg)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None


def _load_demo_module(demo_kind: str) -> Any:
    if demo_kind == "box":
        import box_grasp_demo as demo
    else:
        import ball_pick_place_demo as demo
    return demo


def _demo_args(demo_kind: str, args: argparse.Namespace) -> argparse.Namespace:
    demo = _load_demo_module(demo_kind)
    argv = [
        "world_model_primitive_runner",
        "--no-hold",
        "--warmup",
        str(float(args.warmup)),
        "--rate",
        str(float(args.rate)),
        "--zmq-bind-host",
        str(args.zmq_bind_host),
        "--zmq-port",
        str(int(args.zmq_port)),
        "--start-bursts",
        str(int(args.start_bursts)),
        "--max-upper-body-velocity",
        str(float(args.max_upper_body_velocity)),
    ]
    if args.scene:
        argv.extend(["--scene", str(args.scene)])
    argv.extend(str(item) for item in args.demo_arg)
    saved = sys.argv[:]
    try:
        sys.argv = argv
        return demo.parse_args()
    finally:
        sys.argv = saved


def _teacher_assist_evidence(command: dict[str, Any]) -> dict[str, Any]:
    """Evidence for simulator-only contact-lock demonstrations.

    The caller records ``teacher_assisted`` in metrics and physical leaderboard
    ingestion rejects such episodes.  This evidence is therefore usable for
    imitation/AWR initialization without being confused with physical truth.
    """
    effects = command.get("effects") if isinstance(command.get("effects"), list) else []
    return {
        "schema": "sonic_skill_effect_evidence_v0",
        "passed": bool(effects),
        "source": "teacher_attach",
        "reason": "simulator-only pregrasp attachment teacher evidence",
        "effects": {str(effect): {"passed": True, "teacher_assisted": True} for effect in effects},
    }


def _phase_names(command: dict[str, Any], *, demo_kind: str) -> tuple[str, ...]:
    raw = command.get("phase_names")
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw if str(item))
    skill_name = str(command.get("skill_name") or "")
    mapping = BOX_PHASE_FALLBACKS if demo_kind == "box" else BALL_PHASE_FALLBACKS
    return tuple(mapping.get(skill_name, ()))


def _generic_anchor_to_demo(anchor: dict[str, Any] | None, *, demo_kind: str, target_id: str = "") -> dict[str, Any] | None:
    if not isinstance(anchor, dict):
        return None
    if demo_kind == "ball" and "ball_point_base" in anchor:
        return anchor
    if demo_kind == "box" and "box_point_base" in anchor:
        return anchor
    records = anchor.get("objects") if isinstance(anchor.get("objects"), list) else [anchor]
    objects = [item for item in records if isinstance(item, dict)]
    target = _object_by_id(objects, target_id) or _first_task_object(objects, demo_kind=demo_kind)
    if target is None:
        return None
    place = _first_category(objects, "place_target")
    out: dict[str, Any] = {
        "scene": anchor.get("scene"),
        "source": "world_model_primitive_runner",
        "frame_id": anchor.get("frame_id", "map"),
        "grasp": target.get("grasp") or (target.get("properties") or {}).get("grasp") or {},
    }
    pose_base = _pose_position(target.get("pose_base") or target.get("point_base"))
    pose_map = _pose_position(target.get("pose_map") or target.get("center_map"))
    pose_camera = _pose_position(target.get("pose_camera") or target.get("point_camera_depth"))
    if demo_kind == "box":
        out["box_name"] = str(target.get("object_id") or target.get("id") or "box")
        if pose_base:
            out["box_point_base"] = pose_base
        if pose_map:
            out["box_center_map"] = pose_map
        if pose_camera:
            out["box_point_camera_depth"] = pose_camera
        out["box_size"] = _shape_size(target) or [0.24, 0.16, 0.16]
    else:
        out["ball_name"] = str(target.get("object_id") or target.get("id") or "ball")
        if pose_base:
            out["ball_point_base"] = pose_base
        if pose_map:
            out["ball_center_map"] = pose_map
        if pose_camera:
            out["ball_point_camera_depth"] = pose_camera
        out["ball_radius"] = _shape_radius(target) or 0.045
        if place is not None:
            place_base = _pose_position(place.get("pose_base") or place.get("point_base"))
            place_map = _pose_position(place.get("pose_map") or place.get("center_map"))
            if place_base:
                out["place_point_base"] = place_base
            if place_map:
                out["place_center_map"] = place_map
        else:
            # Pick-only stages still initialize the shared ball IK pose set.
            # A same-point placeholder is never dispatched as a place action.
            if pose_base:
                out["place_point_base"] = list(pose_base)
            if pose_map:
                out["place_center_map"] = list(pose_map)
    return out


def _first_task_object(objects: list[dict[str, Any]], *, demo_kind: str) -> dict[str, Any] | None:
    excluded = {"place_target", "navigation_goal", "table", "counter", "support_surface"}
    for obj in objects:
        category = str(obj.get("category") or obj.get("object_category") or "")
        if category in excluded:
            continue
        if demo_kind == "box" and category in {"box", "package", "cube", "small_box", "snack_box"}:
            return obj
        if demo_kind == "ball" and category not in {"box", "package", "small_box", "snack_box"}:
            return obj
    for obj in objects:
        if str(obj.get("category") or "") not in excluded:
            return obj
    return None


def _object_by_id(objects: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    if not object_id:
        return None
    for obj in objects:
        if str(obj.get("object_id") or obj.get("id") or "") == object_id:
            return obj
    return None


def _first_category(objects: list[dict[str, Any]], category: str) -> dict[str, Any] | None:
    for obj in objects:
        if str(obj.get("category") or obj.get("object_category") or "") == category:
            return obj
    return None


def _pose_position(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("position") or value.get("xyz") or value.get("point")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return out if all(item == item for item in out) else None


def _shape_size(obj: dict[str, Any]) -> list[float] | None:
    shape = obj.get("shape") if isinstance(obj.get("shape"), dict) else {}
    value = shape.get("size") or obj.get("size")
    return _pose_position(value)


def _shape_radius(obj: dict[str, Any]) -> float | None:
    shape = obj.get("shape") if isinstance(obj.get("shape"), dict) else {}
    value = shape.get("radius", obj.get("radius"))
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    return radius if radius > 0.0 else None


def _bounded_param(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, number))


def _parse_json_or_none(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WorldModelPrimitiveRunner(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
