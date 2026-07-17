from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid


REPORT_DIR = Path("reports/rollouts")


PHASE_STAGE: dict[str, str] = {
    "stand_ready": "setup",
    "raise_hand_clear": "workspace",
    "walk_two_steps": "approach",
    "walk_to_table": "approach",
    "approach_object": "approach",
    "fine_align_to_ball": "workspace",
    "fine_align_before_grasp": "workspace",
    "settle_before_grasp": "workspace",
    "settle_before_pick": "workspace",
    "arms_open_table": "workspace",
    "hand_high_ready": "workspace",
    "reach_table_open": "grasp",
    "approach_from_above": "grasp",
    "lower_to_ball_open": "grasp",
    "capture_ball_contact": "grasp",
    "close_on_ball": "grasp",
    "forearm_clamp_box": "grasp",
    "squeeze_box_secure": "grasp",
    "squeeze_ball_secure": "grasp",
    "lift_box_from_table": "lift",
    "lift_ball": "lift",
    "secure_ball": "lift",
    "bring_box_to_chest": "lift",
    "carry_settle": "lift",
    "carry_walk_forward": "transport",
    "move_to_place": "transport",
    "lower_to_place": "place",
    "release_ball": "place",
    "retreat_hand": "place",
    "hold_box_clamped": "done",
    "hold_done": "done",
    "done": "done",
}


PHASE_SKILL: dict[str, str] = {
    "walk_two_steps": "navigate.approach_object",
    "walk_to_table": "navigate.approach_object",
    "fine_align_to_ball": "manip.align_workspace",
    "fine_align_before_grasp": "manip.align_workspace",
    "settle_before_grasp": "manip.align_workspace",
    "settle_before_pick": "manip.align_workspace",
    "arms_open_table": "manip.align_workspace",
    "hand_high_ready": "manip.align_workspace",
    "reach_table_open": "manip.bimanual_clamp",
    "forearm_clamp_box": "manip.bimanual_clamp",
    "approach_from_above": "manip.single_hand_pinch",
    "lower_to_ball_open": "manip.single_hand_pinch",
    "capture_ball_contact": "manip.single_hand_pinch",
    "close_on_ball": "manip.single_hand_pinch",
    "squeeze_box_secure": "manip.bimanual_clamp",
    "squeeze_ball_secure": "manip.single_hand_pinch",
    "lift_box_from_table": "manip.lift_object",
    "lift_ball": "manip.lift_object",
    "secure_ball": "manip.lift_object",
    "bring_box_to_chest": "manip.lift_object",
    "carry_settle": "manip.lift_object",
    "carry_walk_forward": "navigate.approach_object",
    "move_to_place": "manip.transport_object",
    "lower_to_place": "manip.place_object",
    "release_ball": "manip.release",
    "retreat_hand": "manip.release",
}


@dataclass
class RolloutLogger:
    demo_kind: str
    task_id: str
    scene: str | None = None
    path: Path | None = None
    run_id: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if os.environ.get("SONIC_DISABLE_ROLLOUT_LOG", "").lower() in {"1", "true", "yes"}:
            self.enabled = False
        if self.run_id is None:
            self.run_id = os.environ.get("SONIC_ROLLOUT_ID") or self._new_run_id()
        if self.path is None:
            raw_path = os.environ.get("SONIC_ROLLOUT_LOG")
            self.path = Path(raw_path).expanduser() if raw_path else default_rollout_log_path(self.demo_kind, self.run_id)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event: str,
        *,
        phase: str | None = None,
        status: str | None = None,
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "stamp": time.time(),
            "monotonic": time.monotonic(),
            "run_id": self.run_id,
            "demo_kind": self.demo_kind,
            "task_id": self.task_id,
            "scene": self.scene,
            "event": event,
            "phase": phase,
            "primitive_stage": stage_for_phase(phase),
            "skill_name": skill_for_phase(phase),
            "status": status,
            "reason": reason,
            "metrics": _jsonable(metrics or {}),
            "metadata": _jsonable({**self.metadata, **(metadata or {})}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    def phase_start(self, phase: str, *, duration: float | None = None, metadata: dict[str, Any] | None = None) -> None:
        metrics = {"duration": float(duration)} if duration is not None else {}
        self.log_event("phase_start", phase=phase, status="running", metrics=metrics, metadata=metadata)

    def phase_end(
        self,
        phase: str,
        *,
        elapsed: float | None = None,
        status: str = "success",
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(metrics or {})
        if elapsed is not None:
            payload["elapsed"] = float(elapsed)
        self.log_event("phase_end", phase=phase, status=status, reason=reason, metrics=payload)

    def close(self, *, status: str = "closed") -> None:
        self.log_event("logger_close", status=status)

    @staticmethod
    def _new_run_id() -> str:
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def default_rollout_log_path(demo_kind: str, run_id: str | None = None) -> Path:
    run = run_id or RolloutLogger._new_run_id()
    return REPORT_DIR / f"{_safe_name(demo_kind)}_{_safe_name(run)}.jsonl"


def logger_from_args(
    args: Any,
    *,
    demo_kind: str,
    task_id: str,
    scene: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RolloutLogger:
    enabled = not bool(getattr(args, "no_rollout_log", False))
    path = getattr(args, "rollout_log", None)
    run_id = getattr(args, "rollout_id", None)
    return RolloutLogger(
        demo_kind=demo_kind,
        task_id=task_id,
        scene=scene,
        path=Path(path).expanduser() if path else None,
        run_id=run_id,
        enabled=enabled,
        metadata=dict(metadata or {}),
    )


def add_rollout_log_args(parser: Any) -> None:
    parser.add_argument("--rollout-log", help="JSONL rollout log path. Defaults to reports/rollouts/<demo>_<run>.jsonl.")
    parser.add_argument("--rollout-id", help="Stable rollout id for grouping logs across launcher/demo processes.")
    parser.add_argument("--no-rollout-log", action="store_true", help="Disable rollout JSONL logging.")


def stage_for_phase(phase: str | None) -> str | None:
    if not phase:
        return None
    return PHASE_STAGE.get(phase, "unknown")


def skill_for_phase(phase: str | None) -> str | None:
    if not phase:
        return None
    return PHASE_SKILL.get(phase)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
