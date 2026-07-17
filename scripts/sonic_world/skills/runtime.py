from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .specs import SkillGraph, SkillSpec


@dataclass(frozen=True)
class PhaseBinding:
    skill_name: str
    phase_names: tuple[str, ...]
    monitor_events: tuple[str, ...] = ()
    recovery_events: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "phase_names": list(self.phase_names),
            "monitor_events": list(self.monitor_events),
            "recovery_events": list(self.recovery_events),
        }


@dataclass(frozen=True)
class RuntimePlan:
    task_id: str
    demo_kind: str
    bindings: list[PhaseBinding]
    metadata: dict[str, Any] = field(default_factory=dict)

    def phases_for_skill(self, skill_name: str) -> tuple[str, ...]:
        for binding in self.bindings:
            if binding.skill_name == skill_name:
                return binding.phase_names
        return ()

    def skill_for_phase(self, phase_name: str) -> str | None:
        for binding in self.bindings:
            if phase_name in binding.phase_names:
                return binding.skill_name
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "demo_kind": self.demo_kind,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "metadata": self.metadata,
        }


_BOX_PHASES: dict[str, PhaseBinding] = {
    "navigate.approach_object": PhaseBinding(
        "navigate.approach_object",
        ("walk_two_steps",),
        monitor_events=("approach_reached", "approach_retry", "approach_failed"),
        recovery_events=("retry_walk", "replan_standoff"),
    ),
    "manip.align_workspace": PhaseBinding(
        "manip.align_workspace",
        ("settle_before_grasp", "arms_open_table"),
        monitor_events=("anchor_retrack", "ik_pose_update"),
        recovery_events=("retrack_anchor", "micro_adjust_base"),
    ),
    "manip.bimanual_clamp": PhaseBinding(
        "manip.bimanual_clamp",
        ("reach_table_open", "forearm_clamp_box"),
        monitor_events=("box_contact", "clamp_width", "ik_error"),
        recovery_events=("reopen_arms", "adjust_clamp_y", "retry_reach"),
    ),
    "manip.lift_object": PhaseBinding(
        "manip.lift_object",
        ("lift_box_from_table", "squeeze_box_secure", "bring_box_to_chest", "carry_settle"),
        monitor_events=("box_lifted", "box_lift_failed", "balance"),
        recovery_events=("squeeze_more", "lower_and_regrasp"),
    ),
}


_BALL_PHASES: dict[str, PhaseBinding] = {
    "navigate.approach_object": PhaseBinding(
        "navigate.approach_object",
        ("walk_to_table", "fine_align_to_ball"),
        monitor_events=("approach_reached", "workspace_residual", "approach_retry"),
        recovery_events=("retry_walk", "increase_standoff", "retrack_anchor"),
    ),
    "manip.align_workspace": PhaseBinding(
        "manip.align_workspace",
        ("settle_before_pick", "hand_high_ready", "fine_align_before_grasp"),
        monitor_events=("workspace_aligned", "workspace_residual", "ik_pose_update"),
        recovery_events=("micro_step_base", "raise_hand_clear", "retrack_anchor"),
    ),
    "manip.single_hand_pinch": PhaseBinding(
        "manip.single_hand_pinch",
        (
            "approach_from_above",
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
        ),
        monitor_events=("finger_contact_error", "capture_retry", "wrist_target", "grasp_geometry"),
        recovery_events=("retry_capture", "adjust_close_ratio", "switch_side_offset"),
    ),
    "manip.side_grasp": PhaseBinding(
        "manip.side_grasp",
        (
            "approach_from_above",
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
        ),
        monitor_events=("finger_contact_error", "capture_retry", "wrist_target", "grasp_geometry"),
        recovery_events=("retry_capture", "adjust_close_ratio", "switch_side_offset"),
    ),
    "manip.top_grasp": PhaseBinding(
        "manip.top_grasp",
        (
            "approach_from_above",
            "lower_to_ball_open",
            "capture_ball_contact",
            "close_on_ball",
            "squeeze_ball_secure",
        ),
        monitor_events=("finger_contact_error", "capture_retry", "wrist_target", "grasp_geometry"),
        recovery_events=("retry_capture", "adjust_close_ratio", "switch_side_offset"),
    ),
    "manip.lift_object": PhaseBinding(
        "manip.lift_object",
        ("low_hold_ball", "lift_ball", "secure_ball"),
        monitor_events=("ball_lifted", "ball_lift_failed", "contact_quality", "wrist_actual"),
        recovery_events=("squeeze_more", "lower_and_regrasp", "hold_low_if_unstable"),
    ),
    "manip.transport_object": PhaseBinding(
        "manip.transport_object",
        ("move_to_place",),
        monitor_events=("object_in_hand", "place_target_reachable"),
        recovery_events=("hold_pose", "replan_transfer"),
    ),
    "manip.place_object": PhaseBinding(
        "manip.place_object",
        ("lower_to_place",),
        monitor_events=("target_clear", "object_near_destination"),
        recovery_events=("raise_and_retry_place", "choose_nearby_place_region"),
    ),
    "manip.release": PhaseBinding(
        "manip.release",
        ("release_ball", "retreat_hand", "hold_done"),
        monitor_events=("hand_free", "object_released"),
        recovery_events=("open_hand_again", "retreat_hand"),
    ),
}


_NAV_PHASES: dict[str, PhaseBinding] = {
    "navigate.goto": PhaseBinding(
        "navigate.goto",
        ("navigate_to_goal",),
        monitor_events=("goal_received", "global_plan_ready", "goal_reached", "stuck_detected"),
        recovery_events=("replan_global_path", "clear_local_costmap", "relax_goal_tolerance"),
    ),
    "navigate.approach_object": PhaseBinding(
        "navigate.approach_object",
        ("approach_object",),
        monitor_events=("object_localized", "approach_reached", "approach_failed"),
        recovery_events=("relocalize_object", "replan_approach_pose"),
    ),
}


def runtime_plan_for_graph(graph: SkillGraph, *, demo_kind: str | None = None) -> RuntimePlan:
    kind = demo_kind or _infer_demo_kind(graph)
    mapping = _binding_table(kind)
    bindings: list[PhaseBinding] = []
    missing: list[str] = []
    for step in graph.steps:
        binding = mapping.get(step.name)
        if binding is None:
            missing.append(step.name)
            continue
        bindings.append(_merge_recovery(binding, step))
    return RuntimePlan(
        task_id=graph.task_id,
        demo_kind=kind,
        bindings=bindings,
        metadata={
            "missing_skills": missing,
            "skill_count": len(graph.steps),
            "bound_skill_count": len(bindings),
        },
    )


def phase_to_skill_index(plan: RuntimePlan) -> dict[str, str]:
    out: dict[str, str] = {}
    for binding in plan.bindings:
        for phase in binding.phase_names:
            out[phase] = binding.skill_name
    return out


def skill_summary(graph: SkillGraph) -> str:
    return " -> ".join(step.name for step in graph.steps)


def _infer_demo_kind(graph: SkillGraph) -> str:
    template = graph.metadata.get("task_template")
    if template == "navigation":
        return "navigation"
    step_names = {step.name for step in graph.steps}
    if "manip.bimanual_clamp" in step_names:
        return "box"
    if "manip.single_hand_pinch" in step_names:
        return "ball"
    category = graph.metadata.get("object_category")
    if category == "box":
        return "box"
    if category == "navigation_goal":
        return "navigation"
    return "ball"


def _binding_table(kind: str) -> dict[str, PhaseBinding]:
    if kind == "box":
        return _BOX_PHASES
    if kind in {"nav", "navigation", "navigation_goal"}:
        return _NAV_PHASES
    return _BALL_PHASES


def _merge_recovery(binding: PhaseBinding, step: SkillSpec) -> PhaseBinding:
    recovery = tuple(dict.fromkeys((*binding.recovery_events, *tuple(step.recovery))))
    return PhaseBinding(
        skill_name=binding.skill_name,
        phase_names=binding.phase_names,
        monitor_events=binding.monitor_events,
        recovery_events=recovery,
    )
