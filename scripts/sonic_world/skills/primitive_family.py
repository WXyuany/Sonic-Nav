from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrimitiveProfile:
    skill_name: str
    family: str
    handler: str
    required_effects: tuple[str, ...]
    runtime_overrides: dict[str, Any] = field(default_factory=dict)


PRIMITIVE_PROFILES: dict[str, PrimitiveProfile] = {
    "navigate.approach_object": PrimitiveProfile(
        "navigate.approach_object", "navigation", "demo_locomotion_phase_runtime", ("robot_near_object",)
    ),
    "manip.align_workspace": PrimitiveProfile(
        "manip.align_workspace", "workspace", "workspace_alignment_primitive", ("object_in_hand_workspace",)
    ),
    "manip.single_hand_pinch": PrimitiveProfile(
        "manip.single_hand_pinch",
        "single_hand_grasp",
        "contact_grasp_primitive",
        ("object_contact_ready",),
        {"grasp_wrist_pitch": -0.12},
    ),
    "manip.side_grasp": PrimitiveProfile(
        "manip.side_grasp",
        "single_hand_grasp",
        "contact_grasp_primitive",
        ("object_contact_ready",),
        {"grasp_wrist_pitch": -0.05, "palm_pocket_table_z_radius": -0.05},
    ),
    "manip.top_grasp": PrimitiveProfile(
        "manip.top_grasp",
        "single_hand_grasp",
        "contact_grasp_primitive",
        ("object_contact_ready",),
        {"grasp_wrist_pitch": -0.26, "palm_pocket_table_z_radius": -0.35},
    ),
    "manip.bimanual_clamp": PrimitiveProfile(
        "manip.bimanual_clamp", "bimanual_grasp", "contact_grasp_primitive", ("object_contact_ready",)
    ),
    "manip.lift_object": PrimitiveProfile(
        "manip.lift_object",
        "lift",
        "lift_stability_primitive",
        ("object_in_hand",),
        {
            # Lift starts after a contact-only grasp. Bias toward retaining that
            # contact before increasing vertical lead in the physical runner.
            "squeeze_close_ratio": 0.74,
            "hold_close_ratio": 0.82,
            "servo_lift_z_lead": 0.035,
            "low_hold_duration": 0.8,
            "low_hold_min_contacts": 1,
            "lift_z": 0.18,
            "lift_duration": 1.6,
        },
    ),
    "manip.transport_object": PrimitiveProfile(
        "manip.transport_object", "transport", "carry_transfer_primitive", ("object_near_destination",)
    ),
    "manip.place_object": PrimitiveProfile(
        "manip.place_object", "place", "place_primitive", ("object_on_destination",)
    ),
    "manip.release": PrimitiveProfile("manip.release", "release", "release_primitive", ("hand_free",)),
}


def primitive_profile(skill_name: str) -> PrimitiveProfile | None:
    return PRIMITIVE_PROFILES.get(str(skill_name))


def apply_profile_to_namespace(skill_name: str, params: dict[str, Any], namespace: Any) -> dict[str, Any]:
    """Apply bounded task-space profile parameters to a legacy runtime namespace."""

    profile = primitive_profile(skill_name)
    applied: dict[str, Any] = {}
    values = dict(profile.runtime_overrides) if profile is not None else {}
    aliases = {
        "close_ratio": "close_ratio",
        "grasp_close_ratio": "close_ratio",
        "wrist_pitch": "grasp_wrist_pitch",
        "grasp_wrist_pitch": "grasp_wrist_pitch",
        "lift_height": "ik_lift_z",
        "lift_z": "ik_lift_z",
        "radius": "ball_radius",
        "aperture": "ball_radius",
    }
    for key, value in params.items():
        target = aliases.get(str(key), str(key))
        if hasattr(namespace, target) and isinstance(value, (int, float, bool)):
            values[target] = value
    for key, value in values.items():
        if hasattr(namespace, key):
            setattr(namespace, key, value)
            applied[key] = value
    return applied
