from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .heuristic import HeuristicSkillPolicy
from .backend import policy_action_from_dict
from .schema import PolicyAction
from ..planners import PlanningResult
from ..rl.hybrid_ppo import HybridRecurrentActorCritic


RECOVERY = ("continue", "reobserve", "micro_adjust", "replan", "abort")


class HybridPPOPolicyBackend:
    """Run a custom PPO checkpoint and apply bounded residuals to a teacher action."""

    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint).expanduser()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "sonic_world_model_hybrid_ppo_v0":
            raise ValueError(f"unsupported hybrid PPO checkpoint: {payload.get('schema')!r}")
        self.model = HybridRecurrentActorCritic()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.policy_id = self.path.stem
        self.uses_visual_context = bool(payload.get("visual_context", False))
        deployment = payload.get("visual_deployment") if isinstance(payload.get("visual_deployment"), dict) else {}
        self.visual_deployment_status = str(deployment.get("status") or "not_requested")
        training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
        self.training_skills = {str(item) for item in training.get("skills", []) if str(item)}
        if self.uses_visual_context and self.visual_deployment_status != "eligible_for_ab" and os.environ.get("SONIC_ALLOW_SHADOW_VISUAL_POLICY") != "1":
            raise ValueError(
                "visual policy checkpoint is not eligible_for_ab; complete the visual training gate before physical deployment"
            )
        self.teacher = HeuristicSkillPolicy()

    def act(self, result: PlanningResult) -> PolicyAction:
        payload = self.teacher.act(result).to_dict()
        entity, context = _features(result, include_visual=self.uses_visual_context)
        _visual_entity, visual_context = _features(result, include_visual=True)
        with torch.inference_mode():
            residual, recovery, _log_prob, _value, _hidden = self.model.act(entity, context, deterministic=True)
        action = residual[0, 0].tanh().tolist()
        mode = RECOVERY[int(recovery[0, 0])]
        recovery_context = result.request.metadata.get("runtime_recovery") if isinstance(result.request.metadata, dict) else {}
        _apply_residual(
            payload,
            action,
            mode,
            allow_base_residual=not _manipulation_only(self.training_skills),
            allow_grasp_residual=_allows_skill(self.training_skills, "manip.side_grasp"),
            allow_recovery_override=bool(result.recovery_plan.actions),
        )
        metadata = dict(payload.get("metadata") or {})
        metadata["policy_backend"] = {
            "type": "hybrid_ppo",
            "policy_id": self.policy_id,
            "checkpoint": str(self.path),
            "recovery_mode": mode,
            "residual": [round(float(item), 5) for item in action],
            "skill_runtime_overrides": _skill_runtime_overrides(
                action,
                recovery_context=recovery_context,
                training_skills=self.training_skills,
            ),
            "skill_runtime_override_mode": "residual_additive",
            "observation": {
                "entity": entity[0, 0].tolist(),
                "context": context[0, 0].tolist(),
            },
            "visual_observation": {
                "entity": entity[0, 0].tolist(),
                "context": visual_context[0, 0].tolist(),
            },
            "visual_context": self.uses_visual_context,
            "visual_deployment_status": self.visual_deployment_status,
        }
        payload["metadata"] = metadata
        payload["policy_id"] = self.policy_id
        return policy_action_from_dict(payload)


def _features(result: PlanningResult, *, include_visual: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    obj = result.world.get_object(result.request.object_id) if result.request.object_id else result.world.primary_object()
    target = result.world.get_object(result.request.target_id) if result.request.target_id else None
    entity = torch.zeros(1, 1, 2, 12)
    for index, item in enumerate((obj, target)):
        if item is None:
            continue
        pose = item.pose_base or item.pose_map
        if pose is not None:
            entity[0, 0, index, :3] = torch.tensor(pose.position)
        entity[0, 0, index, 3] = float(item.shape.radius or 0.0)
        if item.shape.size:
            entity[0, 0, index, 4:7] = torch.tensor(item.shape.size)
    context = torch.zeros(1, 1, 24)
    context[0, 0, :3] = entity[0, 0, 0, :3]
    context[0, 0, 3:6] = entity[0, 0, 1, :3] - entity[0, 0, 0, :3]
    context[0, 0, 6] = 1.0 if result.world.robot.stable else 0.0
    context[0, 0, 7] = float(len(result.recovery_plan.actions))
    recovery = result.request.metadata.get("runtime_recovery") if isinstance(result.request.metadata, dict) else {}
    if isinstance(recovery, dict):
        context[0, 0, 8] = _skill_failure_code(str(recovery.get("failed_skill") or ""))
        context[0, 0, 9] = _handler_code(str(recovery.get("handler") or ""))
        context[0, 0, 10] = min(1.0, max(0.0, float(recovery.get("attempt") or 0) / 3.0))
        context[0, 0, 11] = 1.0
    if include_visual:
        context[0, 0, 12:18] = torch.tensor(_visual_features(obj))
        context[0, 0, 18:24] = torch.tensor(_visual_features(target))
    return entity, context


def _visual_features(obj: Any) -> list[float]:
    if obj is None:
        return [0.0] * 6
    props = obj.properties if isinstance(getattr(obj, "properties", None), dict) else {}
    source = str(getattr(obj, "source", "") or "").lower()
    uncertainty = props.get("uncertainty") if isinstance(props.get("uncertainty"), dict) else {}
    confidence = _clamp01(props.get("confidence"), default=0.0)
    qwen_or_vlm = 1.0 if any(token in source for token in ("qwen", "vlm", "dino")) else 0.0
    tracked = 1.0 if props.get("tracking_id") else 0.0
    depth_mad = max(0.0, _float(uncertainty.get("depth_mad_m"), 1.0))
    depth_quality = 1.0 / (1.0 + 20.0 * depth_mad)
    samples = min(1.0, max(0.0, _float(uncertainty.get("depth_sample_count"), 0.0) / 49.0))
    pose = 1.0 if getattr(obj, "pose_base", None) or getattr(obj, "pose_map", None) else 0.0
    return [confidence, qwen_or_vlm, tracked, depth_quality, samples, pose]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any, *, default: float) -> float:
    return min(1.0, max(0.0, _float(value, default)))


def _skill_failure_code(skill_name: str) -> float:
    if skill_name.startswith("navigate."):
        return 0.2
    if skill_name == "manip.align_workspace":
        return 0.4
    if skill_name in {"manip.side_grasp", "manip.top_grasp", "manip.single_hand_pinch", "manip.bimanual_clamp"}:
        return 0.6
    if skill_name in {"manip.lift_object", "manip.transport_object", "manip.place_object"}:
        return 0.8
    return 0.0


def _handler_code(handler: str) -> float:
    return {
        "navigation_micro_adjust": 0.25,
        "perception_reobserve": 0.5,
        "affordance_repair": 0.75,
        "runtime_replan": 1.0,
    }.get(handler, 0.0)


def _apply_residual(
    payload: dict[str, Any],
    residual: list[float],
    mode: str,
    *,
    allow_base_residual: bool = True,
    allow_grasp_residual: bool = True,
    allow_recovery_override: bool = True,
) -> None:
    base = payload.get("base_goal") if isinstance(payload.get("base_goal"), dict) else None
    if allow_base_residual and base and isinstance(base.get("position"), list) and len(base["position"]) >= 2:
        base["position"][0] += float(residual[0]) * 0.18
        base["position"][1] += float(residual[1]) * 0.18
        base["yaw"] = float(base.get("yaw") or 0.0) + float(residual[2]) * 0.25
    offsets = payload.get("grasp_offsets") if isinstance(payload.get("grasp_offsets"), dict) else None
    if allow_grasp_residual and offsets and isinstance(offsets.get("contact_offset"), list) and len(offsets["contact_offset"]) >= 3:
        offsets["contact_offset"][0] += float(residual[4]) * 0.03
        offsets["contact_offset"][2] += float(residual[5]) * 0.02
    close = payload.get("grasp_close_ratio") if isinstance(payload.get("grasp_close_ratio"), dict) else None
    if allow_grasp_residual and close and close.get("close_ratio") is not None:
        close["close_ratio"] = max(0.05, min(0.95, float(close["close_ratio"]) + float(residual[6]) * 0.10))
    if allow_recovery_override:
        recovery = payload.get("recovery_decision") if isinstance(payload.get("recovery_decision"), dict) else {}
        # The recovery planner owns handler selection. The learned head is an
        # advisory mode only once a verified recovery action exists.
        recovery["policy_mode"] = mode
        payload["recovery_decision"] = recovery


def _manipulation_only(skills: set[str]) -> bool:
    return bool(skills) and all(str(skill).startswith("manip.") for skill in skills)


def _allows_skill(training_skills: set[str], skill_name: str) -> bool:
    """Empty training metadata denotes a general checkpoint; otherwise scope it exactly."""
    return not training_skills or skill_name in training_skills


def _skill_runtime_overrides(
    residual: list[float],
    *,
    recovery_context: Any = None,
    training_skills: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return bounded deltas added to the primitive's verified profile values."""
    def delta(index: int, scale: float) -> float:
        return round(max(-1.0, min(1.0, float(residual[index]))) * scale, 5)

    skills = training_skills or set()
    overrides: dict[str, dict[str, float]] = {}
    if _allows_skill(skills, "manip.side_grasp"):
        overrides["manip.side_grasp"] = {
            "contact_x_delta_m": delta(4, 0.025),
            "contact_z_delta_m": delta(5, 0.015),
        }
    if _allows_skill(skills, "manip.lift_object"):
        overrides["manip.lift_object"] = {
            "squeeze_close_ratio": delta(6, 0.08),
            "hold_close_ratio": delta(6, 0.08),
            "servo_lift_z_lead": delta(5, 0.010),
            "lift_z": delta(7, 0.08),
        }
    return overrides
