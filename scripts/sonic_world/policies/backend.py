from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Protocol

from .heuristic import HeuristicSkillPolicy
from .schema import PolicyAction
from ..planners import PlanningResult


class PolicyBackend(Protocol):
    policy_id: str

    def act(self, result: PlanningResult) -> PolicyAction:
        ...


class HeuristicPolicyBackend(HeuristicSkillPolicy):
    pass


class MemoryPolicyBackend:
    def __init__(self, model: dict[str, Any], *, fallback: PolicyBackend | None = None) -> None:
        self.model = model
        self.policy_id = str(model.get("schema") or "sonic_task_policy_memory")
        self.fallback = fallback or HeuristicPolicyBackend()
        self.exact = model.get("exact_task_policy") if isinstance(model.get("exact_task_policy"), dict) else {}
        self.fallback_policy = model.get("fallback_policy") if isinstance(model.get("fallback_policy"), dict) else {}

    @classmethod
    def load(cls, path: str | Path, *, fallback: PolicyBackend | None = None) -> "MemoryPolicyBackend":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"policy model must be a JSON object: {path}")
        return cls(payload, fallback=fallback)

    def act(self, result: PlanningResult) -> PolicyAction:
        task_id = _task_id(result)
        entry = self.exact.get(task_id)
        if not isinstance(entry, dict):
            entry = self.fallback_policy.get(_fallback_key(result))
        action = entry.get("recommended_action") if isinstance(entry, dict) else None
        if isinstance(action, dict):
            out = policy_action_from_dict(action)
            metadata = dict(out.metadata)
            metadata["policy_backend"] = {
                "type": "memory",
                "model_schema": self.model.get("schema"),
                "source": self.model.get("source"),
                "matched": "exact" if task_id in self.exact else "fallback",
                "teacher_policy_id": out.policy_id,
            }
            return _replace_policy_metadata(out, metadata, policy_id=self.policy_id)
        fallback_action = self.fallback.act(result)
        metadata = dict(fallback_action.metadata)
        metadata["policy_backend"] = {
            "type": "memory",
            "model_schema": self.model.get("schema"),
            "matched": "heuristic_fallback",
            "teacher_policy_id": fallback_action.policy_id,
        }
        return _replace_policy_metadata(fallback_action, metadata, policy_id=self.policy_id)


class LearnedPolicyBackend:
    """Strict, dependency-light learned policy adapter.

    This backend is the runtime contract for a trainable task/skill policy. It
    intentionally loads JSON/JSONL action templates instead of importing a neural
    framework here; trainer-specific checkpoint code can export this manifest as
    its stable online inference boundary.
    """

    def __init__(self, model: dict[str, Any], *, fallback: PolicyBackend | None = None) -> None:
        _validate_learned_model(model)
        self.model = model
        self.policy_id = str(model.get("policy_id") or model.get("model_id") or "learned_policy")
        self.fallback = fallback or HeuristicPolicyBackend()
        self.exact = model.get("exact_task_policy") if isinstance(model.get("exact_task_policy"), dict) else {}
        self.fallback_policy = model.get("fallback_policy") if isinstance(model.get("fallback_policy"), dict) else {}

    @classmethod
    def load(cls, path: str | Path, *, fallback: PolicyBackend | None = None) -> "LearnedPolicyBackend":
        payload = _read_policy_model(path)
        return cls(payload, fallback=fallback)

    def act(self, result: PlanningResult) -> PolicyAction:
        task_id = _task_id(result)
        matched = "exact"
        entry = self.exact.get(task_id)
        if not isinstance(entry, dict):
            matched = "fallback"
            entry = self.fallback_policy.get(_fallback_key(result))
        action = _action_payload_from_entry(entry)
        if isinstance(action, dict):
            out = policy_action_from_dict(action)
            metadata = dict(out.metadata)
            metadata["policy_backend"] = {
                "type": "learned",
                "model_schema": self.model.get("schema"),
                "model_id": self.model.get("model_id"),
                "policy_id": self.policy_id,
                "checkpoint": (self.model.get("checkpoint") or {}).get("path")
                if isinstance(self.model.get("checkpoint"), dict)
                else None,
                "matched": matched,
                "teacher_policy_id": out.policy_id,
            }
            return _replace_policy_metadata(out, metadata, policy_id=self.policy_id)
        fallback_action = self.fallback.act(result)
        metadata = dict(fallback_action.metadata)
        metadata["policy_backend"] = {
            "type": "learned",
            "model_schema": self.model.get("schema"),
            "model_id": self.model.get("model_id"),
            "policy_id": self.policy_id,
            "matched": "heuristic_fallback",
            "teacher_policy_id": fallback_action.policy_id,
        }
        return _replace_policy_metadata(fallback_action, metadata, policy_id=self.policy_id)


def load_policy_backend(kind: str, *, model_path: str | Path | None = None) -> PolicyBackend:
    normalized = str(kind or "heuristic").strip().lower()
    if normalized in {"heuristic", "teacher"}:
        return HeuristicPolicyBackend()
    if normalized in {"memory", "policy_memory"}:
        if model_path is None:
            raise ValueError("memory policy backend requires --policy-model")
        return MemoryPolicyBackend.load(model_path)
    if normalized in {"learned", "checkpoint", "neural"}:
        if model_path is None:
            raise ValueError("learned policy backend requires --policy-model")
        if Path(model_path).expanduser().suffix in {".pt", ".pth"}:
            from .hybrid_backend import HybridPPOPolicyBackend

            return HybridPPOPolicyBackend(model_path)
        model = _read_policy_model(model_path)
        if model.get("schema") == "sonic_linear_task_policy_v0":
            from .linear import LinearTaskPolicyBackend

            return LinearTaskPolicyBackend(model)
        return LearnedPolicyBackend(model)
    raise ValueError(f"unsupported policy backend {kind!r}")


def policy_action_from_dict(payload: dict[str, Any]) -> PolicyAction:
    names = {field.name for field in fields(PolicyAction)}
    values = {name: payload.get(name) for name in names if name in payload}
    required_defaults = {
        "policy_id": str(payload.get("policy_id") or "loaded_policy_action"),
        "task_id": str(payload.get("task_id") or ""),
        "status": str(payload.get("status") or "ready"),
        "task_intent": payload.get("task_intent") if isinstance(payload.get("task_intent"), dict) else {},
        "object_target_anchors": payload.get("object_target_anchors")
        if isinstance(payload.get("object_target_anchors"), list)
        else [],
        "skill_selection": payload.get("skill_selection") if isinstance(payload.get("skill_selection"), list) else [],
    }
    for key, value in required_defaults.items():
        values.setdefault(key, value)
    values.setdefault("metadata", payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {})
    values.setdefault(
        "ordered_skill_commands",
        payload.get("ordered_skill_commands") if isinstance(payload.get("ordered_skill_commands"), list) else [],
    )
    return PolicyAction(**values)


def _replace_policy_metadata(
    action: PolicyAction,
    metadata: dict[str, Any],
    *,
    policy_id: str | None = None,
) -> PolicyAction:
    payload = action.to_dict()
    payload["metadata"] = metadata
    if policy_id:
        payload["policy_id"] = policy_id
    return policy_action_from_dict(payload)


def _task_id(result: PlanningResult) -> str:
    request_id = result.request.metadata.get("request_id")
    if request_id:
        return str(request_id)
    return str(result.skill_graph.task_id)


def _fallback_key(result: PlanningResult) -> str:
    return "|".join(
        [
            str(result.runtime_plan.demo_kind or "unknown"),
            str(result.skill_graph.metadata.get("grasp_affordance") or "unknown"),
            str(result.skill_graph.metadata.get("object_category") or "unknown"),
        ]
    )


def _read_policy_model(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        exact: dict[str, dict[str, Any]] = {}
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"learned policy JSONL row is not an object: {p}:{line_no}")
            action = payload.get("action") if isinstance(payload.get("action"), dict) else payload
            task_id = str(action.get("task_id") or payload.get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"learned policy JSONL row missing task_id: {p}:{line_no}")
            exact[task_id] = {"recommended_action": action}
        return {
            "schema": "sonic_learned_policy_v0",
            "model_id": p.stem,
            "policy_id": p.stem,
            "exact_task_policy": exact,
            "fallback_policy": {},
            "manifest": {"source": str(p)},
        }
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"learned policy model must be a JSON object: {p}")
    return payload


def _validate_learned_model(model: dict[str, Any]) -> None:
    schema = str(model.get("schema") or "")
    if schema not in {"sonic_learned_policy_v0", "task_skill_policy_learned_v0"}:
        raise ValueError(f"learned policy model has unsupported schema {schema!r}")
    exact = model.get("exact_task_policy")
    fallback = model.get("fallback_policy")
    if not isinstance(exact, dict):
        raise ValueError("learned policy model requires exact_task_policy mapping")
    if fallback is not None and not isinstance(fallback, dict):
        raise ValueError("learned policy model fallback_policy must be a mapping")
    for name, entries in (("exact_task_policy", exact), ("fallback_policy", fallback or {})):
        for key, entry in entries.items():
            action = _action_payload_from_entry(entry)
            if not isinstance(action, dict):
                raise ValueError(f"{name}[{key!r}] missing recommended action")
            _validate_action_payload(action, f"{name}[{key!r}]")


def _action_payload_from_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    action = entry.get("recommended_action") or entry.get("action") or entry.get("policy_action")
    if isinstance(action, dict):
        return action
    if {"task_intent", "skill_selection"}.issubset(entry.keys()):
        return entry
    return None


def _validate_action_payload(action: dict[str, Any], label: str) -> None:
    for key in ("task_id", "status", "task_intent", "object_target_anchors", "skill_selection"):
        if key not in action:
            raise ValueError(f"{label} action missing {key}")
    if not str(action.get("task_id") or "").strip():
        raise ValueError(f"{label} action task_id must be non-empty")
    if str(action.get("status") or "") not in {"ready", "needs_recovery", "blocked", "failed"}:
        raise ValueError(f"{label} action has unsupported status {action.get('status')!r}")
    if not isinstance(action.get("task_intent"), dict):
        raise ValueError(f"{label} action task_intent must be a mapping")
    if not isinstance(action.get("object_target_anchors"), list):
        raise ValueError(f"{label} action object_target_anchors must be a list")
    if not isinstance(action.get("skill_selection"), list):
        raise ValueError(f"{label} action skill_selection must be a list")
    for key in (
        "base_goal",
        "hand_pose_target",
        "wrist_target",
        "grasp_close_ratio",
        "grasp_offsets",
        "lift_place_targets",
        "recovery_decision",
        "metadata",
    ):
        value = action.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{label} action {key} must be a mapping or null")
    commands = action.get("ordered_skill_commands", [])
    if commands is not None and not isinstance(commands, list):
        raise ValueError(f"{label} action ordered_skill_commands must be a list")
