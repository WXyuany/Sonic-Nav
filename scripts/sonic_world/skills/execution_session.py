from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Callable


EffectVerifier = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str, dict[str, Any]]]

_RUNNING_STATES = {"accepted", "queued", "running", "phase_start", "phase_end"}
_SUCCESS_STATES = {"success", "succeeded"}
_FAILURE_STATES = {"failed", "error", "timeout", "cancelled", "canceled", "skipped"}


@dataclass(frozen=True)
class ExecutionTransition:
    kind: str
    plan_id: str | None
    action_id: str | None = None
    status: str = ""
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "plan_id": self.plan_id,
            "action_id": self.action_id,
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
        }


class DispatchExecutionSession:
    """Own the lifecycle of one ordered dispatch plan.

    This class is deliberately ROS-independent. A transport publishes the action
    returned by ``next_action`` and feeds terminal backend status back through
    ``observe_status``. Only a verified terminal success advances the cursor.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 45.0,
        require_effect_evidence: bool = False,
        verifier: EffectVerifier | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout_s = max(0.1, float(timeout_s))
        self.require_effect_evidence = bool(require_effect_evidence)
        self.verifier = verifier
        self.clock = clock
        self.plan_id: str | None = None
        self.task_id: str | None = None
        self.actions: list[dict[str, Any]] = []
        self.cursor = 0
        self.active_action_id: str | None = None
        self.active_since: float | None = None
        self.status = "idle"
        self.completed_action_ids: list[str] = []
        self.failed_action_id: str | None = None
        self.last_failure = ""

    def load(self, plan: dict[str, Any]) -> ExecutionTransition:
        actions = [
            dict(action)
            for action in plan.get("actions", [])
            if isinstance(action, dict) and action.get("kind") == "dispatch"
        ]
        if not actions and isinstance(plan.get("next_action"), dict):
            action = dict(plan["next_action"])
            if action.get("kind") == "dispatch":
                actions = [action]
        plan_id = _plan_id(plan, actions)
        if plan_id == self.plan_id and self.status != "failed":
            return ExecutionTransition("duplicate_plan", plan_id, status=self.status)
        if self.status in {"running", "queued"}:
            return ExecutionTransition(
                "busy",
                self.plan_id,
                action_id=self.active_action_id,
                status=self.status,
                reason=f"reject plan {plan_id}: another plan is active",
            )
        error = _validate_actions(actions)
        if error:
            return ExecutionTransition("invalid_plan", plan_id, status="failed", reason=error)
        self.plan_id = plan_id
        self.task_id = _string_or_none(plan.get("task_id"))
        self.actions = actions
        self.cursor = 0
        self.active_action_id = None
        self.active_since = None
        self.status = "ready" if actions else "succeeded"
        self.completed_action_ids = []
        self.failed_action_id = None
        self.last_failure = ""
        return ExecutionTransition("plan_loaded", plan_id, status=self.status, metrics={"action_count": len(actions)})

    def next_action(self) -> dict[str, Any] | None:
        if self.status not in {"ready", "running"} or self.active_action_id is not None:
            return None
        if self.cursor >= len(self.actions):
            self.status = "succeeded"
            return None
        return self.actions[self.cursor]

    def mark_dispatched(self, action_id: str) -> ExecutionTransition:
        action = self.next_action()
        expected = _action_id(action) if action is not None else None
        if expected is None or action_id != expected:
            return ExecutionTransition(
                "dispatch_rejected",
                self.plan_id,
                action_id=action_id,
                status=self.status,
                reason=f"expected action {expected!r}",
            )
        self.active_action_id = action_id
        self.active_since = self.clock()
        self.status = "running"
        return ExecutionTransition("action_dispatched", self.plan_id, action_id=action_id, status="running")

    def observe_status(self, payload: dict[str, Any]) -> ExecutionTransition:
        action_id = _string_or_none(payload.get("action_id"))
        state = str(payload.get("status") or "").strip().lower()
        if action_id != self.active_action_id:
            kind = "duplicate_status" if action_id in self.completed_action_ids else "stale_status"
            return ExecutionTransition(kind, self.plan_id, action_id=action_id, status=state)
        if state in _RUNNING_STATES:
            return ExecutionTransition("action_feedback", self.plan_id, action_id=action_id, status=state)
        if state in _SUCCESS_STATES:
            return self._complete_active(payload)
        if state in _FAILURE_STATES:
            return self._fail_active(state or "failed", str(payload.get("detail") or state), payload.get("metrics"))
        return ExecutionTransition(
            "invalid_status",
            self.plan_id,
            action_id=action_id,
            status=state,
            reason="backend status is not a recognized lifecycle state",
        )

    def check_timeout(self) -> ExecutionTransition | None:
        if self.active_action_id is None or self.active_since is None:
            return None
        elapsed = self.clock() - self.active_since
        if elapsed < self.timeout_s:
            return None
        return self._fail_active(
            "timeout",
            f"no terminal status within {self.timeout_s:.1f}s",
            {"elapsed_s": elapsed, "timeout_s": self.timeout_s},
        )

    def cancel(self, reason: str = "cancelled by caller") -> ExecutionTransition:
        if self.active_action_id is None:
            return ExecutionTransition("cancel_ignored", self.plan_id, status=self.status, reason="no active action")
        return self._fail_active("cancelled", reason, {})

    def release_for_replan(self, reason: str) -> ExecutionTransition:
        """Release this session so a recovery-generated plan can replace it.

        A recovery replan is deliberately a new execution attempt, even when the
        planner emits deterministic action and plan identifiers. Keeping the old
        identifiers would make ``load`` treat the replacement as a duplicate.
        """
        previous_plan_id = self.plan_id
        metrics = {
            "cursor": self.cursor,
            "action_count": len(self.actions),
            "completed_action_count": len(self.completed_action_ids),
            "failed_action_id": self.failed_action_id,
        }
        self.plan_id = None
        self.task_id = None
        self.actions = []
        self.cursor = 0
        self.active_action_id = None
        self.active_since = None
        self.status = "idle"
        self.completed_action_ids = []
        self.failed_action_id = None
        self.last_failure = ""
        return ExecutionTransition(
            "plan_released_for_replan",
            previous_plan_id,
            status="idle",
            reason=reason,
            metrics=metrics,
        )

    def current_action(self) -> dict[str, Any] | None:
        if self.active_action_id is None or self.cursor >= len(self.actions):
            return None
        return self.actions[self.cursor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sonic_dispatch_execution_state_v0",
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "status": self.status,
            "cursor": self.cursor,
            "action_count": len(self.actions),
            "active_action_id": self.active_action_id,
            "completed_action_ids": list(self.completed_action_ids),
            "failed_action_id": self.failed_action_id,
            "last_failure": self.last_failure,
        }

    def _complete_active(self, payload: dict[str, Any]) -> ExecutionTransition:
        action = self.current_action() or {}
        verified, reason, verify_metrics = self._verify_effect(action, payload)
        metrics = {**dict(payload.get("metrics") or {}), **verify_metrics}
        if not verified:
            return self._fail_active("failed", reason or "skill effects were not verified", metrics)
        action_id = self.active_action_id
        if action_id is not None:
            self.completed_action_ids.append(action_id)
        self.cursor += 1
        self.active_action_id = None
        self.active_since = None
        if self.cursor >= len(self.actions):
            self.status = "succeeded"
            return ExecutionTransition("plan_succeeded", self.plan_id, action_id=action_id, status="success", metrics=metrics)
        self.status = "ready"
        return ExecutionTransition("action_succeeded", self.plan_id, action_id=action_id, status="success", metrics=metrics)

    def _verify_effect(
        self,
        action: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        if self.verifier is not None:
            return self.verifier(action, payload)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        evidence = payload.get("effect_evidence")
        if evidence is None:
            evidence = metrics.get("effect_verified")
        if isinstance(evidence, dict):
            passed = bool(evidence.get("passed"))
            return passed, str(evidence.get("reason") or ""), {"effect_evidence": evidence}
        if evidence is not None:
            return bool(evidence), "" if evidence else "backend rejected skill effects", {"effect_verified": bool(evidence)}
        if self.require_effect_evidence:
            return False, "terminal success did not include effect evidence", {"effect_verified": False}
        return True, "effect evidence not required", {"effect_verified": None}

    def _fail_active(self, status: str, reason: str, metrics: Any) -> ExecutionTransition:
        action_id = self.active_action_id
        self.failed_action_id = action_id
        self.last_failure = reason
        self.active_action_id = None
        self.active_since = None
        self.status = "failed"
        return ExecutionTransition(
            "action_failed",
            self.plan_id,
            action_id=action_id,
            status=status,
            reason=reason,
            metrics=dict(metrics or {}),
        )


def _plan_id(plan: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
    explicit = metadata.get("plan_id") or plan.get("plan_id")
    if explicit:
        return str(explicit)
    canonical = json.dumps(
        {"task_id": plan.get("task_id"), "objective": plan.get("objective"), "actions": actions},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"plan_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _validate_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "dispatch plan contains no dispatch actions"
    seen: set[str] = set()
    for index, action in enumerate(actions):
        action_id = _action_id(action)
        if not action_id:
            return f"dispatch action {index} is missing action_id"
        if action_id in seen:
            return f"duplicate dispatch action_id {action_id!r}"
        seen.add(action_id)
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        contract = metadata.get("contract") if isinstance(metadata.get("contract"), dict) else {}
        if metadata.get("readiness") not in {None, "ready"}:
            return f"dispatch action {action_id!r} is not ready"
        if contract and contract.get("ready") is not True:
            return f"dispatch action {action_id!r} has an unready capability contract"
        if not str(action.get("handler") or ""):
            return f"dispatch action {action_id!r} is missing handler"
    return ""


def _action_id(action: dict[str, Any] | None) -> str | None:
    return _string_or_none(action.get("action_id")) if action is not None else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
