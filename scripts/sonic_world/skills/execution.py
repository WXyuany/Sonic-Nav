from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from .runtime import RuntimePlan


_DONE_PHASES = {"done", "demo done", "hold_done"}
_FAILED_TOKENS = ("failed", "abort", "timeout", "lost")
_SUCCESS_TOKENS = ("reached", "done", "succeeded", "success")


@dataclass(frozen=True)
class ExecutionState:
    task_id: str | None
    status: str
    current_phase: str | None = None
    current_skill: str | None = None
    completed_skills: tuple[str, ...] = ()
    remaining_skills: tuple[str, ...] = ()
    recovery_options: tuple[str, ...] = ()
    last_event: str | None = None
    stamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "current_phase": self.current_phase,
            "current_skill": self.current_skill,
            "completed_skills": list(self.completed_skills),
            "remaining_skills": list(self.remaining_skills),
            "recovery_options": list(self.recovery_options),
            "last_event": self.last_event,
            "stamp": float(self.stamp),
            "metadata": self.metadata,
        }


class SkillExecutionMonitor:
    def __init__(self) -> None:
        self.runtime: RuntimePlan | None = None
        self.state = ExecutionState(task_id=None, status="idle")

    def set_runtime(self, runtime: RuntimePlan) -> ExecutionState:
        self.runtime = runtime
        self.state = ExecutionState(
            task_id=runtime.task_id,
            status="planned",
            remaining_skills=self._skill_order(),
            metadata={"demo_kind": runtime.demo_kind},
        )
        return self.state

    def update_phase(self, phase_name: str, *, event: str = "phase") -> ExecutionState:
        phase = str(phase_name).strip()
        if self.runtime is None:
            self.state = ExecutionState(
                task_id=None,
                status="no_plan",
                current_phase=phase,
                last_event=event,
            )
            return self.state

        normalized = phase.lower()
        if normalized in _DONE_PHASES:
            self.state = ExecutionState(
                task_id=self.runtime.task_id,
                status="succeeded",
                current_phase=phase,
                completed_skills=self._skill_order(),
                last_event=event,
                metadata={"demo_kind": self.runtime.demo_kind},
            )
            return self.state

        skill = self.runtime.skill_for_phase(phase)
        status = self._status_from_text(phase, default="running")
        completed, remaining = self._progress_for_skill(skill)
        self.state = ExecutionState(
            task_id=self.runtime.task_id,
            status=status if skill is not None else "unknown_phase",
            current_phase=phase,
            current_skill=skill,
            completed_skills=completed,
            remaining_skills=remaining,
            recovery_options=self._recovery_for_skill(skill),
            last_event=event,
            metadata={"demo_kind": self.runtime.demo_kind},
        )
        return self.state

    def update_status_text(self, text: str, *, event: str = "status") -> ExecutionState:
        payload = _parse_status_text(text)
        phase = payload.get("state") or payload.get("phase") or payload.get("reason") or text
        if self.runtime is not None and self.runtime.demo_kind == "navigation":
            if str(phase) in {"reached", "carma_reached"}:
                phase = "navigate_to_goal"
                state = self.update_phase(phase, event=event)
                self.state = ExecutionState(
                    task_id=state.task_id,
                    status="succeeded",
                    current_phase=state.current_phase,
                    current_skill=state.current_skill,
                    completed_skills=self._skill_order(),
                    remaining_skills=(),
                    recovery_options=(),
                    last_event=event,
                    metadata={**state.metadata, "status_text": text, "parsed": payload},
                )
                return self.state
            state = self.update_phase("navigate_to_goal", event=event)
            status = self._status_from_text(str(phase), default=state.status)
            self.state = ExecutionState(
                task_id=state.task_id,
                status=status,
                current_phase=state.current_phase,
                current_skill=state.current_skill,
                completed_skills=state.completed_skills,
                remaining_skills=state.remaining_skills,
                recovery_options=state.recovery_options,
                last_event=event,
                metadata={**state.metadata, "status_text": text, "parsed": payload},
            )
            return self.state

        return self.update_phase(str(phase), event=event)

    def _skill_order(self) -> tuple[str, ...]:
        if self.runtime is None:
            return ()
        return tuple(binding.skill_name for binding in self.runtime.bindings)

    def _progress_for_skill(self, skill: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        order = self._skill_order()
        if skill is None or skill not in order:
            return (), order
        idx = order.index(skill)
        return order[:idx], order[idx + 1 :]

    def _recovery_for_skill(self, skill: str | None) -> tuple[str, ...]:
        if self.runtime is None or skill is None:
            return ()
        for binding in self.runtime.bindings:
            if binding.skill_name == skill:
                return binding.recovery_events
        return ()

    def _status_from_text(self, text: str, *, default: str) -> str:
        lower = text.lower()
        if any(token in lower for token in _FAILED_TOKENS):
            return "failed"
        if any(token in lower for token in _SUCCESS_TOKENS):
            return "succeeded"
        return default


def _parse_status_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in str(text).split():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip(",")
        if key:
            out[key] = value
    return out
