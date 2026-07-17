from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .planners import PlanningResult, TaskRequest, WorldModelPipeline
from .world_model import WorldMemory, anchor_to_world


@dataclass(frozen=True)
class ScenarioExpectation:
    steps: tuple[str, ...] = ()
    handlers: tuple[str, ...] = ()
    demo_kind: str | None = None
    grasp_affordance: str | None = None
    missing_skills: tuple[str, ...] | None = ()
    unready_count: int | None = 0
    contract_error_count: int | None = 0
    recovery_suggestions: tuple[str, ...] = ()
    recovery_status: str | None = None
    recovery_action_count: int | None = None
    decision_status: str | None = None
    decision_next_kind: str | None = None
    decision_next_handler: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ScenarioExpectation":
        raw = payload or {}
        return cls(
            steps=_tuple_of_strings(raw.get("steps")),
            handlers=_tuple_of_strings(raw.get("handlers")),
            demo_kind=_optional_string(raw.get("demo_kind")),
            grasp_affordance=_optional_string(raw.get("grasp_affordance")),
            missing_skills=_optional_tuple(raw.get("missing_skills"), default=()),
            unready_count=_optional_int(raw.get("unready_count"), default=0),
            contract_error_count=_optional_int(raw.get("contract_error_count"), default=0),
            recovery_suggestions=_tuple_of_strings(raw.get("recovery_suggestions")),
            recovery_status=_optional_string(raw.get("recovery_status")),
            recovery_action_count=_optional_int(raw.get("recovery_action_count"), default=None),
            decision_status=_optional_string(raw.get("decision_status")),
            decision_next_kind=_optional_string(raw.get("decision_next_kind")),
            decision_next_handler=_optional_string(raw.get("decision_next_handler")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "handlers": list(self.handlers),
            "demo_kind": self.demo_kind,
            "grasp_affordance": self.grasp_affordance,
            "missing_skills": None if self.missing_skills is None else list(self.missing_skills),
            "unready_count": self.unready_count,
            "contract_error_count": self.contract_error_count,
            "recovery_suggestions": list(self.recovery_suggestions),
            "recovery_status": self.recovery_status,
            "recovery_action_count": self.recovery_action_count,
            "decision_status": self.decision_status,
            "decision_next_kind": self.decision_next_kind,
            "decision_next_handler": self.decision_next_handler,
        }


@dataclass(frozen=True)
class ScenarioTask:
    name: str
    request: TaskRequest
    kind: str = "task_request"
    source: str = "scenario"
    expectation: ScenarioExpectation = field(default_factory=ScenarioExpectation)
    recovery_anchors: tuple[dict[str, Any], ...] = ()
    recovery_expectation: ScenarioExpectation | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, index: int) -> "ScenarioTask":
        if not isinstance(payload, dict):
            raise ValueError("scenario task must be an object")
        request_payload = payload.get("request") or payload.get("task_request") or payload
        if not isinstance(request_payload, dict):
            raise ValueError("scenario task request must be an object")
        name = str(payload.get("name") or request_payload.get("id") or f"task_{index}")
        source = str(payload.get("source") or f"scenario:{name}")
        recovery_payload = payload.get("recovery") or {}
        if recovery_payload and not isinstance(recovery_payload, dict):
            raise ValueError("scenario task recovery must be an object")
        recovery_anchors = recovery_payload.get("anchors") if recovery_payload else None
        if recovery_anchors is None:
            recovery_anchors = []
        if not isinstance(recovery_anchors, list):
            raise ValueError("scenario task recovery.anchors must be a list")
        return cls(
            name=name,
            request=TaskRequest.from_dict(request_payload),
            kind=str(payload.get("kind") or "task_request"),
            source=source,
            expectation=ScenarioExpectation.from_dict(payload.get("expect")),
            recovery_anchors=tuple(_copy_dict(anchor, "recovery anchor") for anchor in recovery_anchors),
            recovery_expectation=(
                ScenarioExpectation.from_dict(recovery_payload.get("expect"))
                if recovery_payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "request": self.request.to_dict(),
            "expect": self.expectation.to_dict(),
        }
        if self.recovery_anchors or self.recovery_expectation is not None:
            payload["recovery"] = {
                "anchors": list(self.recovery_anchors),
                "expect": self.recovery_expectation.to_dict() if self.recovery_expectation else None,
            }
        return payload


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    anchors: tuple[dict[str, Any], ...]
    tasks: tuple[ScenarioTask, ...]
    expected_objects: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioSpec":
        if not isinstance(payload, dict):
            raise ValueError("scenario JSON must be an object")
        anchors = payload.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("scenario must contain a non-empty anchors list")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("scenario must contain a non-empty tasks list")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("scenario metadata must be an object")
        return cls(
            name=str(payload.get("name") or "scenario"),
            anchors=tuple(_copy_dict(anchor, "anchor") for anchor in anchors),
            tasks=tuple(ScenarioTask.from_dict(task, index=idx) for idx, task in enumerate(tasks)),
            expected_objects=_tuple_of_strings(payload.get("expected_objects")),
            metadata=metadata,
        )

    @classmethod
    def load(cls, value: str | Path | dict[str, Any]) -> "ScenarioSpec":
        if isinstance(value, dict):
            return cls.from_dict(value)
        path = Path(value)
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchors": list(self.anchors),
            "tasks": [task.to_dict() for task in self.tasks],
            "expected_objects": list(self.expected_objects),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScenarioCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ScenarioTaskReplay:
    task: ScenarioTask
    result: PlanningResult
    checks: tuple[ScenarioCheck, ...]
    recovery_result: PlanningResult | None = None
    recovery_checks: tuple[ScenarioCheck, ...] = ()

    @property
    def passed(self) -> bool:
        return all(check.passed for check in (*self.checks, *self.recovery_checks))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task": self.task.to_dict(),
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "result": self.result.to_dict(),
        }
        if self.recovery_result is not None:
            payload["recovery_checks"] = [check.to_dict() for check in self.recovery_checks]
            payload["recovery_result"] = self.recovery_result.to_dict()
        return payload


@dataclass(frozen=True)
class ScenarioReplay:
    scenario: ScenarioSpec
    world_objects: tuple[str, ...]
    checks: tuple[ScenarioCheck, ...]
    tasks: tuple[ScenarioTaskReplay, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks) and all(task.passed for task in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "passed": self.passed,
            "world_objects": list(self.world_objects),
            "checks": [check.to_dict() for check in self.checks],
            "tasks": [task.to_dict() for task in self.tasks],
        }


def replay_scenario(
    spec: ScenarioSpec,
    *,
    pipeline: WorldModelPipeline | None = None,
) -> ScenarioReplay:
    pipeline = pipeline or WorldModelPipeline(memory=WorldMemory(stale_after_s=0.0))
    for anchor in spec.anchors:
        pipeline.memory.update(anchor_to_world(anchor))
    world = pipeline.memory.current()
    checks = tuple(_world_checks(spec, tuple(world.objects)))
    task_replays: list[ScenarioTaskReplay] = []
    for task in spec.tasks:
        result = pipeline.plan_current(task.request, kind=task.kind, source=task.source)
        recovery_result = None
        recovery_checks: tuple[ScenarioCheck, ...] = ()
        if task.recovery_anchors:
            for anchor in task.recovery_anchors:
                pipeline.memory.update(anchor_to_world(anchor))
            recovery_result = pipeline.plan_current(
                task.request,
                kind=task.kind,
                source=f"{task.source}:after_recovery",
            )
            recovery_checks = tuple(_task_checks(task.recovery_expectation or ScenarioExpectation(), recovery_result))
        task_replays.append(
            ScenarioTaskReplay(
                task=task,
                result=result,
                checks=tuple(_task_checks(task.expectation, result)),
                recovery_result=recovery_result,
                recovery_checks=recovery_checks,
            )
        )
    return ScenarioReplay(
        scenario=spec,
        world_objects=tuple(sorted(world.objects)),
        checks=checks,
        tasks=tuple(task_replays),
    )


def load_scenarios(paths: list[str | Path]) -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(ScenarioSpec.load(file_path) for file_path in sorted(path.glob("*.json")))
        else:
            out.append(ScenarioSpec.load(path))
    return out


def _world_checks(spec: ScenarioSpec, object_ids: tuple[str, ...]) -> list[ScenarioCheck]:
    if not spec.expected_objects:
        return []
    expected = set(spec.expected_objects)
    actual = set(object_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    passed = not missing
    detail = f"missing={missing} extra={extra}" if not passed else f"objects={sorted(actual)}"
    return [ScenarioCheck("world.expected_objects", passed, detail)]


def _task_checks(expectation: ScenarioExpectation, result: PlanningResult) -> list[ScenarioCheck]:
    checks: list[ScenarioCheck] = []
    if expectation.steps:
        actual = tuple(step.name for step in result.skill_graph.steps)
        checks.append(_equals_check("skill.steps", actual, expectation.steps))
    if expectation.handlers:
        actual = tuple(step.handler for step in result.dispatch_plan.steps)
        checks.append(_equals_check("dispatch.handlers", actual, expectation.handlers))
    if expectation.demo_kind is not None:
        checks.append(_equals_check("runtime.demo_kind", result.runtime_plan.demo_kind, expectation.demo_kind))
    if expectation.grasp_affordance is not None:
        actual = result.skill_graph.metadata.get("grasp_affordance")
        checks.append(_equals_check("skill.grasp_affordance", actual, expectation.grasp_affordance))
    if expectation.missing_skills is not None:
        actual = tuple(result.runtime_plan.metadata.get("missing_skills") or ())
        checks.append(_equals_check("runtime.missing_skills", actual, expectation.missing_skills))
    if expectation.unready_count is not None:
        actual = result.dispatch_plan.metadata.get("unready_count")
        checks.append(_equals_check("dispatch.unready_count", actual, expectation.unready_count))
    if expectation.contract_error_count is not None:
        actual = result.dispatch_plan.metadata.get("contract_error_count")
        checks.append(_equals_check("dispatch.contract_error_count", actual, expectation.contract_error_count))
    if expectation.recovery_suggestions:
        actual = tuple(result.dispatch_plan.metadata.get("recovery_suggestions") or ())
        missing = tuple(item for item in expectation.recovery_suggestions if item not in actual)
        checks.append(
            ScenarioCheck(
                "dispatch.recovery_suggestions",
                not missing,
                f"missing={missing!r} actual={actual!r}",
            )
        )
    if expectation.recovery_status is not None:
        checks.append(_equals_check("recovery.status", result.recovery_plan.status, expectation.recovery_status))
    if expectation.recovery_action_count is not None:
        checks.append(_equals_check("recovery.action_count", len(result.recovery_plan.actions), expectation.recovery_action_count))
    if expectation.decision_status is not None:
        checks.append(_equals_check("decision.status", result.decision_plan.status, expectation.decision_status))
    if expectation.decision_next_kind is not None:
        actual = result.decision_plan.next_action.kind if result.decision_plan.next_action else None
        checks.append(_equals_check("decision.next_kind", actual, expectation.decision_next_kind))
    if expectation.decision_next_handler is not None:
        actual = result.decision_plan.next_action.handler if result.decision_plan.next_action else None
        checks.append(_equals_check("decision.next_handler", actual, expectation.decision_next_handler))
    return checks


def _equals_check(name: str, actual: Any, expected: Any) -> ScenarioCheck:
    passed = actual == expected
    return ScenarioCheck(name, passed, f"actual={actual!r} expected={expected!r}")


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value)


def _optional_tuple(value: Any, *, default: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if value is None:
        return default
    return _tuple_of_strings(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    return int(value)


def _copy_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"scenario {label} must be an object")
    return dict(value)
