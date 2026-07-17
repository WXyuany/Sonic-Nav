from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .requests import TaskRequest
from .task_planner import TaskPlanner
from ..skills.decision import DecisionPlan, decision_plan_for_plans
from ..skills.dispatch import DispatchPlan, dispatch_plan_for_graph
from ..skills.recovery import RecoveryPlan, recovery_plan_for_dispatch
from ..skills.runtime import RuntimePlan, runtime_plan_for_graph
from ..skills.specs import SkillGraph
from ..world_model.anchors import anchor_to_world, detect_anchor_kind
from ..world_model.entities import WorldState
from ..world_model.memory import WorldMemory


@dataclass(frozen=True)
class PlanningResult:
    kind: str
    source: str
    world: WorldState
    request: TaskRequest
    skill_graph: SkillGraph
    runtime_plan: RuntimePlan
    dispatch_plan: DispatchPlan
    recovery_plan: RecoveryPlan
    decision_plan: DecisionPlan

    def world_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source, "world": self.world.to_dict()}

    def active_task_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source, "request": self.request.to_dict()}

    def skill_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "request": self.request.to_dict(),
            "skill_graph": self.skill_graph.to_dict(),
        }

    def runtime_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "runtime_plan": self.runtime_plan.to_dict(),
        }

    def dispatch_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "dispatch_plan": self.dispatch_plan.to_dict(),
        }

    def recovery_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "recovery_plan": self.recovery_plan.to_dict(),
        }

    def decision_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "decision_plan": self.decision_plan.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "world": self.world.to_dict(),
            "request": self.request.to_dict(),
            "skill_graph": self.skill_graph.to_dict(),
            "runtime_plan": self.runtime_plan.to_dict(),
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "recovery_plan": self.recovery_plan.to_dict(),
            "decision_plan": self.decision_plan.to_dict(),
        }


class WorldModelPipeline:
    def __init__(
        self,
        *,
        memory: WorldMemory | None = None,
        planner: TaskPlanner | None = None,
        box_verb: str = "pick",
        ball_verb: str = "pick_place",
    ) -> None:
        self.memory = memory or WorldMemory()
        self.planner = planner or TaskPlanner()
        self.box_verb = box_verb
        self.ball_verb = ball_verb

    def observe_anchor(
        self,
        anchor: dict[str, Any],
        *,
        source: str = "anchor",
        request: TaskRequest | None = None,
        kind: str | None = None,
    ) -> PlanningResult:
        anchor_kind = detect_anchor_kind(anchor)
        observation = anchor_to_world(anchor)
        world = self.memory.update(observation)
        task_request = request or self.request_for_anchor(anchor_kind, anchor)
        result_kind = kind if kind is not None else anchor_kind
        return self.plan(world, task_request, kind=result_kind, source=source)

    def plan_current(
        self,
        request: TaskRequest,
        *,
        kind: str = "task_request",
        source: str = "task_request",
    ) -> PlanningResult:
        world = self.memory.current()
        if not world.objects:
            raise ValueError("world memory is empty; send an object anchor or goal first")
        return self.plan(world, request, kind=kind, source=source)

    def plan(
        self,
        world: WorldState,
        request: TaskRequest,
        *,
        kind: str,
        source: str,
    ) -> PlanningResult:
        graph = self.planner.plan(world, request)
        runtime = runtime_plan_for_graph(graph, demo_kind=None if kind == "task_request" else kind)
        dispatch = dispatch_plan_for_graph(graph, runtime, world)
        recovery = recovery_plan_for_dispatch(dispatch)
        decision = decision_plan_for_plans(dispatch, recovery)
        return PlanningResult(
            kind=kind,
            source=source,
            world=world,
            request=request,
            skill_graph=graph,
            runtime_plan=runtime,
            dispatch_plan=dispatch,
            recovery_plan=recovery,
            decision_plan=decision,
        )

    def request_for_anchor(self, kind: str, anchor: dict[str, Any]) -> TaskRequest:
        if kind == "ball":
            return TaskRequest(
                verb=str(self.ball_verb),
                object_id=str(anchor.get("ball_name", "ball")),
                target_id="place_target" if "place_center_map" in anchor else None,
            )
        if kind in {"object", "objects"}:
            return self._request_for_generic_anchor(anchor)
        if kind == "navigation_goal":
            return TaskRequest(
                verb="navigate",
                object_id=str(anchor.get("goal_name", "navigation_goal")),
            )
        return TaskRequest(
            verb=str(self.box_verb),
            object_id=str(anchor.get("box_name", "box")),
        )

    def _request_for_generic_anchor(self, anchor: dict[str, Any]) -> TaskRequest:
        objects = anchor.get("objects")
        records = objects if isinstance(objects, list) else [anchor]
        task_object_id: str | None = None
        target_id: str | None = None
        navigation_id: str | None = None
        for record in records:
            if not isinstance(record, dict):
                continue
            object_id = str(record.get("object_id") or record.get("id") or record.get("name") or "").strip()
            category = str(record.get("category") or record.get("object_category") or record.get("class") or "")
            if not object_id:
                continue
            if category == "navigation_goal":
                navigation_id = object_id
            elif category == "place_target":
                target_id = object_id
            elif task_object_id is None:
                task_object_id = object_id
        if navigation_id is not None and task_object_id is None:
            return TaskRequest(verb="navigate", object_id=navigation_id)
        return TaskRequest(
            verb="pick_place" if target_id is not None else "pick",
            object_id=task_object_id,
            target_id=target_id,
        )
