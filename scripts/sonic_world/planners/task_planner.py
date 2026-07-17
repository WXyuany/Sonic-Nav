from __future__ import annotations

from typing import Any
import uuid

from ..skills.specs import SkillGraph
from ..world_model.affordance import approach_pose_from_affordance, select_grasp_affordance
from ..world_model.entities import WorldObject, WorldState
from .requests import TaskRequest
from .templates import PlanContext, TaskTemplate, default_task_templates


class TaskPlanner:
    def __init__(self, templates: tuple[TaskTemplate, ...] | None = None):
        self.templates = templates or default_task_templates()

    def plan(self, world: WorldState, request: TaskRequest) -> SkillGraph:
        template = self._select_template(request.verb)
        ctx = self._build_context(world, request, template)
        steps = template.builder(ctx)

        return SkillGraph(
            task_id=str(request.metadata.get("request_id") or uuid.uuid4()),
            objective=self._objective_text(request, ctx.task_object, ctx.target),
            steps=steps,
            metadata={
                "planner": "world_model_task_planner_v0",
                "task_template": template.name,
                "object_category": ctx.task_object.category if ctx.task_object else None,
                "target_category": ctx.target.category if ctx.target else None,
                "grasp_affordance": ctx.grasp.name if ctx.grasp else None,
                "world_frame": world.frame_id,
                "warnings": list(ctx.warnings),
            },
        )

    def _select_template(self, verb: str) -> TaskTemplate:
        normalized = verb.strip().lower().replace("-", "_")
        for template in self.templates:
            if template.matches(normalized):
                return template
        known = sorted({verb for template in self.templates for verb in template.verbs})
        raise ValueError(f"unsupported task verb {verb!r}; expected one of {known}")

    def _build_context(
        self,
        world: WorldState,
        request: TaskRequest,
        template: TaskTemplate,
    ) -> PlanContext:
        target = (
            self._select_target(world, request, allow_object_fallback=template.prefers_target)
            if template.prefers_target or request.target_id
            else None
        )
        obj = self._select_object(
            world,
            request,
            fallback=target if not template.requires_grasp else None,
        )
        grasp = (
            select_grasp_affordance(obj, preferred=self._preferred_grasp(request, obj))
            if obj is not None and template.requires_grasp
            else None
        )
        approach_params = self._approach_params(obj)
        warnings = self._warnings(request, template, obj, target, grasp)
        return PlanContext(
            world=world,
            request=request,
            task_object=obj,
            target=target,
            grasp=grasp,
            approach_params=approach_params,
            warnings=tuple(warnings),
        )

    def _select_object(
        self,
        world: WorldState,
        request: TaskRequest,
        *,
        fallback: WorldObject | None = None,
    ) -> WorldObject | None:
        if request.object_id:
            obj = world.get_object(request.object_id)
            if obj is None:
                matches = world.find_by_category(request.object_category) if request.object_category else []
                if len(matches) != 1:
                    raise ValueError(f"object_id not found: {request.object_id}")
                obj = matches[0]
            return obj
        if request.object_category:
            matches = world.find_by_category(request.object_category)
            if not matches:
                raise ValueError(f"object category not found: {request.object_category}")
            return matches[0]
        if fallback is not None:
            return fallback
        obj = world.primary_object()
        return obj

    def _select_target(
        self,
        world: WorldState,
        request: TaskRequest,
        *,
        allow_object_fallback: bool = False,
    ) -> WorldObject | None:
        if request.target_id:
            target = world.get_object(request.target_id)
            if target is None:
                candidates = [
                    *world.find_by_category("place_target"),
                    *world.find_by_category("navigation_goal"),
                ]
                if len(candidates) != 1:
                    raise ValueError(f"target_id not found: {request.target_id}")
                target = candidates[0]
            return target
        targets = world.find_by_category("place_target")
        if targets:
            return targets[0]
        goals = world.find_by_category("navigation_goal")
        if goals:
            return goals[0]
        return world.primary_object() if allow_object_fallback else None

    def _approach_params(self, obj: WorldObject | None) -> dict[str, Any]:
        if obj is None:
            return {}
        approach_pose = approach_pose_from_affordance(obj)
        approach_params: dict[str, Any] = {}
        if approach_pose is not None:
            approach_params["pose"] = approach_pose.to_dict()
        for aff in obj.affordances:
            if aff.name == "approach_pose":
                approach_params.update(aff.params)
        return approach_params

    def _preferred_grasp(self, request: TaskRequest, obj: WorldObject | None) -> str | None:
        if obj is None:
            return None
        raw = request.metadata.get("preferred_grasp_affordance") if isinstance(request.metadata, dict) else None
        if raw:
            return str(raw)
        grasp = obj.properties.get("grasp") if isinstance(obj.properties.get("grasp"), dict) else {}
        raw = grasp.get("preferred_affordance") or grasp.get("grasp_affordance")
        if raw:
            return str(raw)
        raw = obj.properties.get("preferred_grasp_affordance")
        return str(raw) if raw else None

    def _warnings(
        self,
        request: TaskRequest,
        template: TaskTemplate,
        obj: WorldObject | None,
        target: WorldObject | None,
        grasp: Any,
    ) -> list[str]:
        warnings: list[str] = []
        if obj is None:
            warnings.append("missing_task_object")
        elif obj.pose_base is None and template.requires_grasp:
            warnings.append("missing_object_base_pose")
        if template.requires_grasp and grasp is None:
            warnings.append("missing_grasp_affordance")
        if template.name == "pick_place" and target is None:
            warnings.append("missing_place_target")
        if request.metadata:
            raw_required = request.metadata.get("required_capabilities")
            if isinstance(raw_required, list):
                warnings.extend(f"requires:{capability}" for capability in raw_required)
        return warnings

    def _objective_text(
        self,
        request: TaskRequest,
        obj: WorldObject | None,
        target: WorldObject | None,
    ) -> str:
        if obj is None and target is not None:
            return f"{request.verb}->{target.object_id}"
        if obj is None:
            return request.verb
        if target is not None:
            return f"{request.verb}:{obj.object_id}->{target.object_id}"
        return f"{request.verb}:{obj.object_id}"
