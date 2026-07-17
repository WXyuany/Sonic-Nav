from __future__ import annotations

from dataclasses import dataclass, field
import time

from .entities import WorldRelation, WorldState


@dataclass
class WorldMemory:
    """Small in-process world state accumulator for anchor-driven demos."""

    frame_id: str = "map"
    stale_after_s: float = 5.0
    state: WorldState = field(default_factory=WorldState)
    object_stamps: dict[str, float] = field(default_factory=dict)

    def update(self, observation: WorldState) -> WorldState:
        now = float(observation.stamp or time.time())
        self.state.frame_id = observation.frame_id or self.frame_id
        self.state.stamp = now
        self.state.properties.update(observation.properties)
        self.state.robot = observation.robot
        for object_id, obj in observation.objects.items():
            self.state.objects[object_id] = obj
            self.object_stamps[object_id] = now
        self._merge_relations(observation.relations)
        self.prune(now)
        return self.state

    def current(self) -> WorldState:
        self.prune(time.time())
        return self.state

    def prune(self, now: float | None = None) -> None:
        if self.stale_after_s <= 0.0:
            return
        stamp = time.time() if now is None else float(now)
        stale_ids = [
            object_id
            for object_id, seen_at in self.object_stamps.items()
            if stamp - float(seen_at) > self.stale_after_s
        ]
        for object_id in stale_ids:
            self.object_stamps.pop(object_id, None)
            self.state.objects.pop(object_id, None)
        if stale_ids:
            stale = set(stale_ids)
            self.state.relations = [
                relation
                for relation in self.state.relations
                if relation.subject_id not in stale and relation.object_id not in stale
            ]

    def _merge_relations(self, relations: list[WorldRelation]) -> None:
        indexed = {
            (relation.subject_id, relation.relation, relation.object_id): relation
            for relation in self.state.relations
        }
        for relation in relations:
            indexed[(relation.subject_id, relation.relation, relation.object_id)] = relation
        self.state.relations = list(indexed.values())
