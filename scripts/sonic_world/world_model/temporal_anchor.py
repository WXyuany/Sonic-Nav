from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import math
from statistics import median
from typing import Any


class TemporalAnchorFilter:
    """Stabilize generic object anchors with a bounded coordinate-wise median."""

    def __init__(self, *, window_size: int = 3, min_observations: int = 3):
        self.window_size = max(1, int(window_size))
        self.min_observations = max(1, int(min_observations))
        self._tracks: dict[str, deque[dict[str, Any]]] = {}

    def update(self, anchor: dict[str, Any]) -> dict[str, Any]:
        output = deepcopy(anchor)
        raw_objects = anchor.get("objects")
        if not isinstance(raw_objects, list):
            output["objects"] = []
            output["relations"] = []
            return output

        stable_objects: list[dict[str, Any]] = []
        stable_ids: set[str] = set()
        for raw in raw_objects:
            if not isinstance(raw, dict):
                continue
            key = _track_key(raw)
            track = self._tracks.setdefault(key, deque(maxlen=self.window_size))
            track.append(deepcopy(raw))
            if len(track) < self.min_observations:
                continue
            stable = _median_object(list(track))
            stable_objects.append(stable)
            stable_ids.add(str(stable.get("object_id") or ""))

        output["objects"] = stable_objects
        relations = anchor.get("relations")
        output["relations"] = [
            relation
            for relation in relations
            if isinstance(relation, dict) and str(relation.get("subject_id") or relation.get("subject") or "") in stable_ids
        ] if isinstance(relations, list) else []
        properties = dict(output.get("properties") or {})
        properties["temporal_filter"] = {
            "window_size": self.window_size,
            "min_observations": self.min_observations,
            "stable_object_count": len(stable_objects),
        }
        output["properties"] = properties
        return output


def _track_key(record: dict[str, Any]) -> str:
    for key in ("tracking_id", "object_id", "id", "category"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return "object"


def _median_object(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = deepcopy(records[-1])
    for pose_key in ("pose_base", "pose_map", "pose_camera"):
        positions = [_position(record.get(pose_key)) for record in records]
        valid = [position for position in positions if position is not None]
        if not valid:
            continue
        latest_pose = output.get(pose_key) if isinstance(output.get(pose_key), dict) else {}
        output[pose_key] = {
            **latest_pose,
            "position": [float(median([position[index] for position in valid])) for index in range(3)],
        }
    supports = [str(record.get("support") or "") for record in records if str(record.get("support") or "")]
    if supports:
        output["support"] = Counter(supports).most_common(1)[0][0]
    return output


def _position(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("position") or value.get("xyz")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        position = [float(raw[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return position if all(math.isfinite(item) for item in position) else None
