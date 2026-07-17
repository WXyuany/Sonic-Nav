from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class TaskRequest:
    verb: str
    object_id: str | None = None
    object_category: str | None = None
    target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRequest":
        verb = _string_first(payload, "verb", "task", "action", "intent")
        if verb is None:
            raise ValueError("task request missing verb/task/action")
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("task request metadata must be an object")
        request_id = _string_first(payload, "request_id", "id")
        if request_id is not None and "request_id" not in metadata:
            metadata = {**metadata, "request_id": request_id}
        return cls(
            verb=_normalize_verb(verb),
            object_id=_string_first(payload, "object_id", "object", "object_name"),
            object_category=_string_first(payload, "object_category", "category", "class"),
            target_id=_string_first(payload, "target_id", "target", "destination", "goal"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "object_id": self.object_id,
            "object_category": self.object_category,
            "target_id": self.target_id,
            "metadata": dict(self.metadata),
        }


def task_request_from_json(text: str) -> TaskRequest:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("task request JSON must be an object")
    return TaskRequest.from_dict(payload)


def task_request_to_json(request: TaskRequest) -> str:
    return json.dumps(request.to_dict(), separators=(",", ":"))


def _string_first(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_verb(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
