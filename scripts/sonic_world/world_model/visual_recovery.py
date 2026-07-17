from __future__ import annotations

import time


class VisualRecoveryBudget:
    """Bound automatic re-observation retries for missing visual objects."""

    def __init__(self, expected_object_ids: list[str], *, max_attempts: int = 2, cooldown_s: float = 1.0):
        self.expected_object_ids = {str(value).strip() for value in expected_object_ids if str(value).strip()}
        self.max_attempts = max(0, int(max_attempts))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._attempts: dict[str, int] = {}
        self._last_request: dict[str, float] = {}

    def observe(self, object_ids: set[str] | list[str] | tuple[str, ...]) -> set[str]:
        observed = {str(value).strip() for value in object_ids if str(value).strip()}
        for object_id in self.expected_object_ids & observed:
            self._attempts.pop(object_id, None)
            self._last_request.pop(object_id, None)
        return self.expected_object_ids - observed

    def request(self, object_id: str, *, now: float | None = None) -> int | None:
        object_id = str(object_id).strip()
        if not object_id or object_id not in self.expected_object_ids:
            return None
        now = time.monotonic() if now is None else float(now)
        attempts = self._attempts.get(object_id, 0)
        if attempts >= self.max_attempts:
            return None
        previous = self._last_request.get(object_id)
        if previous is not None and now - previous < self.cooldown_s:
            return None
        attempts += 1
        self._attempts[object_id] = attempts
        self._last_request[object_id] = now
        return attempts

    def attempts(self, object_id: str) -> int:
        return self._attempts.get(str(object_id).strip(), 0)
