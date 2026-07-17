from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable


def apply_translation_offset(pose: dict[str, Any], offset: Iterable[float]) -> dict[str, Any]:
    """Return a pose translated by a validated three-dimensional offset."""

    values = _vector(offset)
    position = _position(pose)
    if values is None or position is None:
        raise ValueError("calibration requires a pose position and a three-dimensional offset")
    corrected = dict(pose)
    corrected["position"] = [position[index] + values[index] for index in range(3)]
    return corrected


def translation_residual(reference: dict[str, Any], prediction: dict[str, Any]) -> list[float] | None:
    expected, observed = _position(reference), _position(prediction)
    if expected is None or observed is None:
        return None
    return [expected[index] - observed[index] for index in range(3)]


def robust_translation_offset(residuals: Iterable[Iterable[float]]) -> list[float] | None:
    rows = [values for values in (_vector(item) for item in residuals) if values is not None]
    if not rows:
        return None
    return [float(median(row[index] for row in rows)) for index in range(3)]


def residual_norm(residual: Iterable[float] | None) -> float | None:
    values = _vector(residual)
    return math.sqrt(sum(value * value for value in values)) if values is not None else None


def _position(pose: Any) -> list[float] | None:
    if not isinstance(pose, dict):
        return None
    return _vector(pose.get("position") or pose.get("xyz"))


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = [float(value[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None
