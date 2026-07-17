from __future__ import annotations

import math
from statistics import median
from typing import Any


def evaluate_anchor_pairs(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate VLM/RGB-D anchors against paired privileged anchors."""

    reference_objects = _objects(references)
    prediction_objects = _objects(predictions)
    matches = _match(reference_objects, prediction_objects)
    pose_errors = [
        _distance(_pose_position(reference, "pose_base"), _pose_position(prediction, "pose_base"))
        for reference, prediction in matches
    ]
    pose_errors = [value for value in pose_errors if value is not None]
    support_pairs = [
        (str(reference.get("support") or ""), str(prediction.get("support") or ""))
        for reference, prediction in matches
        if reference.get("support")
    ]
    tracking = _tracking_consistency(matches)
    reference_targets = sum(1 for item in reference_objects if _category(item) == "place_target")
    matched_targets = sum(1 for reference, _prediction in matches if _category(reference) == "place_target")
    matched_count = len(matches)
    return {
        "schema": "sonic_vlm_anchor_eval_v0",
        "reference_object_count": len(reference_objects),
        "prediction_object_count": len(prediction_objects),
        "matched_object_count": matched_count,
        "precision": _ratio(matched_count, len(prediction_objects)),
        "recall": _ratio(matched_count, len(reference_objects)),
        "base_pose_coverage": _ratio(len(pose_errors), matched_count),
        "base_pose_error_mean_m": _mean(pose_errors),
        "base_pose_error_median_m": _median(pose_errors),
        "support_accuracy": _ratio(sum(1 for expected, actual in support_pairs if _support_equivalent(expected, actual)), len(support_pairs)),
        "support_coverage": _ratio(len(support_pairs), matched_count),
        "reference_target_count": reference_targets,
        "target_region_recall": _ratio(matched_targets, reference_targets),
        "tracking_consistency": tracking,
        "uncertainty_coverage": _ratio(
            sum(1 for _reference, prediction in matches if isinstance(_property(prediction, "uncertainty"), dict)),
            matched_count,
        ),
    }


def gate_anchor_metrics(
    metrics: dict[str, Any],
    *,
    min_precision: float,
    min_recall: float,
    max_base_pose_error_m: float,
    min_support_accuracy: float,
    min_tracking_consistency: float,
    min_target_region_recall: float,
) -> dict[str, Any]:
    checks = {
        "precision": _at_least(metrics.get("precision"), min_precision),
        "recall": _at_least(metrics.get("recall"), min_recall),
        "base_pose_error_median_m": _at_most(metrics.get("base_pose_error_median_m"), max_base_pose_error_m),
        "support_accuracy": _at_least(metrics.get("support_accuracy"), min_support_accuracy),
        "tracking_consistency": _at_least(metrics.get("tracking_consistency"), min_tracking_consistency),
        "target_region_recall": (
            int(metrics.get("reference_target_count") or 0) == 0
            or _at_least(metrics.get("target_region_recall"), min_target_region_recall)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def _objects(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for anchor in anchors:
        raw = anchor.get("objects") if isinstance(anchor.get("objects"), list) else [anchor]
        objects.extend(item for item in raw if isinstance(item, dict))
    return objects


def _match(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    unmatched = list(predictions)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference in references:
        object_id = str(reference.get("object_id") or reference.get("id") or "")
        category = _category(reference)
        candidate = next(
            (item for item in unmatched if object_id and str(item.get("object_id") or item.get("id") or "") == object_id),
            None,
        )
        if candidate is None:
            candidate = next((item for item in unmatched if _category(item) == category), None)
        if candidate is not None:
            unmatched.remove(candidate)
            pairs.append((reference, candidate))
    return pairs


def _tracking_consistency(matches: list[tuple[dict[str, Any], dict[str, Any]]]) -> float | None:
    by_reference: dict[str, set[str]] = {}
    for reference, prediction in matches:
        object_id = str(reference.get("object_id") or reference.get("id") or "")
        tracking_id = _property_string(prediction, "tracking_id", "track_id")
        if not object_id or not tracking_id:
            continue
        by_reference.setdefault(object_id, set()).add(tracking_id)
    if not by_reference:
        return None
    stable = sum(1 for values in by_reference.values() if len(values) == 1)
    return stable / len(by_reference)


def _pose_position(record: dict[str, Any], key: str) -> list[float] | None:
    pose = record.get(key)
    if not isinstance(pose, dict):
        return None
    position = pose.get("position") or pose.get("xyz")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        return [float(position[0]), float(position[1]), float(position[2])]
    except (TypeError, ValueError):
        return None


def _distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _category(record: dict[str, Any]) -> str:
    return str(record.get("category") or record.get("object_category") or "object")


def _support_equivalent(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    return _support_kind(expected) == _support_kind(actual)


def _support_kind(value: str) -> str:
    text = str(value).lower()
    if "table" in text or "support" in text or "counter" in text:
        return "table"
    if "shelf" in text:
        return "shelf"
    if "floor" in text:
        return "floor"
    return text


def _property(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record.get(key)
    properties = record.get("properties")
    return properties.get(key) if isinstance(properties, dict) else None


def _property_string(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _property(record, key)
        if value is not None and str(value):
            return str(value)
    return ""


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _at_least(value: Any, threshold: float) -> bool:
    if value is None:
        return float(threshold) <= 0.0
    return float(value) >= float(threshold)


def _at_most(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= float(threshold)
