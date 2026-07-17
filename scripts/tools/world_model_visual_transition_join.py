#!/usr/bin/env /usr/bin/python3
"""Attach time-aligned Qwen/RGB-D observations to physical residual transitions."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual-policy transitions from shadow Qwen/RGB-D anchors and physical outcomes.")
    parser.add_argument("--transitions", required=True, help="Output of world_model_episode_dataset.py.")
    parser.add_argument("--anchors", action="append", required=True, help="Qwen/RGB-D shadow-anchor JSONL; repeatable.")
    parser.add_argument("--output", default="reports/policy_data/physical_qwen_visual_residual_v0.jsonl")
    parser.add_argument("--summary", help="CSV-like JSON summary path; defaults to <output>.summary.json.")
    parser.add_argument("--max-skew-s", type=float, default=2.0)
    parser.add_argument("--require-visual-pose", action="store_true", default=True)
    parser.add_argument("--allow-missing-visual-pose", dest="require_visual_pose", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transitions = _read_jsonl(_path(args.transitions))
    anchors = [record for value in args.anchors for record in _read_jsonl(_path(value))]
    rows = []
    rejected = {"missing_object": 0, "stale_anchor": 0, "missing_visual_pose": 0, "missing_observation": 0}
    for transition in transitions:
        row, reason = _attach_visual(transition, anchors, max_skew_s=max(0.0, float(args.max_skew_s)))
        if row is None:
            rejected[reason or "missing_object"] = rejected.get(reason or "missing_object", 0) + 1
            continue
        if bool(args.require_visual_pose) and not row["visual_alignment"]["pose_available"]:
            rejected["missing_visual_pose"] += 1
            continue
        rows.append(row)
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema": "sonic_world_model_visual_transition_join_summary_v0",
        "input_transition_count": len(transitions),
        "visual_transition_count": len(rows),
        "rejected": rejected,
        "max_skew_s": float(args.max_skew_s),
        "visual_source": "qwen_rgbd_shadow",
        "deployment_status": "shadow_training_only",
    }
    summary_path = _path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(_relative(output))
    return 0


def _attach_visual(
    transition: dict[str, Any], anchors: list[dict[str, Any]], *, max_skew_s: float
) -> tuple[dict[str, Any] | None, str | None]:
    observation = transition.get("observation")
    if not isinstance(observation, dict):
        return None, "missing_observation"
    object_id = str((transition.get("policy") or {}).get("object_id") or "")
    if not object_id:
        return None, "missing_object"
    stamp = _float(transition.get("policy_stamp"), default=0.0)
    match = _nearest_object(anchors, object_id, stamp)
    if match is None:
        return None, "missing_object"
    record, item, skew = match
    if skew > max_skew_s:
        return None, "stale_anchor"
    visual = copy.deepcopy(observation)
    entity = visual.get("entity")
    context = visual.get("context")
    if not isinstance(entity, list) or not entity or not isinstance(entity[0], list) or len(entity[0]) < 3:
        return None, "missing_observation"
    if not isinstance(context, list) or len(context) < 24:
        return None, "missing_observation"
    pose = _pose(item.get("pose_base")) or _pose(item.get("pose_map"))
    if pose is not None:
        entity[0][0:3] = pose
        context[0:3] = pose
        # The target-relative state must be recomputed after replacing the object position.
        if len(entity) > 1 and isinstance(entity[1], list) and len(entity[1]) >= 3:
            context[3:6] = [float(entity[1][index]) - pose[index] for index in range(3)]
    context[12:18] = _visual_features(item)
    row = copy.deepcopy(transition)
    row["schema"] = "sonic_world_model_visual_residual_transition_v0"
    row["observation"] = visual
    row["visual_alignment"] = {
        "source": "qwen_rgbd_shadow",
        "sample_id": record.get("sample_id"),
        "object_id": object_id,
        "anchor_stamp": _record_stamp(record),
        "policy_stamp": stamp,
        "skew_s": round(skew, 6),
        "pose_available": pose is not None,
        "tracking_id": _property(item, "tracking_id"),
        "uncertainty": _property(item, "uncertainty"),
    }
    return row, None


def _nearest_object(
    anchors: list[dict[str, Any]], object_id: str, stamp: float
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    candidates = []
    for record in anchors:
        for item in _objects(record):
            identifier = str(item.get("object_id") or item.get("id") or "")
            if identifier != object_id:
                continue
            candidates.append((record, item, abs(_record_stamp(record) - stamp)))
    return min(candidates, key=lambda value: value[2]) if candidates else None


def _visual_features(item: dict[str, Any]) -> list[float]:
    props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
    uncertainty = _property(item, "uncertainty")
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    confidence = _clamp(_property(item, "confidence"), 0.0)
    source = str(item.get("source") or "").lower()
    visual_source = 1.0 if any(value in source for value in ("qwen", "vlm", "dino")) else 0.0
    tracked = 1.0 if _property(item, "tracking_id") else 0.0
    depth_mad = max(0.0, _float(uncertainty.get("depth_mad_m"), 1.0))
    quality = 1.0 / (1.0 + 20.0 * depth_mad)
    samples = _clamp(_float(uncertainty.get("depth_sample_count"), 0.0) / 49.0, 0.0)
    pose = 1.0 if _pose(item.get("pose_base")) or _pose(item.get("pose_map")) else 0.0
    return [confidence, visual_source, tracked, quality, samples, pose]


def _property(item: dict[str, Any], key: str) -> Any:
    if key in item:
        return item.get(key)
    props = item.get("properties")
    return props.get(key) if isinstance(props, dict) else None


def _objects(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("objects") if isinstance(record.get("objects"), list) else [record]
    return [item for item in raw if isinstance(item, dict)]


def _record_stamp(record: dict[str, Any]) -> float:
    return _float(record.get("recorded_at"), default=_float(record.get("stamp"), default=0.0))


def _pose(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("position") or value.get("xyz")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        pose = [float(raw[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return pose if all(math.isfinite(value) for value in pose) else None


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: Any, default: float) -> float:
    return min(1.0, max(0.0, _float(value, default)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not an object")
        rows.append(value)
    return rows


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
