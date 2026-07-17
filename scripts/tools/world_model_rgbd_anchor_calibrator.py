#!/usr/bin/env /usr/bin/python3
"""Fit a robust translation correction from paired privileged and RGB-D anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.world_model.visual_calibration import residual_norm, robust_translation_offset, translation_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit RGB-D/VLM base and map translation corrections from a shadow calibration set.")
    parser.add_argument("--reference", required=True, help="Privileged-shadow anchor JSONL used only for calibration.")
    parser.add_argument("--prediction", required=True, help="Paired RGB-D/VLM anchor JSONL used only for calibration.")
    parser.add_argument("--output", required=True, help="Calibration JSON written for --qwen-vl-calibration-file.")
    parser.add_argument("--key", default="sample_id", help="Frame pairing key, falling back to scene then input order.")
    parser.add_argument("--min-samples", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = _read_jsonl(_repo_path(args.reference))
    predictions = _read_jsonl(_repo_path(args.prediction))
    pairs = _pair_records(references, predictions, str(args.key))
    residuals = {"pose_base": [], "pose_map": []}
    for reference, prediction in pairs:
        for expected, observed in _match_objects(reference, prediction):
            for pose_key in residuals:
                residual = translation_residual(expected.get(pose_key), observed.get(pose_key))
                if residual is not None:
                    residuals[pose_key].append(residual)

    base_offset = robust_translation_offset(residuals["pose_base"])
    if base_offset is None or len(residuals["pose_base"]) < max(1, int(args.min_samples)):
        raise SystemExit(
            f"insufficient base-frame matches: {len(residuals['pose_base'])}; require --min-samples {max(1, int(args.min_samples))}"
        )
    map_offset = robust_translation_offset(residuals["pose_map"])
    report = {
        "schema": "sonic_rgbd_anchor_translation_calibration_v0",
        "calibration_split": "shadow_train_only",
        "reference": str(_repo_path(args.reference)),
        "prediction": str(_repo_path(args.prediction)),
        "paired_frames": len(pairs),
        "base_sample_count": len(residuals["pose_base"]),
        "map_sample_count": len(residuals["pose_map"]),
        "base_offset_m": base_offset,
        "map_offset_m": map_offset,
        "base_error_median_before_m": _median_norm(residuals["pose_base"]),
        "base_error_median_after_m": _median_norm(residuals["pose_base"], base_offset),
        "map_error_median_before_m": _median_norm(residuals["pose_map"]),
        "map_error_median_after_m": _median_norm(residuals["pose_map"], map_offset),
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "rgbd_anchor_calibration "
        f"base_samples={report['base_sample_count']} base_before={report['base_error_median_before_m']:.4f} "
        f"base_after={report['base_error_median_after_m']:.4f}"
    )
    print(f"Wrote RGB-D anchor calibration: {_relative(output)}")
    return 0


def _pair_records(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]], key: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(predictions):
        indexed.setdefault(_record_key(record, key, index), []).append(record)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, reference in enumerate(references):
        candidates = indexed.get(_record_key(reference, key, index), [])
        if candidates:
            pairs.append((reference, candidates.pop(0)))
    return pairs


def _match_objects(reference: dict[str, Any], prediction: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expected = _objects(reference)
    remaining = _objects(prediction)
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in expected:
        object_id = str(item.get("object_id") or item.get("id") or "")
        category = str(item.get("category") or item.get("object_category") or "")
        candidate = next((value for value in remaining if object_id and str(value.get("object_id") or value.get("id") or "") == object_id), None)
        if candidate is None:
            candidate = next((value for value in remaining if category and str(value.get("category") or value.get("object_category") or "") == category), None)
        if candidate is not None:
            remaining.remove(candidate)
            matches.append((item, candidate))
    return matches


def _objects(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("objects") if isinstance(record.get("objects"), list) else [record]
    return [item for item in value if isinstance(item, dict)]


def _record_key(record: dict[str, Any], key: str, index: int) -> str:
    for value in (record.get(key), record.get("scene"), record.get("frame_id")):
        if value is not None and str(value):
            return str(value)
    return f"line:{index}"


def _median_norm(residuals: list[list[float]], offset: list[float] | None = None) -> float | None:
    values = []
    for residual in residuals:
        corrected = [residual[index] - offset[index] for index in range(3)] if offset is not None else residual
        value = residual_norm(corrected)
        if value is not None:
            values.append(value)
    return sorted(values)[len(values) // 2] if values else None


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


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
