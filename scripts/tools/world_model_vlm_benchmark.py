#!/usr/bin/env python3
"""Aggregate paired visual-anchor evaluations across multiple task rollouts."""
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

from sonic_world.world_model import evaluate_anchor_pairs, gate_anchor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate strict visual/RGB-D anchor metrics across task rollout prefixes.")
    parser.add_argument("--perception-dir", default="reports/perception")
    parser.add_argument("--prefix", action="append", default=[], help="Rollout prefix to include; repeat as needed. Defaults to all matching references.")
    parser.add_argument("--reference-suffix", default="_privileged_anchors.jsonl")
    parser.add_argument("--prediction-suffix", default="_qwen_rgbd_anchors.jsonl")
    parser.add_argument("--output", default="reports/perception/vlm_multitask_eval.json")
    parser.add_argument("--min-task-count", type=int, default=20)
    parser.add_argument("--min-paired-frames", type=int, default=100)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-base-pose-error-m", type=float, default=0.08)
    parser.add_argument("--min-support-accuracy", type=float, default=0.90)
    parser.add_argument("--min-tracking-consistency", type=float, default=0.90)
    parser.add_argument("--min-target-region-recall", type=float, default=0.90)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    perception_dir = _repo_path(args.perception_dir)
    pairs = _resolve_pairs(perception_dir, args)
    if not pairs:
        raise SystemExit(f"No paired anchor files found under {perception_dir}")

    task_rows: list[dict[str, Any]] = []
    all_pair_metrics: list[dict[str, Any]] = []
    all_references: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    unmatched_references: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    for prefix, reference_path, prediction_path in pairs:
        references, predictions = _read_jsonl(reference_path), _read_jsonl(prediction_path)
        paired, missing_reference, missing_prediction = _pair_records(references, predictions)
        per_pair_metrics = [evaluate_anchor_pairs([reference], [prediction]) for reference, prediction in paired]
        metrics = _aggregate_metrics(per_pair_metrics, missing_reference, missing_prediction)
        task_rows.append(
            {
                "prefix": prefix,
                "reference": _relative(reference_path),
                "prediction": _relative(prediction_path),
                "metrics": metrics,
            }
        )
        all_pair_metrics.extend(per_pair_metrics)
        all_references.extend(reference for reference, _prediction in paired)
        all_predictions.extend(prediction for _reference, prediction in paired)
        unmatched_references.extend(missing_reference)
        unmatched_predictions.extend(missing_prediction)

    metrics = _aggregate_metrics(all_pair_metrics, unmatched_references, unmatched_predictions)
    tracking = evaluate_anchor_pairs(all_references, all_predictions).get("tracking_consistency") if all_references else None
    metrics["tracking_consistency"] = tracking
    gate = gate_anchor_metrics(
        metrics,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_base_pose_error_m=args.max_base_pose_error_m,
        min_support_accuracy=args.min_support_accuracy,
        min_tracking_consistency=args.min_tracking_consistency,
        min_target_region_recall=args.min_target_region_recall,
    )
    gate["checks"]["minimum_task_count"] = len(task_rows) >= max(1, int(args.min_task_count))
    gate["checks"]["minimum_paired_frames"] = int(metrics["paired_frames"]) >= max(1, int(args.min_paired_frames))
    gate["failed_checks"] = [name for name, passed in gate["checks"].items() if not passed]
    gate["passed"] = not gate["failed_checks"]
    report = {
        "schema": "sonic_vlm_anchor_eval_report_v0",
        "metrics": metrics,
        "gate": gate,
        "aggregate": {
            "task_count": len(task_rows),
            "minimum_task_count": max(1, int(args.min_task_count)),
            "minimum_paired_frames": max(1, int(args.min_paired_frames)),
            "tasks": task_rows,
        },
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"vlm_multitask_eval tasks={len(task_rows)} pairs={metrics['paired_frames']} "
        f"passed={gate['passed']} failed={','.join(gate['failed_checks']) or '-'}"
    )
    print(f"Wrote VLM multitask report: {_relative(output)}")
    return 0 if gate["passed"] or not args.strict else 1


def _resolve_pairs(directory: Path, args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    suffix = str(args.reference_suffix)
    prefixes = [str(item) for item in args.prefix if str(item)]
    if not prefixes:
        prefixes = sorted(path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}") if path.name.endswith(suffix))
    pairs = []
    for prefix in prefixes:
        reference, prediction = directory / f"{prefix}{suffix}", directory / f"{prefix}{args.prediction_suffix}"
        if not reference.is_file() or not prediction.is_file():
            raise ValueError(f"missing paired anchors for {prefix}: reference={reference.is_file()} prediction={prediction.is_file()}")
        pairs.append((prefix, reference, prediction))
    return pairs


def _pair_records(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for index, prediction in enumerate(predictions):
        indexed.setdefault(_record_key(prediction, index), []).append(prediction)
    paired, missing_reference = [], []
    for index, reference in enumerate(references):
        bucket = indexed.get(_record_key(reference, index), [])
        if bucket:
            paired.append((reference, bucket.pop(0)))
        else:
            missing_reference.append(reference)
    return paired, missing_reference, [item for items in indexed.values() for item in items]


def _aggregate_metrics(
    metrics: list[dict[str, Any]], missing_reference: list[dict[str, Any]], missing_prediction: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_count = sum(int(item["reference_object_count"]) for item in metrics) + _object_count(missing_reference)
    prediction_count = sum(int(item["prediction_object_count"]) for item in metrics) + _object_count(missing_prediction)
    matched_count = sum(int(item["matched_object_count"]) for item in metrics)
    numeric: dict[str, list[float]] = {}
    excluded = {"reference_object_count", "prediction_object_count", "matched_object_count", "tracking_consistency"}
    for item in metrics:
        for key, value in item.items():
            if key not in excluded and isinstance(value, (float, int)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    result: dict[str, Any] = {
        "paired_frames": len(metrics),
        "unmatched_reference_frames": len(missing_reference),
        "unmatched_prediction_frames": len(missing_prediction),
        "reference_object_count": reference_count,
        "prediction_object_count": prediction_count,
        "matched_object_count": matched_count,
        "precision": matched_count / prediction_count if prediction_count else None,
        "recall": matched_count / reference_count if reference_count else None,
    }
    result.update({key: sum(values) / len(values) for key, values in numeric.items() if values})
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} is not an object")
        rows.append(payload)
    return rows


def _record_key(record: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "scene", "frame_id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return f"line:{index}"


def _object_count(records: list[dict[str, Any]]) -> int:
    return sum(len(item.get("objects")) if isinstance(item.get("objects"), list) else 1 for item in records)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
