#!/usr/bin/env python3
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

from sonic_world.world_model.vlm_eval import evaluate_anchor_pairs, gate_anchor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL/RGB-D anchors against paired privileged anchors.")
    parser.add_argument("--reference", required=True, help="JSONL privileged-anchor reference file.")
    parser.add_argument("--prediction", required=True, help="JSONL Qwen-VL/RGB-D anchor file.")
    parser.add_argument("--key", default="sample_id", help="Pairing key. Falls back to scene then line order.")
    parser.add_argument("--output", default="reports/perception/vlm_anchor_eval.json")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-base-pose-error-m", type=float, default=0.08)
    parser.add_argument("--min-support-accuracy", type=float, default=0.90)
    parser.add_argument("--min-tracking-consistency", type=float, default=0.90)
    parser.add_argument("--min-target-region-recall", type=float, default=0.90)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a gate check fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = _read_jsonl(_repo_path(args.reference))
    predictions = _read_jsonl(_repo_path(args.prediction))
    pairs, unmatched_reference, unmatched_prediction = _pair_records(references, predictions, key=str(args.key))
    pair_metrics = [evaluate_anchor_pairs([reference], [prediction]) for reference, prediction in pairs]
    tracking_metrics = evaluate_anchor_pairs(
        [reference for reference, _prediction in pairs],
        [prediction for _reference, prediction in pairs],
    )
    metrics = _aggregate(
        pair_metrics,
        unmatched_reference=unmatched_reference,
        unmatched_prediction=unmatched_prediction,
        tracking_consistency=tracking_metrics.get("tracking_consistency"),
    )
    gate = gate_anchor_metrics(
        metrics,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_base_pose_error_m=args.max_base_pose_error_m,
        min_support_accuracy=args.min_support_accuracy,
        min_tracking_consistency=args.min_tracking_consistency,
        min_target_region_recall=args.min_target_region_recall,
    )
    report = {"schema": "sonic_vlm_anchor_eval_report_v0", "metrics": metrics, "gate": gate}
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"vlm_anchor_eval pairs={len(pairs)} passed={gate['passed']} failed={','.join(gate['failed_checks']) or '-'}")
    print(f"Wrote VLM anchor report: {_relative(output)}")
    return 0 if gate["passed"] or not args.strict else 1


def _pair_records(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]], *, key: str
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(predictions):
        indexed.setdefault(_record_key(record, key, index), []).append(record)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_reference: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        candidates = indexed.get(_record_key(reference, key, index), [])
        if not candidates:
            unmatched_reference.append(reference)
            continue
        pairs.append((reference, candidates.pop(0)))
    unmatched_prediction = [item for items in indexed.values() for item in items]
    return pairs, unmatched_reference, unmatched_prediction


def _aggregate(
    metrics: list[dict[str, Any]],
    *,
    unmatched_reference: list[dict[str, Any]],
    unmatched_prediction: list[dict[str, Any]],
    tracking_consistency: Any,
) -> dict[str, Any]:
    reference_count = sum(int(item["reference_object_count"]) for item in metrics) + _object_count(unmatched_reference)
    prediction_count = sum(int(item["prediction_object_count"]) for item in metrics) + _object_count(unmatched_prediction)
    matched_count = sum(int(item["matched_object_count"]) for item in metrics)
    values: dict[str, list[float]] = {}
    for item in metrics:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in {"reference_object_count", "prediction_object_count", "matched_object_count"}:
                values.setdefault(key, []).append(float(value))
    out: dict[str, Any] = {
        "paired_frames": len(metrics),
        "unmatched_reference_frames": len(unmatched_reference),
        "unmatched_prediction_frames": len(unmatched_prediction),
        "reference_object_count": reference_count,
        "prediction_object_count": prediction_count,
        "matched_object_count": matched_count,
        "precision": matched_count / prediction_count if prediction_count else None,
        "recall": matched_count / reference_count if reference_count else None,
    }
    for key, rows in values.items():
        out[key] = sum(rows) / len(rows) if rows else None
    out["tracking_consistency"] = tracking_consistency
    return out


def _object_count(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        objects = record.get("objects") if isinstance(record.get("objects"), list) else [record]
        total += sum(1 for item in objects if isinstance(item, dict))
    return total


def _record_key(record: dict[str, Any], key: str, index: int) -> str:
    for candidate in (record.get(key), record.get("scene"), record.get("frame_id")):
        if candidate is not None and str(candidate):
            return str(candidate)
    return f"line:{index}"


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
