#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.policies.linear import (
    TARGET_PATHS,
    feature_names,
    get_path,
    observation_feature_values,
    vector_from_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generalizing linear task-space policy from rollout outcomes.")
    parser.add_argument("--input", default="reports/policy_outcomes/sonic_policy_outcomes.jsonl")
    parser.add_argument("--output", default="reports/policy_models/sonic_linear_task_policy_v0.json")
    parser.add_argument("--metrics", default="reports/policy_models/sonic_linear_task_policy_v0_metrics.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--held-out-category", action="append", default=[])
    parser.add_argument("--min-samples", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(_repo_path(args.input))
    samples = [_sample(row) for row in rows]
    samples = [sample for sample in samples if sample is not None]
    if len(samples) < max(2, int(args.min_samples)):
        raise SystemExit(f"Need at least {args.min_samples} matched policy samples; found {len(samples)}")
    categories = sorted({str(sample["values"].get("category") or "unknown") for sample in samples})
    names = feature_names(categories)
    train, val = _group_split(samples, ratio=float(args.val_ratio), held_out=set(args.held_out_category))
    if not train:
        raise SystemExit("Training split is empty")
    targets: dict[str, dict[str, Any]] = {}
    target_metrics: dict[str, Any] = {}
    for path in TARGET_PATHS:
        fitted = _fit_target(train, val, path=path, names=names, ridge=float(args.ridge))
        if fitted is None:
            continue
        record, metrics = fitted
        targets[path] = record
        target_metrics[path] = metrics
    if not targets:
        raise SystemExit("No numeric policy targets could be trained")

    source = _repo_path(args.input)
    model = {
        "schema": "sonic_linear_task_policy_v0",
        "version": "v0",
        "model_id": Path(args.output).stem,
        "policy_id": Path(args.output).stem,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "feature_names": names,
        "targets": targets,
        "manifest": {
            "source": _rel(source),
            "source_sha256": _sha256(source),
            "split_strategy": "task_id_group_hash_with_optional_held_out_category",
            "train_task_ids": sorted({sample["task_id"] for sample in train}),
            "val_task_ids": sorted({sample["task_id"] for sample in val}),
            "held_out_categories": sorted(set(args.held_out_category)),
            "uses_task_id_feature": False,
            "controller_boundary": "frozen_sonic_low_level",
        },
    }
    checkpoint_hash = hashlib.sha256(json.dumps(model, sort_keys=True).encode("utf-8")).hexdigest()
    model["manifest"]["checkpoint_hash"] = checkpoint_hash
    metrics = {
        "schema": "sonic_linear_task_policy_metrics_v0",
        "train_samples": len(train),
        "val_samples": len(val),
        "feature_count": len(names),
        "target_count": len(targets),
        "targets": target_metrics,
        "split": model["manifest"],
    }
    output = _repo_path(args.output)
    metric_path = _repo_path(args.metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metric_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"linear_policy train={len(train)} val={len(val)} features={len(names)} "
        f"targets={len(targets)} checkpoint={checkpoint_hash[:12]}"
    )
    print(f"Wrote model: {_rel(output)}")
    print(f"Wrote metrics: {_rel(metric_path)}")
    return 0


def _sample(row: dict[str, Any]) -> dict[str, Any] | None:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else None
    action = row.get("teacher_action") if isinstance(row.get("teacher_action"), dict) else None
    if observation is None or action is None:
        return None
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
    task_id = str(observation.get("task_id") or action.get("task_id") or rollout.get("task_id") or "unknown")
    dense = max(0.0, min(1.0, float(outcome.get("dense_score") or 0.0)))
    weight = 0.15 + 0.55 * dense + (0.30 if outcome.get("success") else 0.0)
    return {
        "task_id": task_id,
        "values": observation_feature_values(observation, action),
        "action": action,
        "weight": max(0.05, min(1.0, weight)),
    }


def _group_split(
    samples: list[dict[str, Any]],
    *,
    ratio: float,
    held_out: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = int(max(0.0, min(0.8, ratio)) * 10000)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for sample in samples:
        category = str(sample["values"].get("category") or "")
        bucket = int(hashlib.sha1(sample["task_id"].encode("utf-8")).hexdigest()[:8], 16) % 10000
        (val if category in held_out or bucket < threshold else train).append(sample)
    if not val and len(train) > 1:
        task = train[-1]["task_id"]
        moved = [sample for sample in train if sample["task_id"] == task]
        train = [sample for sample in train if sample["task_id"] != task]
        val.extend(moved)
    return train, val


def _fit_target(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    path: str,
    names: list[str],
    ridge: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    rows = [(sample, get_path(sample["action"], path)) for sample in train]
    rows = [(sample, target) for sample, target in rows if target is not None]
    if len(rows) < 2:
        return None
    x = np.stack([vector_from_values(sample["values"], names) for sample, _target in rows])
    y = np.asarray([target for _sample, target in rows], dtype=np.float64)
    weights = np.sqrt(np.asarray([sample["weight"] for sample, _target in rows], dtype=np.float64))
    xw, yw = x * weights[:, None], y * weights
    penalty = max(0.0, ridge) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(xw.T @ xw + penalty) @ (xw.T @ yw)
    train_mae = float(np.mean(np.abs(x @ coefficients - y)))
    val_rows = [(sample, get_path(sample["action"], path)) for sample in val]
    val_rows = [(sample, target) for sample, target in val_rows if target is not None]
    val_mae = None
    if val_rows:
        vx = np.stack([vector_from_values(sample["values"], names) for sample, _target in val_rows])
        vy = np.asarray([target for _sample, target in val_rows], dtype=np.float64)
        val_mae = float(np.mean(np.abs(vx @ coefficients - vy)))
    record = {
        "weights": [round(float(value), 10) for value in coefficients],
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "sample_count": len(rows),
    }
    metrics = {"train_mae": train_mae, "val_mae": val_mae, "train_count": len(rows), "val_count": len(val_rows)}
    return record, metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
