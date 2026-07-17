#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


DEFAULT_INPUT = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"
DEFAULT_OUTPUT_DIR = "reports/policy_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a first task-level policy-memory baseline from Sonic policy outcomes. "
            "This trains the high-level task/skill layer only; SONIC low-level control remains frozen."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="task_policy_memory_v0")
    parser.add_argument("--min-positive-score", type=float, default=0.45)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--include-failures", action="store_true", default=True)
    parser.add_argument("--positive-only", dest="include_failures", action="store_false")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.input)
    if not rows:
        raise SystemExit("No policy outcome rows found.")

    examples = [_training_example(row, min_positive_score=float(args.min_positive_score)) for row in rows]
    examples = [item for item in examples if item is not None]
    if not args.include_failures:
        examples = [item for item in examples if item["label"] == "positive"]
    if not examples:
        raise SystemExit("No usable training examples after filtering.")

    train, val = _split_examples(examples, val_ratio=float(args.val_ratio))
    model = _build_policy_memory(examples, train_count=len(train), val_count=len(val), source=_rel(_repo_path(args.input)))

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(args.name)
    model_path = output_dir / f"{stem}.json"
    train_path = output_dir / f"{stem}_train.jsonl"
    val_path = output_dir / f"{stem}_val.jsonl"
    csv_path = output_dir / f"{stem}_summary.csv"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    _write_summary_csv(csv_path, examples)

    if args.print_json:
        print(json.dumps(model, indent=2, sort_keys=True))
    else:
        _print_summary(model)
    print(f"\nWrote task policy model: {_rel(model_path)}")
    print(f"Wrote train split: {_rel(train_path)}")
    print(f"Wrote val split: {_rel(val_path)}")
    print(f"Wrote training summary: {_rel(csv_path)}")
    return 0


def _training_example(row: dict[str, Any], *, min_positive_score: float) -> dict[str, Any] | None:
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else None
    action = row.get("teacher_action") if isinstance(row.get("teacher_action"), dict) else None
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
    match = row.get("match") if isinstance(row.get("match"), dict) else {}
    if action is None:
        return None
    task_id = _training_task_id(action=action, rollout=rollout, observation=observation, match=match)
    policy_task_id = str(action.get("task_id") or (observation or {}).get("task_id") or "")
    rollout_task_id = str(rollout.get("task_id") or "")
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    intent = action.get("task_intent") if isinstance(action.get("task_intent"), dict) else {}
    dense_score = float(outcome.get("dense_score") or 0.0)
    success = bool(outcome.get("success"))
    clean = bool(outcome.get("clean_success"))
    recovered = bool(outcome.get("recovered_success"))
    label = "positive" if success and dense_score >= min_positive_score else "negative"
    weight = _sample_weight(success=success, clean=clean, recovered=recovered, dense_score=dense_score)
    return {
        "schema": "sonic_task_policy_training_example_v0",
        "example_id": str(row.get("link_id") or f"{task_id}:{rollout.get('run_id') or _stable_hash(row)}"),
        "task_id": task_id,
        "demo_kind": str(rollout.get("demo_kind") or intent.get("demo_kind") or ""),
        "object_category": str(intent.get("object_category") or metadata.get("object_category") or ""),
        "grasp_affordance": str(metadata.get("grasp_affordance") or ""),
        "label": label,
        "sample_weight": weight,
        "dense_score": round(dense_score, 4),
        "success": success,
        "quality": str(outcome.get("quality") or ""),
        "primary_issue": str(outcome.get("primary_issue") or ""),
        "terminal_stage": str(outcome.get("terminal_stage") or ""),
        "retry_count": int(outcome.get("retry_count") or 0),
        "run_id": str(rollout.get("run_id") or ""),
        "match": match,
        "policy_task_id": policy_task_id,
        "rollout_task_id": rollout_task_id,
        "observation": observation,
        "action": action,
        "outcome": outcome,
        "metadata": {
            "controller_boundary": "frozen_sonic_low_level",
            "training_scope": "task_and_skill_policy_only",
            "trainable_outputs": [
                "task_intent",
                "object_target_anchors",
                "skill_selection",
                "base_goal",
                "hand_pose_target",
                "wrist_target",
                "grasp_close_ratio",
                "grasp_offsets",
                "lift_place_targets",
                "recovery_decision",
            ],
        },
    }


def _training_task_id(
    *,
    action: dict[str, Any],
    rollout: dict[str, Any],
    observation: dict[str, Any] | None,
    match: dict[str, Any],
) -> str:
    policy_task = str(action.get("task_id") or (observation or {}).get("task_id") or "")
    rollout_task = str(rollout.get("task_id") or "")
    if match.get("type") == "task_id_exact" and policy_task:
        return policy_task
    # Compatible-demo matches are useful fallback data, but they are not proof that
    # the matched suite task itself was rolled out. Keep them under the actual
    # rollout task id so exact task coverage stays honest.
    return rollout_task or policy_task or "unknown"


def _sample_weight(*, success: bool, clean: bool, recovered: bool, dense_score: float) -> float:
    if clean:
        return 1.0
    if recovered:
        return round(max(0.45, min(0.85, dense_score)), 4)
    if success:
        return round(max(0.30, min(0.65, dense_score)), 4)
    return round(max(0.05, min(0.35, dense_score * 0.5)), 4)


def _split_examples(examples: list[dict[str, Any]], *, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ratio = max(0.0, min(0.8, val_ratio))
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    cutoff = int(ratio * 10000)
    for item in examples:
        bucket = int(_stable_hash(item["example_id"])[:8], 16) % 10000
        target = val if bucket < cutoff else train
        split_item = dict(item)
        split_item["split"] = "val" if target is val else "train"
        target.append(split_item)
    if not train and val:
        train.append(val.pop())
        train[-1]["split"] = "train"
    return train, val


def _build_policy_memory(examples: list[dict[str, Any]], *, train_count: int, val_count: int, source: str) -> dict[str, Any]:
    exact: dict[str, list[dict[str, Any]]] = {}
    fallback: dict[str, list[dict[str, Any]]] = {}
    issues: dict[str, dict[str, Any]] = {}
    for item in examples:
        exact.setdefault(item["task_id"], []).append(item)
        fallback.setdefault(_fallback_key(item), []).append(item)
        issue = item.get("primary_issue") or "none"
        bucket = issues.setdefault(
            str(issue),
            {
                "issue": str(issue),
                "count": 0,
                "positive": 0,
                "negative": 0,
                "avg_dense_score": 0.0,
                "stages": {},
            },
        )
        bucket["count"] += 1
        bucket["positive" if item["label"] == "positive" else "negative"] += 1
        bucket["avg_dense_score"] += float(item["dense_score"])
        stage = str(item.get("terminal_stage") or "unknown")
        bucket["stages"][stage] = bucket["stages"].get(stage, 0) + 1

    exact_policy = {task_id: _memory_entry(items) for task_id, items in sorted(exact.items())}
    fallback_policy = {key: _memory_entry(items) for key, items in sorted(fallback.items())}
    for bucket in issues.values():
        if bucket["count"]:
            bucket["avg_dense_score"] = round(float(bucket["avg_dense_score"]) / int(bucket["count"]), 4)
        bucket["stages"] = dict(sorted(bucket["stages"].items()))

    return {
        "schema": "sonic_task_policy_memory_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "controller_boundary": "frozen_sonic_low_level",
        "training_scope": "task_and_skill_policy_only",
        "example_count": len(examples),
        "train_count": train_count,
        "val_count": val_count,
        "positive_count": sum(1 for item in examples if item["label"] == "positive"),
        "negative_count": sum(1 for item in examples if item["label"] == "negative"),
        "exact_task_policy": exact_policy,
        "fallback_policy": fallback_policy,
        "failure_feedback": sorted(issues.values(), key=lambda item: (-item["count"], item["issue"])),
    }


def _memory_entry(items: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        items,
        key=lambda item: (
            item["label"] != "positive",
            -float(item["dense_score"]),
            int(item.get("retry_count") or 0),
            item.get("example_id") or "",
        ),
    )
    best = ranked[0]
    positives = [item for item in items if item["label"] == "positive"]
    return {
        "task_id": best["task_id"],
        "demo_kind": best["demo_kind"],
        "object_category": best["object_category"],
        "grasp_affordance": best["grasp_affordance"],
        "candidate_count": len(items),
        "positive_count": len(positives),
        "best_example_id": best["example_id"],
        "best_run_id": best["run_id"],
        "best_dense_score": best["dense_score"],
        "best_quality": best["quality"],
        "best_retry_count": best["retry_count"],
        "recommended_action": best["action"],
    }


def _fallback_key(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("demo_kind") or "unknown"),
            str(item.get("grasp_affordance") or "unknown"),
            str(item.get("object_category") or "unknown"),
        ]
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "example_id",
        "task_id",
        "policy_task_id",
        "rollout_task_id",
        "demo_kind",
        "object_category",
        "grasp_affordance",
        "label",
        "sample_weight",
        "dense_score",
        "success",
        "quality",
        "primary_issue",
        "terminal_stage",
        "retry_count",
        "run_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_summary(model: dict[str, Any]) -> None:
    print(
        "task_policy_model="
        f"{model['schema']} examples={model['example_count']} "
        f"positive={model['positive_count']} negative={model['negative_count']} "
        f"train={model['train_count']} val={model['val_count']}"
    )
    print(f"{'key':42s} {'cand':>5s} {'pos':>4s} {'score':>6s} {'retry':>5s} best_run")
    for key, entry in list(model["exact_task_policy"].items())[:24]:
        print(
            f"{key[:42]:42s} "
            f"{int(entry['candidate_count']):>5d} "
            f"{int(entry['positive_count']):>4d} "
            f"{float(entry['best_dense_score']):>6.2f} "
            f"{int(entry['best_retry_count']):>5d} "
            f"{entry.get('best_run_id') or '-'}"
        )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = _repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"input JSONL not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"bad JSONL row at {p}:{line_no}")
            rows.append(payload)
    return rows


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
