#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


DEFAULT_MODEL = "reports/policy_models/task_policy_memory_v0.json"
DEFAULT_OUTPUT = "reports/policy_data/task_policy_memory_actions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export recommended high-level actions from a Sonic task-policy memory model. "
            "The exported JSONL can be passed to manipulation demos with --policy-action-json."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", action="append", help="Only export selected task ids.")
    parser.add_argument("--demo", choices=["ball", "box"], help="Only export this demo kind.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", help="CSV summary path. Defaults to <output>.csv.")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = _read_model(args.model)
    rows = _export_rows(model, tasks=set(args.task or ()), demo=args.demo)
    if not rows:
        raise SystemExit("No policy actions selected.")

    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    _write_csv(summary_path, rows)
    if args.print_json:
        print(json.dumps({"actions": rows}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(rows, source=_rel(_repo_path(args.model)))
    print(f"\nWrote policy action JSONL: {_rel(output)}")
    print(f"Wrote policy action summary: {_rel(summary_path)}")
    return 0


def _export_rows(model: dict[str, Any], *, tasks: set[str], demo: str | None) -> list[dict[str, Any]]:
    exact = model.get("exact_task_policy") if isinstance(model.get("exact_task_policy"), dict) else {}
    rows: list[dict[str, Any]] = []
    for task_id, entry in sorted(exact.items()):
        if tasks and str(task_id) not in tasks:
            continue
        if demo and str(entry.get("demo_kind") or "") != demo:
            continue
        action = entry.get("recommended_action") if isinstance(entry.get("recommended_action"), dict) else None
        if action is None:
            continue
        rows.append(
            {
                "schema": "sonic_policy_action_export_v0",
                "source_model_schema": model.get("schema"),
                "source_model": model.get("source"),
                "task_id": str(task_id),
                "demo_kind": entry.get("demo_kind"),
                "object_category": entry.get("object_category"),
                "grasp_affordance": entry.get("grasp_affordance"),
                "best_example_id": entry.get("best_example_id"),
                "best_run_id": entry.get("best_run_id"),
                "best_dense_score": entry.get("best_dense_score"),
                "best_quality": entry.get("best_quality"),
                "candidate_count": entry.get("candidate_count"),
                "positive_count": entry.get("positive_count"),
                "policy_action": action,
            }
        )
    return rows


def _read_model(path: str | Path) -> dict[str, Any]:
    p = _repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy model not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"policy model must be a JSON object: {p}")
    if payload.get("schema") != "sonic_task_policy_memory_v0":
        raise ValueError(f"unsupported policy model schema: {payload.get('schema')}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "demo_kind",
        "object_category",
        "grasp_affordance",
        "best_run_id",
        "best_dense_score",
        "best_quality",
        "candidate_count",
        "positive_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_table(rows: list[dict[str, Any]], *, source: str) -> None:
    print(f"policy_export={source} actions={len(rows)}")
    print(f"{'task_id':34s} {'demo':6s} {'score':>5s} {'pos':>4s} best_run")
    for row in rows:
        print(
            f"{str(row.get('task_id') or '-')[:34]:34s} "
            f"{str(row.get('demo_kind') or '-')[:6]:6s} "
            f"{float(row.get('best_dense_score') or 0.0):>5.2f} "
            f"{int(row.get('positive_count') or 0):>4d} "
            f"{row.get('best_run_id') or '-'}"
        )


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
