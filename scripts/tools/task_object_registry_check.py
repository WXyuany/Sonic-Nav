#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from sonic_world.task_suites import load_robocasa_task_suite
from sonic_world.world_model import TaskObjectRegistry


DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_OUTPUT = "reports/readiness/task_object_registry_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate task object registry mappings for a Sonic task suite.")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task", action="append", help="Task id to validate. May be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-scene-names", action="store_true", help="Skip XML name existence checks.")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_path = _repo_path(args.suite)
    suite = load_robocasa_task_suite(suite_path, repo_root=REPO)
    tasks = list(suite.tasks)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.task_id in wanted]
        missing = sorted(wanted - {task.task_id for task in tasks})
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    if not tasks:
        raise SystemExit("No tasks selected for registry validation.")

    rows = [_validate_task(task, check_scene_names=not args.no_scene_names) for task in tasks]
    summary = {
        "suite": suite.name,
        "suite_version": suite.version,
        "suite_path": _rel(suite_path),
        "task_count": len(rows),
        "passed": sum(1 for row in rows if row["validation"]["ok"]),
        "failed": sum(1 for row in rows if not row["validation"]["ok"]),
        "warning_count": sum(len(row["validation"].get("warnings") or []) for row in rows),
    }
    report = {
        "schema": "sonic_task_object_registry_report_v0",
        "summary": summary,
        "tasks": rows,
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(rows, summary)
    print(f"\nWrote registry report: {_rel(output)}")
    return 0 if summary["failed"] == 0 else 1


def _validate_task(task: Any, *, check_scene_names: bool) -> dict[str, Any]:
    registry = TaskObjectRegistry.from_task_case(task)
    validation = registry.validate(task=task, repo_root=REPO, check_scene_names=check_scene_names)
    return {
        "task_id": task.task_id,
        "scene": str(task.scene.scene_xml),
        "object_id": str(task.request.object_id or ""),
        "target_id": str(task.request.target_id or ""),
        "record_count": len(registry.records),
        "registry": registry.to_dict(),
        "validation": validation.to_dict(),
    }


def _print_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(
        "task_object_registry="
        f"{summary['suite']}:{summary['suite_version']} tasks={summary['task_count']} "
        f"passed={summary['passed']} failed={summary['failed']} warnings={summary['warning_count']}"
    )
    print(f"{'task_id':42s} {'records':>7s} {'errors':>6s} {'warn':>5s} object -> target")
    for row in rows:
        validation = row["validation"]
        errors = validation.get("errors") or []
        warnings = validation.get("warnings") or []
        print(
            f"{str(row['task_id'])[:42]:42s} "
            f"{int(row['record_count']):>7d} "
            f"{len(errors):>6d} "
            f"{len(warnings):>5d} "
            f"{row.get('object_id') or '-'} -> {row.get('target_id') or '-'}"
        )
        for error in errors[:3]:
            print(f"  ERROR {error}")
        for warning in warnings[:2]:
            print(f"  WARN  {warning}")


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
