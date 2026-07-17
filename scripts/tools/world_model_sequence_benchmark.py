#!/usr/bin/env python3
"""Evaluate ordered multi-stage task-suite sequences and write a leaderboard."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_runner import _evaluate_task
from sonic_world.task_suites import load_robocasa_task_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ordered world-model sequence checks and write a sequence leaderboard.")
    parser.add_argument("--suite", default="configs/world_model/task_suites/sonic_general_v0.yaml")
    parser.add_argument("--sequence", action="append", default=[], help="Sequence id to evaluate; repeat as needed.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum sequences; 0 evaluates all sequences.")
    parser.add_argument("--output-dir", default="reports/benchmarks")
    parser.add_argument("--name", default="sonic_sequence_v0")
    parser.add_argument("--no-scene-validate", action="store_true")
    parser.add_argument("--headless-probe", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless every evaluated sequence succeeds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_robocasa_task_suite(_repo_path(args.suite), repo_root=REPO)
    sequences = _group_sequences(suite.tasks)
    if args.sequence:
        wanted = set(str(item) for item in args.sequence)
        sequences = {key: value for key, value in sequences.items() if key in wanted}
        missing = sorted(wanted - set(sequences))
        if missing:
            raise SystemExit(f"Unknown sequence id(s): {', '.join(missing)}")
    items = list(sequences.items())
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if not items:
        raise SystemExit("No sequence stages were selected.")

    rows = []
    for sequence_id, stages in items:
        evaluated = [
            _evaluate_task(
                task,
                validate_scene=not args.no_scene_validate,
                headless_probe=bool(args.headless_probe),
                probe_steps=int(args.probe_steps),
                probe_fall_height=0.35,
                probe_fall_angle_deg=45.0,
            )
            for task in stages
        ]
        rows.append(_sequence_row(sequence_id, evaluated))

    rows.sort(key=lambda item: (-int(item["sequence_success"]), -float(item["completion_rate"]), str(item["sequence_id"])))
    summary = _summary(rows, suite_name=suite.name, suite_version=suite.version, suite_path=args.suite)
    report = {"schema": "sonic_world_model_sequence_benchmark_v0", "summary": summary, "leaderboard": rows}
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(args.name)
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / f"{stem}.csv", rows)
    (output_dir / f"{stem}.md").write_text(_markdown(summary, rows), encoding="utf-8")
    _print_table(summary, rows)
    print(f"Wrote sequence leaderboard: {_relative(output_dir / f'{stem}.json')}")
    return 0 if not args.strict or summary["sequence_success_count"] == summary["sequence_count"] else 1


def _group_sequences(tasks: tuple[Any, ...]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for task in tasks:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        sequence_id = str(metadata.get("sequence_id") or "").strip()
        if sequence_id:
            grouped.setdefault(sequence_id, []).append(task)
    for stages in grouped.values():
        stages.sort(key=_stage_key)
    return dict(sorted(grouped.items()))


def _stage_key(task: Any) -> tuple[int, str]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    value = metadata.get("sequence_stage") or metadata.get("stage_index")
    try:
        return int(value), str(task.task_id)
    except (TypeError, ValueError):
        match = re.search(r"stage_(\d+)", str(task.task_id))
        return (int(match.group(1)) if match else 9999, str(task.task_id))


def _sequence_row(sequence_id: str, stages: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for stage in stages if stage.get("offline_success"))
    navigation_stages = sum(1 for stage in stages if "navigate.approach_object" in (stage.get("steps") or []))
    return {
        "sequence_id": sequence_id,
        "stage_count": len(stages),
        "successful_stage_count": successes,
        "completion_rate": successes / len(stages) if stages else 0.0,
        "sequence_success": successes == len(stages) and bool(stages),
        "skill_count": sum(int(stage.get("skill_count") or 0) for stage in stages),
        "navigation_stage_count": navigation_stages,
        "recovery_required_stage_count": sum(1 for stage in stages if stage.get("decision_status") == "needs_recovery"),
        "stages": stages,
    }


def _summary(rows: list[dict[str, Any]], *, suite_name: str, suite_version: str, suite_path: str) -> dict[str, Any]:
    total_stages = sum(int(row["stage_count"]) for row in rows)
    successful_stages = sum(int(row["successful_stage_count"]) for row in rows)
    sequence_successes = sum(1 for row in rows if row["sequence_success"])
    return {
        "suite": suite_name,
        "version": suite_version,
        "suite_path": str(suite_path),
        "sequence_count": len(rows),
        "sequence_success_count": sequence_successes,
        "sequence_success_rate": sequence_successes / len(rows) if rows else 0.0,
        "stage_count": total_stages,
        "successful_stage_count": successful_stages,
        "stage_success_rate": successful_stages / total_stages if total_stages else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["rank", "sequence_id", "stage_count", "successful_stage_count", "completion_rate", "sequence_success", "skill_count", "navigation_stage_count", "recovery_required_stage_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({"rank": index, **{field: row.get(field) for field in fields if field != "rank"}})


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Sonic Sequence Leaderboard",
        "",
        f"- sequences: {summary['sequence_success_count']}/{summary['sequence_count']} ({summary['sequence_success_rate']:.1%})",
        f"- stages: {summary['successful_stage_count']}/{summary['stage_count']} ({summary['stage_success_rate']:.1%})",
        "",
        "| rank | sequence | stages | completion | navigation stages | recovery stages |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"| {index} | {row['sequence_id']} | {row['successful_stage_count']}/{row['stage_count']} | {row['completion_rate']:.1%} | {row['navigation_stage_count']} | {row['recovery_required_stage_count']} |")
    return "\n".join(lines) + "\n"


def _print_table(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(
        f"sequence_benchmark={summary['suite']}:{summary['version']} "
        f"sequences={summary['sequence_success_count']}/{summary['sequence_count']} "
        f"stages={summary['successful_stage_count']}/{summary['stage_count']}"
    )
    for index, row in enumerate(rows, start=1):
        print(f"{index:2d} {row['sequence_id'][:38]:38s} stages={row['successful_stage_count']}/{row['stage_count']} completion={row['completion_rate']:.1%} nav={row['navigation_stage_count']} recovery={row['recovery_required_stage_count']}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
