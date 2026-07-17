#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rollout_report import _read_events, summarize


DEFAULT_MANIFESTS = "reports/task_batches"
DEFAULT_ROLLOUTS = "reports/rollouts"
DEFAULT_OUTPUT = "reports/task_batches/summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize suite-level Sonic task batches by task id.")
    parser.add_argument("manifests", nargs="*", default=[DEFAULT_MANIFESTS], help="manifest.json files or directories.")
    parser.add_argument("--rollouts", nargs="*", default=[DEFAULT_ROLLOUTS])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", help="JSON report path. Defaults to <output>.json.")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = list(_read_manifests(args.manifests))
    if not manifests:
        raise SystemExit("No task batch manifests found.")
    rollouts = summarize(list(_read_events(args.rollouts)))
    rows = _summarize_batches(manifests, rollouts)
    report = {
        "schema": "sonic_task_batch_report_v0",
        "batch_count": len(manifests),
        "task_count": len(rows),
        "success_count": sum(int(row["success_count"]) for row in rows),
        "observed_run_count": sum(int(row["observed_runs"]) for row in rows),
        "tasks": rows,
    }

    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    json_path = _repo_path(args.json_output) if args.json_output else output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(rows, batch_count=len(manifests))
    print(f"\nWrote task batch summary: {_rel(output)}")
    print(f"Wrote task batch JSON: {_rel(json_path)}")
    return 0


def _summarize_batches(manifests: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run = {str(row.get("run_id") or ""): row for row in rollouts}
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        batch_name = Path(str(manifest.get("_path") or "batch")).parent.name
        for task in manifest.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            expected = [str(item) for item in task.get("expected_run_ids") or []]
            observed = [by_run[run_id] for run_id in expected if run_id in by_run]
            success = [row for row in observed if row.get("final_status") == "success"]
            lift_success = [row for row in observed if row.get("lift_success")]
            issues = _issue_counts(observed)
            rows.append(
                {
                    "batch": batch_name,
                    "task_id": task.get("task_id"),
                    "demo_kind": task.get("demo_kind"),
                    "scene": task.get("scene"),
                    "expected_runs": len(expected),
                    "observed_runs": len(observed),
                    "success_count": len(success),
                    "success_rate": round(len(success) / len(observed), 4) if observed else 0.0,
                    "lift_success_count": len(lift_success),
                    "avg_retry": round(sum(int(row.get("retry_count") or 0) for row in observed) / len(observed), 3) if observed else 0.0,
                    "top_issue": issues[0][0] if issues else "",
                    "top_issue_count": issues[0][1] if issues else 0,
                    "run_id_prefix": task.get("run_id_prefix"),
                    "status": task.get("status"),
                    "exit_code": task.get("exit_code"),
                }
            )
    rows.sort(key=lambda row: (str(row.get("batch") or ""), str(row.get("task_id") or "")))
    return rows


def _issue_counts(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        issue = str(row.get("fail_reason") or row.get("retry_reason") or row.get("fail_stage") or row.get("retry_stage") or "")
        if not issue:
            continue
        counts[issue] = counts.get(issue, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _read_manifests(paths: Iterable[str]) -> Iterable[dict[str, Any]]:
    for raw in paths:
        path = _repo_path(raw)
        if path.is_dir():
            yield from _read_manifests(str(child) for child in sorted(path.glob("**/manifest.json")))
            continue
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["_path"] = _rel(path)
            yield payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "batch",
        "task_id",
        "demo_kind",
        "scene",
        "expected_runs",
        "observed_runs",
        "success_count",
        "success_rate",
        "lift_success_count",
        "avg_retry",
        "top_issue",
        "top_issue_count",
        "run_id_prefix",
        "status",
        "exit_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_table(rows: list[dict[str, Any]], *, batch_count: int) -> None:
    print(f"task_batches={batch_count} tasks={len(rows)}")
    print(f"{'batch':22s} {'task':30s} {'demo':6s} {'obs':>3s} {'succ':>4s} {'rate':>5s} {'retry':>5s} issue")
    for row in rows:
        print(
            f"{str(row.get('batch') or '-')[:22]:22s} "
            f"{str(row.get('task_id') or '-')[:30]:30s} "
            f"{str(row.get('demo_kind') or '-')[:6]:6s} "
            f"{int(row.get('observed_runs') or 0):>3d} "
            f"{int(row.get('success_count') or 0):>4d} "
            f"{float(row.get('success_rate') or 0.0):>5.2f} "
            f"{float(row.get('avg_retry') or 0.0):>5.1f} "
            f"{row.get('top_issue') or '-'}"
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
