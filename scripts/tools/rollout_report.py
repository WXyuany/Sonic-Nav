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
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


DEFAULT_INPUT = "reports/rollouts"
DEFAULT_OUTPUT = "reports/rollouts/summary.csv"
STAGES = ("approach", "workspace", "grasp", "lift", "transport", "place", "fall", "done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Sonic rollout JSONL logs by run and primitive stage.")
    parser.add_argument("paths", nargs="*", default=[DEFAULT_INPUT], help="JSONL files or directories.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = list(_read_events(args.paths))
    rows = summarize(events)
    if args.print_json:
        print(json.dumps({"rollouts": rows}, indent=2, sort_keys=True))
    else:
        _print_table(rows)
    out = _repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out, rows)
    message = f"Wrote rollout summary: {_rel(out)}"
    if args.print_json:
        print(message, file=sys.stderr)
    else:
        print(f"\n{message}")
    return 0


def summarize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        run_id = str(event.get("run_id") or "unknown")
        grouped.setdefault(run_id, []).append(event)

    rows: list[dict[str, Any]] = []
    for run_id, items in sorted(grouped.items(), key=lambda pair: _first_stamp(pair[1])):
        items.sort(key=lambda event: float(event.get("monotonic") or event.get("stamp") or 0.0))
        first = items[0]
        task_end = _last_event(items, "task_end")
        raw_failed = _first_failed(items)
        retry_events = [event for event in items if event.get("event") == "retry"]
        first_retry = retry_events[0] if retry_events else None
        lift_checks = [event for event in items if event.get("event") == "lift_check"]
        stage_counts = {stage: 0 for stage in STAGES}
        stage_failures = {stage: 0 for stage in STAGES}
        stage_retries = {stage: 0 for stage in STAGES}
        for event in items:
            stage = event.get("primitive_stage")
            if stage in stage_counts and event.get("event") in {"phase_end", "retry", "lift_check"}:
                stage_counts[stage] += 1
                if event.get("status") == "failed":
                    stage_failures[stage] += 1
                if event.get("status") == "retry":
                    stage_retries[stage] += 1
        lift_success = any(event.get("status") == "success" for event in lift_checks)
        lift_failed = any(event.get("status") == "failed" for event in lift_checks)
        final_status = str((task_end or {}).get("status") or ("failed" if raw_failed else "unknown"))
        if final_status == "success" and lift_failed and not lift_success:
            final_status = "failed"
            raw_failed = _last_failed(lift_checks) or raw_failed
        failed = raw_failed if final_status != "success" else None
        fail_stage = str((failed or {}).get("primitive_stage") or "")
        fail_reason = str((failed or {}).get("reason") or "")
        retry_stage = str((first_retry or {}).get("primitive_stage") or "")
        retry_reason = str((first_retry or {}).get("reason") or "")
        rows.append(
            {
                "run_id": run_id,
                "demo_kind": first.get("demo_kind"),
                "task_id": first.get("task_id"),
                "scene": first.get("scene"),
                "event_count": len(items),
                "phase_count": sum(1 for event in items if event.get("event") == "phase_end"),
                "retry_count": len(retry_events),
                "lift_success": lift_success,
                "lift_failed": lift_failed,
                "final_status": final_status,
                "fail_stage": fail_stage,
                "fail_reason": fail_reason,
                "retry_stage": retry_stage,
                "retry_reason": retry_reason,
                "retry_summary": _retry_summary(stage_retries),
                **{f"{stage}_events": stage_counts[stage] for stage in STAGES},
                **{f"{stage}_failures": stage_failures[stage] for stage in STAGES},
                **{f"{stage}_retries": stage_retries[stage] for stage in STAGES},
            }
        )
    return rows


def _read_events(paths: Iterable[str]) -> Iterable[dict[str, Any]]:
    for raw in paths:
        path = _repo_path(raw)
        if path.is_dir():
            yield from _read_events(str(child) for child in sorted(path.glob("*.jsonl")))
            continue
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSONL in {path}:{line_no}: {exc}") from exc
                if isinstance(payload, dict):
                    payload.setdefault("_source_file", _rel(path))
                    yield payload


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"rollouts={len(rows)}")
    print(f"{'run_id':28s} {'demo':8s} {'task':24s} {'status':10s} {'retry':>5s} {'lift':>5s} {'retry_by_stage':20s} issue")
    for row in rows:
        lift = "yes" if row["lift_success"] else ("fail" if row["lift_failed"] else "-")
        if row.get("fail_reason") or row.get("fail_stage"):
            issue = row["fail_reason"] or row["fail_stage"]
        elif row.get("retry_reason") or row.get("retry_stage"):
            issue = f"retry:{row['retry_reason'] or row['retry_stage']}"
        else:
            issue = "-"
        print(
            f"{str(row['run_id'])[:28]:28s} "
            f"{str(row.get('demo_kind') or '-')[:8]:8s} "
            f"{str(row.get('task_id') or '-')[:24]:24s} "
            f"{str(row.get('final_status') or '-')[:10]:10s} "
            f"{int(row['retry_count']):>5d} "
            f"{lift:>5s} "
            f"{str(row.get('retry_summary') or '-')[:20]:20s} "
            f"{issue}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base_fields = [
        "run_id",
        "demo_kind",
        "task_id",
        "scene",
        "event_count",
        "phase_count",
        "retry_count",
        "lift_success",
        "lift_failed",
        "final_status",
        "fail_stage",
        "fail_reason",
        "retry_stage",
        "retry_reason",
        "retry_summary",
    ]
    fields = (
        base_fields
        + [f"{stage}_events" for stage in STAGES]
        + [f"{stage}_failures" for stage in STAGES]
        + [f"{stage}_retries" for stage in STAGES]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _last_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == name:
            return event
    return None


def _first_failed(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("status") == "failed":
            return event
    return None


def _last_failed(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("status") == "failed":
            return event
    return None


def _first_stamp(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    return float(events[0].get("stamp") or 0.0)


def _retry_summary(stage_retries: dict[str, int]) -> str:
    parts = [f"{stage}:{count}" for stage, count in stage_retries.items() if count]
    return ",".join(parts)


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
