#!/usr/bin/env python3
"""Rank completed carry-state episode logs separately from offline benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a physical episode leaderboard from executor-backed JSONL logs.")
    parser.add_argument("--input", action="append", default=[], help="Episode JSONL file or directory; repeatable.")
    parser.add_argument("--output-dir", default="reports/leaderboards")
    parser.add_argument("--name", default="sonic_physical_episode_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _paths(args.input or ["reports/rollouts", "reports/episodes"])
    rows = [row for path in paths if (row := _row(path)) is not None]
    rows.sort(key=lambda item: (-int(item["physical_sequence_success"]), -float(item["physical_stage_success_rate"]), int(item["recovery_event_count"]), str(item["sequence_id"])))
    summary = {
        "episode_count": len(rows),
        "physical_sequence_success_count": sum(int(item["physical_sequence_success"]) for item in rows),
        "physical_stage_count": sum(int(item["stage_count"]) for item in rows),
        "physical_stage_success_count": sum(int(item["successful_stage_count"]) for item in rows),
    }
    summary["physical_sequence_success_rate"] = summary["physical_sequence_success_count"] / summary["episode_count"] if rows else 0.0
    summary["physical_stage_success_rate"] = summary["physical_stage_success_count"] / summary["physical_stage_count"] if summary["physical_stage_count"] else 0.0
    report = {"schema": "sonic_world_model_physical_leaderboard_v0", "summary": summary, "leaderboard": rows}
    output = _repo_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(args.name)
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _csv(output / f"{stem}.csv", rows)
    (output / f"{stem}.md").write_text(_markdown(summary, rows), encoding="utf-8")
    print(f"physical_leaderboard=episodes={summary['episode_count']} sequences={summary['physical_sequence_success_count']}/{summary['episode_count']} stages={summary['physical_stage_success_count']}/{summary['physical_stage_count']}")
    print(_relative(output / f"{stem}.json"))
    return 0


def _paths(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = _repo_path(raw)
        if path.is_file() and path.suffix == ".jsonl":
            found.append(path)
        elif path.is_dir():
            # Episode producers use meaningful prefixes (for example
            # ``curriculum_`` and ``sequence_``), so discovery must be based on
            # the event contract in _row rather than a historical filename.
            found.extend(sorted(path.rglob("*.jsonl")))
    return list(dict.fromkeys(found))


def _row(path: Path) -> dict[str, Any] | None:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("schema") == "sonic_world_model_episode_event_v0":
            events.append(item)
    terminal = next((item for item in reversed(events) if item.get("event") == "episode_terminal"), None)
    if terminal is None:
        return None
    # Simulator contact-lock trajectories are useful teacher data, but they
    # must never contribute to a physical capability claim or leaderboard.
    if any(
        bool((item.get("metrics") or {}).get("teacher_assisted"))
        for item in events
        if item.get("event") == "primitive_status" and isinstance(item.get("metrics"), dict)
    ):
        return None
    stages = [item for item in events if item.get("event") == "stage_terminal"]
    success = [item for item in stages if item.get("status") == "succeeded"]
    starts = [float(item.get("stamp") or 0.0) for item in events if item.get("event") == "stage_start"]
    end = float(terminal.get("stamp") or 0.0)
    return {
        "sequence_id": str(next((item.get("sequence_id") for item in events if item.get("sequence_id")), path.stem)),
        "source_log": _relative(path),
        "final_status": str(terminal.get("status") or "unknown"),
        # A stage-isolated curriculum success is valuable skill evidence, but
        # cannot be advertised as a carry-state sequence success. Legacy logs
        # without an explicit scope are treated conservatively as stage-only.
        "episode_scope": str(terminal.get("episode_scope") or "legacy_unknown"),
        "physical_sequence_success": str(terminal.get("episode_scope") or "") == "full_sequence" and str(terminal.get("status") or "") == "succeeded",
        "stage_count": len(stages),
        "successful_stage_count": len(success),
        "physical_stage_success_rate": len(success) / len(stages) if stages else 0.0,
        "recovery_event_count": sum(1 for item in events if item.get("event") == "recovery_status"),
        "duration_s": round(max(0.0, end - min(starts)), 3) if starts else None,
        "failed_stages": terminal.get("failed_stages") or [],
    }


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["rank", "sequence_id", "episode_scope", "final_status", "physical_sequence_success", "stage_count", "successful_stage_count", "physical_stage_success_rate", "recovery_event_count", "duration_s", "source_log"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({"rank": index, **{field: row.get(field) for field in fields if field != "rank"}})


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# Sonic Physical Episode Leaderboard", "", f"- sequences: {summary['physical_sequence_success_count']}/{summary['episode_count']} ({summary['physical_sequence_success_rate']:.1%})", f"- stages: {summary['physical_stage_success_count']}/{summary['physical_stage_count']} ({summary['physical_stage_success_rate']:.1%})", "", "| rank | sequence | result | stages | recovery | duration |", "|---:|---|---|---:|---:|---:|"]
    for index, row in enumerate(rows, 1):
        lines.append(f"| {index} | {row['sequence_id']} | {row['final_status']} | {row['successful_stage_count']}/{row['stage_count']} | {row['recovery_event_count']} | {row['duration_s'] or 0:.1f}s |")
    return "\n".join(lines) + "\n"


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
