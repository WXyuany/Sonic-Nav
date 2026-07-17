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

from rollout_report import STAGES, _read_events, summarize
from sonic_world.task_suites import evaluate_task_success, load_robocasa_task_suite


DEFAULT_ROLLOUTS = "reports/rollouts"
DEFAULT_OUTCOMES = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"
DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_OUTPUT = "reports/datasets/sonic_rollout_episodes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact training/evaluation episode dataset from Sonic rollout logs. "
            "The dataset is for task/skill policies; it references sensor streams and physics state but does not store raw SONIC controls."
        )
    )
    parser.add_argument("--rollouts", nargs="*", default=[DEFAULT_ROLLOUTS])
    parser.add_argument("--policy-outcomes", default=DEFAULT_OUTCOMES)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--run-id-prefix", action="append")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", help="CSV output path. Defaults to <output>.csv.")
    parser.add_argument("--include-events", action="store_true", default=True)
    parser.add_argument("--no-events", dest="include_events", action="store_false")
    parser.add_argument("--max-events", type=int, default=192)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = list(_read_events(args.rollouts))
    summaries = summarize(events)
    if args.run_id_prefix:
        prefixes = tuple(str(item) for item in args.run_id_prefix)
        summaries = [row for row in summaries if str(row.get("run_id") or "").startswith(prefixes)]
    grouped = _events_by_run(events)
    outcomes = _outcomes_by_run(args.policy_outcomes)
    tasks = _tasks_by_id(args.suite)

    episodes = [
        _episode(
            summary,
            events=grouped.get(str(summary.get("run_id") or ""), []),
            outcome=outcomes.get(str(summary.get("run_id") or "")),
            task=tasks.get(str(summary.get("task_id") or "")),
            include_events=bool(args.include_events),
            max_events=max(0, int(args.max_events)),
        )
        for summary in summaries
    ]

    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    _write_csv(summary_path, episodes)
    if args.print_json:
        print(json.dumps({"episodes": episodes}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(episodes)
    print(f"\nWrote rollout episode dataset: {_rel(output)}")
    print(f"Wrote rollout episode summary: {_rel(summary_path)}")
    return 0


def _episode(
    summary: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    outcome: dict[str, Any] | None,
    task: Any | None,
    include_events: bool,
    max_events: int,
) -> dict[str, Any]:
    run_id = str(summary.get("run_id") or "unknown")
    compact_events = _compact_events(events, limit=max_events) if include_events else []
    stages = _stage_table(summary, compact_events)
    outcome_payload = outcome.get("outcome") if isinstance(outcome, dict) else None
    teacher_action = outcome.get("teacher_action") if isinstance(outcome, dict) else None
    observation = outcome.get("observation") if isinstance(outcome, dict) else None
    task_spec = task.to_dict() if task is not None and hasattr(task, "to_dict") else None
    sensor_contract = _sensor_contract(task_spec, observation)
    oracle = evaluate_task_success(task, rollout_summary=summary).to_dict() if task is not None else None
    return {
        "schema": "sonic_rollout_episode_v0",
        "run_id": run_id,
        "demo_kind": summary.get("demo_kind"),
        "task_id": summary.get("task_id"),
        "scene": summary.get("scene"),
        "final_status": summary.get("final_status"),
        "success": summary.get("final_status") == "success",
        "oracle_success": oracle.get("success") if isinstance(oracle, dict) else None,
        "quality": (outcome_payload or {}).get("quality"),
        "dense_score": (outcome_payload or {}).get("dense_score"),
        "retry_count": int(summary.get("retry_count") or 0),
        "lift_success": bool(summary.get("lift_success")),
        "lift_failed": bool(summary.get("lift_failed")),
        "terminal_stage": (outcome_payload or {}).get("terminal_stage") or summary.get("fail_stage") or "done",
        "primary_issue": (outcome_payload or {}).get("primary_issue") or summary.get("fail_reason") or summary.get("retry_reason") or "",
        "stage_summary": stages,
        "task_oracle": oracle,
        "task_spec": task_spec,
        "policy": {
            "observation": observation,
            "teacher_action": teacher_action,
            "outcome": outcome_payload,
            "match": outcome.get("match") if isinstance(outcome, dict) else None,
        },
        "sensor_contract": sensor_contract,
        "artifacts": _artifact_manifest(run_id, summary, events=events),
        "timeline": compact_events,
        "metadata": {
            "controller_boundary": "frozen_sonic_low_level",
            "training_scope": "task_and_skill_policy_only",
            "source_log": _source_file(events),
            "source_policy_outcome": outcome.get("link_id") if isinstance(outcome, dict) else None,
        },
    }


def _stage_table(summary: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    first_time = events[0].get("t") if events else None
    for stage in STAGES:
        event_count = int(summary.get(f"{stage}_events") or 0)
        failure_count = int(summary.get(f"{stage}_failures") or 0)
        retry_count = int(summary.get(f"{stage}_retries") or 0)
        if event_count == 0 and failure_count == 0 and retry_count == 0:
            continue
        stage_events = [event for event in events if event.get("primitive_stage") == stage]
        start_t = stage_events[0].get("t") if stage_events else None
        end_t = stage_events[-1].get("t") if stage_events else None
        out.append(
            {
                "stage": stage,
                "event_count": event_count,
                "failure_count": failure_count,
                "retry_count": retry_count,
                "start_s": None if start_t is None or first_time is None else round(float(start_t) - float(first_time), 4),
                "end_s": None if end_t is None or first_time is None else round(float(end_t) - float(first_time), 4),
            }
        )
    return out


def _compact_events(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for event in events[:limit]:
        compact.append(
            {
                "t": event.get("stamp"),
                "event": event.get("event"),
                "phase": event.get("phase"),
                "primitive_stage": event.get("primitive_stage"),
                "skill_name": event.get("skill_name"),
                "status": event.get("status"),
                "reason": event.get("reason"),
                "metrics": _small_mapping(event.get("metrics")),
                "metadata": _small_mapping(event.get("metadata"), keep=("summary", "error", "policy_notes", "task_id")),
            }
        )
    if len(events) > limit:
        compact.append({"event": "truncated", "dropped_event_count": len(events) - limit})
    return compact


def _small_mapping(value: Any, *, keep: tuple[str, ...] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    items = value.items()
    if keep is not None:
        items = [(key, value[key]) for key in keep if key in value]
    out: dict[str, Any] = {}
    for key, item in items:
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
        elif isinstance(item, (list, tuple)) and len(item) <= 8:
            out[str(key)] = item
        elif isinstance(item, dict) and len(item) <= 8:
            out[str(key)] = item
    return out


def _sensor_contract(task_spec: dict[str, Any] | None, observation: dict[str, Any] | None) -> list[str]:
    if isinstance(observation, dict) and isinstance(observation.get("sensor_contract"), list):
        return [str(item) for item in observation["sensor_contract"]]
    metadata = task_spec.get("metadata") if isinstance(task_spec, dict) else {}
    if isinstance(metadata, dict) and isinstance(metadata.get("sensor_use"), list):
        return [str(item) for item in metadata["sensor_use"]]
    return ["odom", "tf", "lidar", "rgb", "depth", "object_anchor", "world_model", "privileged_physics_state"]


def _artifact_manifest(run_id: str, summary: dict[str, Any], *, events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rollout_jsonl": _source_file(events) or summary.get("_source_file") or "",
        "sensor_snapshot_policy": "online_stream_not_embedded",
        "expected_online_streams": {
            "qpos": "/tmp/sonic_qpos.npy",
            "lidar": "/tmp/sonic_lidar.npy",
            "mid360": "/tmp/sonic_mid360.npy",
            "rgbd": "/tmp/sonic_camera_*",
        },
        "run_id": run_id,
    }


def _events_by_run(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("run_id") or "unknown"), []).append(event)
    for items in grouped.values():
        items.sort(key=lambda event: float(event.get("monotonic") or event.get("stamp") or 0.0))
    return grouped


def _outcomes_by_run(path: str | Path) -> dict[str, dict[str, Any]]:
    p = _repo_path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            rollout = payload.get("rollout") if isinstance(payload.get("rollout"), dict) else {}
            run_id = str(rollout.get("run_id") or "")
            if run_id:
                out[run_id] = payload
    return out


def _tasks_by_id(path: str | Path) -> dict[str, Any]:
    try:
        suite = load_robocasa_task_suite(_repo_path(path), repo_root=REPO)
    except Exception:
        return {}
    return {task.task_id: task for task in suite.tasks}


def _source_file(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("_source_file"):
            return str(event["_source_file"])
    return ""


def _write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "demo_kind",
        "task_id",
        "scene",
        "final_status",
        "success",
        "oracle_success",
        "quality",
        "dense_score",
        "retry_count",
        "lift_success",
        "lift_failed",
        "terminal_stage",
        "primary_issue",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({field: episode.get(field) for field in fields})


def _print_table(episodes: list[dict[str, Any]]) -> None:
    print(f"episodes={len(episodes)}")
    print(f"{'run_id':28s} {'demo':8s} {'task':26s} {'status':10s} {'score':>5s} {'retry':>5s} issue")
    for episode in episodes:
        score = episode.get("dense_score")
        score_text = "-" if score is None else f"{float(score):.2f}"
        print(
            f"{str(episode.get('run_id') or '-')[:28]:28s} "
            f"{str(episode.get('demo_kind') or '-')[:8]:8s} "
            f"{str(episode.get('task_id') or '-')[:26]:26s} "
            f"{str(episode.get('final_status') or '-')[:10]:10s} "
            f"{score_text:>5s} "
            f"{int(episode.get('retry_count') or 0):>5d} "
            f"{episode.get('primary_issue') or '-'}"
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
