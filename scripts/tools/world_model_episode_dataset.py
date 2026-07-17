#!/usr/bin/env python3
"""Convert executor-backed physical episode logs into residual-policy transitions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build real-physics residual policy transitions from world-model episode JSONL logs."
    )
    parser.add_argument("--input", action="append", required=True, help="Episode JSONL file or directory; repeatable.")
    parser.add_argument("--output", default="reports/policy_data/physical_episode_residual_v0.jsonl")
    parser.add_argument("--summary", help="CSV summary path; defaults to <output>.csv.")
    parser.add_argument("--include-without-features", action="store_true", help="Retain historical rows that predate PPO feature logging.")
    parser.add_argument("--visual-context", action="store_true", help="Use recorded Qwen/RGB-D visual context rather than the behavior-policy context.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in _episode_paths(args.input):
        rows.extend(
            _transitions(
                path,
                include_without_features=bool(args.include_without_features),
                visual_context=bool(args.visual_context),
            )
        )
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = _summary(rows)
    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    _write_csv(summary_path, rows)
    print(json.dumps(summary, sort_keys=True))
    print(_relative(output))
    return 0


def _episode_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = _repo_path(raw)
        if path.is_file() and path.suffix == ".jsonl":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
    return list(dict.fromkeys(paths))


def _transitions(path: Path, *, include_without_features: bool, visual_context: bool) -> list[dict[str, Any]]:
    events = _read_events(path)
    policy_by_stage: dict[tuple[int, str], list[dict[str, Any]]] = {}
    recovery_by_stage: dict[tuple[int, str], list[dict[str, Any]]] = {}
    terminals: list[dict[str, Any]] = []
    for event in events:
        stage_key = (int(event.get("stage_index") or 0), str(event.get("task_id") or ""))
        if event.get("event") == "policy_action":
            policy_by_stage.setdefault(stage_key, []).append(event)
        elif event.get("event") == "recovery_status":
            recovery_by_stage.setdefault(stage_key, []).append(event)
        elif event.get("event") == "primitive_status" and _is_terminal_primitive(event):
            terminals.append(event)

    rows: list[dict[str, Any]] = []
    for index, terminal in enumerate(terminals):
        stage_index = int(terminal.get("stage_index") or 0)
        task_id = str(terminal.get("task_id") or "")
        key = (stage_index, task_id)
        policy_event = _last_before(policy_by_stage.get(key, []), float(terminal.get("stamp") or 0.0))
        if policy_event is None:
            continue
        metadata = policy_event.get("policy_metadata") if isinstance(policy_event.get("policy_metadata"), dict) else {}
        observation_key = "visual_observation" if visual_context else "observation"
        observation = metadata.get(observation_key) if isinstance(metadata.get(observation_key), dict) else None
        if observation is None and not include_without_features:
            continue
        evidence = terminal.get("effect_evidence") if isinstance(terminal.get("effect_evidence"), dict) else {}
        passed = bool(evidence.get("passed"))
        effect_source = str(evidence.get("source") or "")
        # Recovery is causally attached only to a failed effect. A later recovery
        # in the same stage often belongs to a different primitive.
        recovery = _first_after(recovery_by_stage.get(key, []), float(terminal.get("stamp") or 0.0)) if not passed else None
        reward = 1.0 if passed else -1.0
        if passed and terminal.get("skill_name") in {"manip.side_grasp", "manip.lift_object"}:
            reward += 0.25
        reason = str(evidence.get("reason") or terminal.get("detail") or "")
        metrics = terminal.get("metrics") if isinstance(terminal.get("metrics"), dict) else {}
        low_hold = metrics.get("low_hold_snapshot") if isinstance(metrics.get("low_hold_snapshot"), dict) else {}
        low_hold_rejected = "low_hold_contact_lost" in reason
        teacher_assisted = bool(metrics.get("teacher_assisted"))
        rows.append(
            {
                "schema": "sonic_world_model_residual_transition_v0",
                "transition_id": f"{path.stem}:{index:04d}",
                "source_log": _relative(path),
                "stage_index": stage_index,
                "task_id": task_id,
                "policy_stamp": float(policy_event.get("stamp") or 0.0),
                "terminal_stamp": float(terminal.get("stamp") or 0.0),
                "policy": {
                    "policy_id": policy_event.get("policy_id"),
                    "type": policy_event.get("policy_type"),
                    "metadata": metadata,
                    "base_goal": policy_event.get("base_goal"),
                    "grasp_offsets": policy_event.get("grasp_offsets"),
                    "object_id": _policy_object_id(policy_event),
                },
                "observation": observation,
                "primitive": {
                    "skill_name": terminal.get("skill_name"),
                    "status": terminal.get("status"),
                    "metrics": terminal.get("metrics") or {},
                    "effect_evidence": evidence,
                },
                "outcome": {
                    "effect_passed": passed,
                    "effect_source": effect_source or None,
                    "reward": round(reward, 3),
                    "failure_reason": reason,
                    "low_hold_guard_rejected": low_hold_rejected,
                    "low_hold_contact_count": int(low_hold.get("target_contact_count") or 0) if low_hold else None,
                    "teacher_assisted": teacher_assisted,
                    "recovery_handler": _recovery_handler(recovery),
                    "recovery_requested": recovery is not None,
                },
            }
        )
    return rows


def _is_terminal_primitive(event: dict[str, Any]) -> bool:
    status = str(event.get("status") or "").lower()
    return status in {"success", "failed", "error", "timeout", "cancelled", "skipped"}


def _last_before(events: list[dict[str, Any]], stamp: float) -> dict[str, Any] | None:
    candidates = [event for event in events if float(event.get("stamp") or 0.0) <= stamp]
    return candidates[-1] if candidates else None


def _first_after(events: list[dict[str, Any]], stamp: float) -> dict[str, Any] | None:
    for event in events:
        if float(event.get("stamp") or 0.0) >= stamp:
            return event
    return None


def _recovery_handler(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return str(payload.get("handler") or "") or None


def _policy_object_id(event: dict[str, Any]) -> str | None:
    offsets = event.get("grasp_offsets") if isinstance(event.get("grasp_offsets"), dict) else {}
    value = offsets.get("object_id")
    return str(value) if value is not None and str(value) else None


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("schema") == "sonic_world_model_episode_event_v0":
            rows.append(event)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool((row.get("outcome") or {}).get("effect_passed")) for row in rows)
    skills: dict[str, dict[str, int]] = {}
    for row in rows:
        skill = str((row.get("primitive") or {}).get("skill_name") or "unknown")
        item = skills.setdefault(skill, {"count": 0, "passed": 0})
        item["count"] += 1
        item["passed"] += int(bool((row.get("outcome") or {}).get("effect_passed")))
    return {
        "schema": "sonic_world_model_residual_dataset_summary_v0",
        "transition_count": len(rows),
        "effect_success_count": passed,
        "effect_success_rate": round(passed / len(rows), 4) if rows else 0.0,
        "skills": dict(sorted(skills.items())),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["transition_id", "task_id", "skill_name", "effect_passed", "reward", "recovery_handler", "policy_id", "source_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            outcome = row.get("outcome") or {}
            primitive = row.get("primitive") or {}
            policy = row.get("policy") or {}
            writer.writerow({
                "transition_id": row.get("transition_id"), "task_id": row.get("task_id"),
                "skill_name": primitive.get("skill_name"), "effect_passed": outcome.get("effect_passed"),
                "reward": outcome.get("reward"), "recovery_handler": outcome.get("recovery_handler"),
                "policy_id": policy.get("policy_id"), "source_log": row.get("source_log"),
            })


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
