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

from sonic_world.policies import HeuristicSkillPolicy, PolicySample
from sonic_world.scenarios import ScenarioSpec, replay_scenario
from sonic_world.task_suites import load_robocasa_task_suite


DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_OUTPUT = "reports/policy_data/sonic_general_v0_heuristic.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build task/skill-level policy samples from a Sonic task suite. "
            "These samples train the high-level policy only; SONIC low-level control remains frozen."
        )
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task", action="append", help="Task id to include. May be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", help="CSV summary path. Defaults to <output>.csv.")
    parser.add_argument("--include-planning", action="store_true", help="Store full planning payload in each JSONL row.")
    parser.add_argument("--print-sample", action="store_true", help="Print the first generated sample JSON.")
    parser.add_argument("--allow-failures", action="store_true", help="Write recoverable/error samples instead of exiting.")
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
        tasks = tasks[: max(0, args.limit)]
    if not tasks:
        raise SystemExit("No tasks selected.")

    policy = HeuristicSkillPolicy()
    samples: list[PolicySample] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for task in tasks:
        try:
            sample = _sample_for_task(task, suite=suite, policy=policy, include_planning=args.include_planning)
        except Exception as exc:
            failures.append({"task_id": task.task_id, "error": str(exc)})
            if not args.allow_failures:
                raise
            continue
        samples.append(sample)
        rows.append(_summary_row(sample))

    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_path, rows)
    _print_table(rows, failures, suite_name=suite.name, suite_version=suite.version)
    if args.print_sample and samples:
        print(json.dumps(samples[0].to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nWrote policy JSONL: {_rel(output)}")
    print(f"Wrote policy summary: {_rel(summary_path)}")
    return 1 if failures and not args.allow_failures else 0


def _sample_for_task(
    task: Any,
    *,
    suite: Any,
    policy: HeuristicSkillPolicy,
    include_planning: bool,
) -> PolicySample:
    replay = replay_scenario(ScenarioSpec.from_dict(task.scenario()))
    result = replay.tasks[0].result
    return policy.sample(
        result,
        suite_name=suite.name,
        suite_version=suite.version,
        suite_metadata=suite.metadata,
        task_metadata=task.metadata,
        task_tags=task.tags,
        include_planning=include_planning,
    )


def _summary_row(sample: PolicySample) -> dict[str, Any]:
    action = sample.action
    hand = action.hand_pose_target or {}
    close = action.grasp_close_ratio or {}
    lift_place = action.lift_place_targets or {}
    base = action.base_goal or {}
    return {
        "sample_id": sample.sample_id,
        "task_id": action.task_id,
        "status": action.status,
        "verb": action.task_intent.get("verb"),
        "demo_kind": action.task_intent.get("demo_kind"),
        "object_id": action.task_intent.get("object_id"),
        "target_id": action.task_intent.get("target_id"),
        "skills": " ".join(action.skill_selection),
        "skill_count": len(action.skill_selection),
        "base_goal": _short_vec(base.get("position")),
        "hand_mode": hand.get("mode"),
        "hand": hand.get("hand"),
        "close_ratio": close.get("close_ratio"),
        "lift_pose_base": _short_vec(lift_place.get("lift_pose_base")),
        "place_target": lift_place.get("target_id"),
        "recovery_status": (action.recovery_decision or {}).get("status"),
        "policy_id": action.policy_id,
    }


def _print_table(rows: list[dict[str, Any]], failures: list[dict[str, str]], *, suite_name: str, suite_version: str) -> None:
    print(f"policy_dataset={suite_name}:{suite_version} samples={len(rows)} failures={len(failures)}")
    print(f"{'task_id':32s} {'status':14s} {'kind':8s} {'mode':16s} {'close':>6s} {'skills':>6s} base_goal")
    for row in rows:
        close = row.get("close_ratio")
        close_text = "-" if close is None else f"{float(close):.2f}"
        print(
            f"{str(row['task_id'])[:32]:32s} "
            f"{str(row['status'])[:14]:14s} "
            f"{str(row.get('demo_kind') or '-')[:8]:8s} "
            f"{str(row.get('hand_mode') or '-')[:16]:16s} "
            f"{close_text:>6s} "
            f"{int(row['skill_count']):>6d} "
            f"{row.get('base_goal') or '-'}"
        )
    for failure in failures:
        print(f"FAILED {failure['task_id']}: {failure['error']}", file=sys.stderr)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "task_id",
        "status",
        "verb",
        "demo_kind",
        "object_id",
        "target_id",
        "skills",
        "skill_count",
        "base_goal",
        "hand_mode",
        "hand",
        "close_ratio",
        "lift_pose_base",
        "place_target",
        "recovery_status",
        "policy_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _short_vec(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return ""
    return ",".join(f"{float(item):.3f}" for item in value[:3])


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
