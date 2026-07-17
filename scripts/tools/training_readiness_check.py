#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent

DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_ROLLOUTS = "reports/rollouts"
DEFAULT_POLICY = "reports/policy_data/sonic_general_v0_heuristic.jsonl"
DEFAULT_ADJUSTED_POLICY = "reports/policy_data/sonic_general_v0_feedback_adjusted.jsonl"
DEFAULT_OUTCOMES = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"
DEFAULT_EPISODES = "reports/datasets/sonic_rollout_episodes.jsonl"
DEFAULT_FEEDBACK = "reports/policy_outcomes/sonic_feedback_profile.json"
DEFAULT_MODEL_DIR = "reports/policy_models"
DEFAULT_MODEL_NAME = "task_policy_memory_v0"
DEFAULT_ACTIONS = "reports/policy_data/task_policy_memory_actions.jsonl"
DEFAULT_REPORT_DIR = "reports/readiness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Sonic world-model/task-policy training readiness chain. "
            "This validates only the high-level task/skill policy layer; SONIC low-level control stays frozen."
        )
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--rollouts", nargs="*", default=[DEFAULT_ROLLOUTS])
    parser.add_argument("--offline-limit", type=int, default=50, help="Offline benchmark task limit. Use 0 for the full suite.")
    parser.add_argument("--policy-limit", type=int, default=0, help="Teacher policy sample limit. Use 0 for the full suite.")
    parser.add_argument("--benchmark-name", default="sonic_general_v0_readiness_check")
    parser.add_argument("--policy-jsonl", default=DEFAULT_POLICY)
    parser.add_argument("--adjusted-policy-jsonl", default=DEFAULT_ADJUSTED_POLICY)
    parser.add_argument("--outcomes-jsonl", default=DEFAULT_OUTCOMES)
    parser.add_argument("--episodes-jsonl", default=DEFAULT_EPISODES)
    parser.add_argument("--feedback-json", default=DEFAULT_FEEDBACK)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--actions-jsonl", default=DEFAULT_ACTIONS)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--include-planning", action="store_true", default=True)
    parser.add_argument("--no-planning", dest="include_planning", action="store_false")
    parser.add_argument("--include-events", action="store_true", help="Embed compact rollout timelines in episode data.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = _repo_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"training_readiness_{stamp}.json"
    latest_path = report_dir / "training_readiness_latest.json"
    md_path = report_dir / "training_readiness_latest.md"

    steps = _build_steps(args)
    results: list[dict[str, Any]] = []
    failed = False
    print(f"training_readiness suite={args.suite} steps={len(steps)}")
    for step in steps:
        result = _run_step(step)
        results.append(result)
        status = "ok" if result["returncode"] == 0 else "fail"
        print(f"[{status}] {step['name']} ({result['elapsed_s']:.2f}s)")
        if result["returncode"] != 0:
            failed = True
            _print_tail(result)
            if not args.continue_on_error:
                break

    metrics = _collect_metrics(args)
    report = {
        "schema": "sonic_training_readiness_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "controller_boundary": "frozen_sonic_low_level",
        "training_scope": "task_and_skill_policy_only",
        "suite": args.suite,
        "rollouts": args.rollouts,
        "status": "failed" if failed else _readiness_status(metrics),
        "steps": results,
        "metrics": metrics,
        "artifacts": _artifact_manifest(args),
        "next_batch": _next_batch_plan(metrics),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    report_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    if args.print_json:
        print(text)
    else:
        _print_summary(report)
    print(f"\nWrote readiness report: {_rel(report_path)}")
    print(f"Wrote latest readiness report: {_rel(latest_path)}")
    print(f"Wrote latest readiness note: {_rel(md_path)}")
    return 1 if failed else 0


def _build_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    benchmark_cmd = [
        sys.executable,
        "scripts/tools/benchmark_runner.py",
        "--suite",
        args.suite,
        "--name",
        args.benchmark_name,
    ]
    if int(args.offline_limit) > 0:
        benchmark_cmd.extend(["--limit", str(int(args.offline_limit))])

    policy_cmd = [
        sys.executable,
        "scripts/tools/policy_dataset_builder.py",
        "--suite",
        args.suite,
        "--output",
        args.policy_jsonl,
        "--summary",
        str(_repo_path(args.policy_jsonl).with_suffix(".csv")),
    ]
    if int(args.policy_limit) > 0:
        policy_cmd.extend(["--limit", str(int(args.policy_limit))])
    if args.include_planning:
        policy_cmd.append("--include-planning")

    outcome_cmd = [
        sys.executable,
        "scripts/tools/policy_outcome_joiner.py",
        "--policy-jsonl",
        args.policy_jsonl,
        "--rollouts",
        *args.rollouts,
        "--output",
        args.outcomes_jsonl,
        "--summary",
        str(_repo_path(args.outcomes_jsonl).with_suffix(".csv")),
    ]

    episode_cmd = [
        sys.executable,
        "scripts/tools/rollout_dataset_builder.py",
        "--rollouts",
        *args.rollouts,
        "--policy-outcomes",
        args.outcomes_jsonl,
        "--suite",
        args.suite,
        "--output",
        args.episodes_jsonl,
        "--summary",
        str(_repo_path(args.episodes_jsonl).with_suffix(".csv")),
    ]
    if not args.include_events:
        episode_cmd.append("--no-events")

    feedback_cmd = [
        sys.executable,
        "scripts/tools/policy_feedback_report.py",
        "--input",
        args.outcomes_jsonl,
        "--output",
        args.feedback_json,
        "--min-count",
        "1",
    ]

    adjusted_cmd = [
        sys.executable,
        "scripts/tools/policy_apply_feedback.py",
        "--policy-jsonl",
        args.policy_jsonl,
        "--feedback",
        args.feedback_json,
        "--output",
        args.adjusted_policy_jsonl,
        "--summary",
        str(_repo_path(args.adjusted_policy_jsonl).with_suffix(".csv")),
        "--min-count",
        "2",
        "--max-modes",
        "4",
    ]

    train_cmd = [
        sys.executable,
        "scripts/tools/task_policy_train.py",
        "--input",
        args.outcomes_jsonl,
        "--output-dir",
        args.model_dir,
        "--name",
        args.model_name,
    ]

    export_cmd = [
        sys.executable,
        "scripts/tools/task_policy_export.py",
        "--model",
        str(_repo_path(args.model_dir) / f"{args.model_name}.json"),
        "--output",
        args.actions_jsonl,
        "--summary",
        str(_repo_path(args.actions_jsonl).with_suffix(".csv")),
    ]

    return [
        {"name": "offline_benchmark", "cmd": benchmark_cmd},
        {"name": "teacher_policy_dataset", "cmd": policy_cmd},
        {"name": "policy_outcome_join", "cmd": outcome_cmd},
        {"name": "rollout_episode_dataset", "cmd": episode_cmd},
        {"name": "feedback_profile", "cmd": feedback_cmd},
        {"name": "feedback_adjusted_policy", "cmd": adjusted_cmd},
        {"name": "task_policy_train", "cmd": train_cmd},
        {"name": "task_policy_export", "cmd": export_cmd},
    ]


def _run_step(step: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        step["cmd"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "name": step["name"],
        "cmd": step["cmd"],
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - start, 4),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _collect_metrics(args: argparse.Namespace) -> dict[str, Any]:
    policy_rows = _read_jsonl(_repo_path(args.policy_jsonl))
    adjusted_rows = _read_jsonl(_repo_path(args.adjusted_policy_jsonl))
    outcome_rows = _read_jsonl(_repo_path(args.outcomes_jsonl))
    episode_rows = _read_jsonl(_repo_path(args.episodes_jsonl))
    feedback = _read_json(_repo_path(args.feedback_json))
    model = _read_json(_repo_path(args.model_dir) / f"{args.model_name}.json")
    actions = _read_jsonl(_repo_path(args.actions_jsonl))
    benchmark = _read_json(REPO / "reports" / "benchmarks" / f"{args.benchmark_name}.json")
    outcomes = [_outcome(row) for row in outcome_rows]
    matches = [_match(row) for row in outcome_rows]
    qualities: dict[str, int] = {}
    issues: dict[str, int] = {}
    task_coverage: dict[str, int] = {}
    exact_suite_task_coverage: dict[str, int] = {}
    for row, outcome in zip(outcome_rows, outcomes):
        quality = str(outcome.get("quality") or "unknown")
        qualities[quality] = qualities.get(quality, 0) + 1
        issue = str(outcome.get("primary_issue") or "-")
        issues[issue] = issues.get(issue, 0) + 1
        rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        task_id = str(rollout.get("task_id") or "unknown")
        task_coverage[task_id] = task_coverage.get(task_id, 0) + 1
        if match.get("type") == "task_id_exact" and task_id not in {"unknown", "ball_demo", "box_demo"}:
            exact_suite_task_coverage[task_id] = exact_suite_task_coverage.get(task_id, 0) + 1

    exact_policy = model.get("exact_task_policy") if isinstance(model.get("exact_task_policy"), dict) else {}
    feedback_summary = feedback.get("summary") if isinstance(feedback.get("summary"), dict) else {}
    benchmark_summary = benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
    return {
        "offline": {
            "task_count": int(benchmark_summary.get("task_count") or 0),
            "success_count": int(benchmark_summary.get("offline_success_count") or 0),
            "success_rate": float(benchmark_summary.get("offline_success_rate") or 0.0),
            "scene_valid_count": int(benchmark_summary.get("scene_valid_count") or 0),
            "plan_ready_count": int(benchmark_summary.get("plan_ready_count") or 0),
        },
        "teacher_policy": {
            "sample_count": len(policy_rows),
            "adjusted_sample_count": len(adjusted_rows),
        },
        "rollout_outcomes": {
            "outcome_count": len(outcome_rows),
            "matched_count": sum(1 for match in matches if match.get("type") != "none"),
            "task_id_exact_count": sum(1 for match in matches if match.get("type") == "task_id_exact"),
            "success_count": sum(1 for outcome in outcomes if outcome.get("success")),
            "failed_count": sum(1 for outcome in outcomes if outcome.get("final_status") == "failed"),
            "skipped_count": sum(1 for outcome in outcomes if outcome.get("final_status") == "skipped"),
            "quality_counts": dict(sorted(qualities.items())),
            "top_issues": _top_counts(issues),
            "covered_task_count": len([task for task in task_coverage if task not in {"unknown", "ball_demo", "box_demo"}]),
            "exact_suite_task_coverage_count": len(exact_suite_task_coverage),
            "exact_suite_task_coverage": dict(sorted(exact_suite_task_coverage.items())),
            "task_coverage": dict(sorted(task_coverage.items())),
        },
        "episodes": {
            "episode_count": len(episode_rows),
        },
        "feedback": {
            "failure_mode_count": int(feedback_summary.get("failure_mode_count") or len(feedback.get("failure_modes") or [])),
            "avg_dense_score": float(feedback_summary.get("avg_dense_score") or 0.0),
            "top_failure_modes": _top_failure_modes(feedback),
        },
        "policy_model": {
            "example_count": int(model.get("example_count") or 0),
            "positive_count": int(model.get("positive_count") or 0),
            "negative_count": int(model.get("negative_count") or 0),
            "train_count": int(model.get("train_count") or 0),
            "val_count": int(model.get("val_count") or 0),
            "exact_task_policy_count": len(exact_policy),
            "exported_action_count": len(actions),
        },
    }


def _readiness_status(metrics: dict[str, Any]) -> str:
    offline = metrics.get("offline", {})
    policy = metrics.get("teacher_policy", {})
    outcomes = metrics.get("rollout_outcomes", {})
    model = metrics.get("policy_model", {})
    if int(policy.get("sample_count") or 0) <= 0:
        return "blocked_no_teacher_policy"
    if int(outcomes.get("matched_count") or 0) <= 0:
        return "blocked_no_matched_rollouts"
    if int(model.get("example_count") or 0) <= 0:
        return "blocked_no_training_examples"
    if float(offline.get("success_rate") or 0.0) < 1.0:
        return "needs_offline_suite_fix"
    if int(model.get("positive_count") or 0) <= 0:
        return "needs_positive_rollouts"
    return "ready_for_next_rollout_batch"


def _next_batch_plan(metrics: dict[str, Any]) -> dict[str, Any]:
    outcomes = metrics.get("rollout_outcomes", {})
    top_issues = outcomes.get("top_issues") or []
    return {
        "recommended_mode": "reuse_stack_headless_with_reset",
        "minimum_next_rollouts": 20,
        "target_mix": {
            "existing_exact_tasks": [
                "ball_left_to_tray",
                "fruit_right_to_plate",
                "small_cube_pick",
                "apartment_package_table_pick",
            ],
            "new_uncovered_tasks": "sample from sonic_general_v0 by category and affordance",
        },
        "why": (
            "Teacher coverage exists for the full suite, but real rollout policy coverage is only "
            f"{int(outcomes.get('exact_suite_task_coverage_count') or 0)} exact suite tasks. The next batch should expand exact coverage "
            "before tuning individual primitive constants."
        ),
        "watch_first": [item["key"] for item in top_issues[:5]],
    }


def _artifact_manifest(args: argparse.Namespace) -> dict[str, str]:
    return {
        "policy_jsonl": _rel(_repo_path(args.policy_jsonl)),
        "adjusted_policy_jsonl": _rel(_repo_path(args.adjusted_policy_jsonl)),
        "outcomes_jsonl": _rel(_repo_path(args.outcomes_jsonl)),
        "episodes_jsonl": _rel(_repo_path(args.episodes_jsonl)),
        "feedback_json": _rel(_repo_path(args.feedback_json)),
        "policy_model": _rel(_repo_path(args.model_dir) / f"{args.model_name}.json"),
        "actions_jsonl": _rel(_repo_path(args.actions_jsonl)),
    }


def _print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    offline = metrics["offline"]
    outcomes = metrics["rollout_outcomes"]
    model = metrics["policy_model"]
    print()
    print(f"readiness_status={report['status']}")
    print(
        "offline="
        f"{offline['success_count']}/{offline['task_count']} "
        f"teacher={metrics['teacher_policy']['sample_count']} "
        f"outcomes={outcomes['outcome_count']} matched={outcomes['matched_count']} "
        f"suite_exact={outcomes['exact_suite_task_coverage_count']} "
        f"policy_entries={model['exact_task_policy_count']} actions={model['exported_action_count']}"
    )
    print(
        "policy_examples="
        f"{model['example_count']} positive={model['positive_count']} negative={model['negative_count']} "
        f"train={model['train_count']} val={model['val_count']}"
    )
    print("top_issues=" + ", ".join(f"{item['key']}:{item['count']}" for item in outcomes["top_issues"][:6]))
    print("top_feedback_modes=")
    for mode in metrics["feedback"]["top_failure_modes"][:6]:
        print(f"  {mode['demo_kind']}/{mode['stage']} {mode['issue']} count={mode['count']} score={mode['avg_dense_score']}")


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    offline = metrics["offline"]
    outcomes = metrics["rollout_outcomes"]
    model = metrics["policy_model"]
    lines = [
        "# Sonic Training Readiness",
        "",
        f"- status: `{report['status']}`",
        f"- offline: `{offline['success_count']}/{offline['task_count']}`",
        f"- teacher policy samples: `{metrics['teacher_policy']['sample_count']}`",
        f"- rollout outcomes: `{outcomes['outcome_count']}` matched `{outcomes['matched_count']}`",
        f"- exact rollout-covered suite tasks: `{outcomes['exact_suite_task_coverage_count']}`",
        f"- policy memory entries: `{model['exact_task_policy_count']}`",
        f"- policy examples: `{model['example_count']}` positive `{model['positive_count']}` negative `{model['negative_count']}`",
        f"- exported actions: `{model['exported_action_count']}`",
        "",
        "## Top Issues",
    ]
    for item in outcomes["top_issues"][:8]:
        lines.append(f"- `{item['key']}`: {item['count']}")
    lines.extend(["", "## Next Batch", ""])
    plan = report.get("next_batch") or {}
    lines.append(f"- mode: `{plan.get('recommended_mode')}`")
    lines.append(f"- minimum rollouts: `{plan.get('minimum_next_rollouts')}`")
    lines.append(f"- watch first: `{', '.join(plan.get('watch_first') or [])}`")
    lines.append("")
    return "\n".join(lines)


def _outcome(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("outcome") if isinstance(row.get("outcome"), dict) else {}


def _match(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("match") if isinstance(row.get("match"), dict) else {}


def _top_failure_modes(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    modes = [mode for mode in feedback.get("failure_modes") or [] if isinstance(mode, dict)]
    modes.sort(key=lambda item: (-int(item.get("count") or 0), str(item.get("demo_kind") or ""), str(item.get("issue") or "")))
    return [
        {
            "demo_kind": str(item.get("demo_kind") or ""),
            "stage": str(item.get("stage") or ""),
            "issue": str(item.get("issue") or ""),
            "count": int(item.get("count") or 0),
            "avg_dense_score": float(item.get("avg_dense_score") or 0.0),
        }
        for item in modes[:12]
    ]


def _top_counts(counts: dict[str, int], *, limit: int = 12) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _print_tail(result: dict[str, Any]) -> None:
    if result.get("stdout_tail"):
        print(result["stdout_tail"])
    if result.get("stderr_tail"):
        print(result["stderr_tail"], file=sys.stderr)


def _tail(text: str, *, max_lines: int = 24) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


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
