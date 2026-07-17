#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sonic_world.task_suites import load_robocasa_task_suite, task_executability


DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_OUTCOMES = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"
DEFAULT_POLICY = "reports/policy_data/sonic_general_v0_feedback_adjusted.jsonl"
DEFAULT_OUTPUT_DIR = "reports/task_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the next Sonic task-suite rollout batch from coverage gaps. "
            "The selector expands exact task coverage before repeatedly tuning the same primitive."
        )
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--demo", choices=["ball", "box"])
    parser.add_argument(
        "--executable-tier",
        choices=["current", "all"],
        default="current",
        help=(
            "Task pool to select from. 'current' keeps only tasks supported by the wired "
            "ball/box primitives; 'all' samples the full benchmark suite."
        ),
    )
    parser.add_argument("--include-covered", action="store_true")
    parser.add_argument("--balance-demo", action="store_true", default=True)
    parser.add_argument("--no-balance-demo", dest="balance_demo", action="store_false")
    parser.add_argument("--max-per-bucket", type=int, default=2)
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-action-json", default=DEFAULT_POLICY)
    parser.add_argument("--policy-action-apply", choices=["off", "safe", "full"], default="safe")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--continue-state", action="store_true")
    parser.add_argument("--print-command", action="store_true", default=True)
    parser.add_argument("--no-command", dest="print_command", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    suite = load_robocasa_task_suite(_repo_path(args.suite), repo_root=REPO)
    coverage = _coverage_by_task(args.outcomes)
    all_candidates = [
        _task_meta(index, task, coverage.get(task.task_id, {}), executable_tier=args.executable_tier)
        for index, task in enumerate(suite.tasks)
    ]
    candidates = list(all_candidates)
    if args.demo:
        candidates = [item for item in candidates if item["demo_kind"] == args.demo]
    if args.executable_tier == "current":
        candidates = [item for item in candidates if item["executable"]]
    selected = _select(
        candidates,
        count=int(args.count),
        include_covered=bool(args.include_covered),
        max_per_bucket=int(args.max_per_bucket),
        balance_demo=bool(args.balance_demo and not args.demo),
    )
    if not selected:
        raise SystemExit("No tasks selected.")

    tag = args.tag or f"selected_{time.strftime('%Y%m%d_%H%M%S')}"
    command = _rollout_command(args, selected, tag=tag)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(tag)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    payload = {
        "schema": "sonic_task_rollout_selection_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": suite.name,
        "suite_version": suite.version,
        "suite_path": _rel(_repo_path(args.suite)),
        "outcomes": _rel(_repo_path(args.outcomes)),
        "executable_tier": str(args.executable_tier),
        "selection_count": len(selected),
        "coverage": _coverage_summary(all_candidates, filtered_candidates=candidates),
        "tasks": selected,
        "command": command,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, selected)
    _print_table(selected, coverage_summary=payload["coverage"])
    if args.print_command:
        print("\nCommand:")
        print(" ".join(_shell_quote(part) for part in command))
    print(f"\nWrote task selection JSON: {_rel(json_path)}")
    print(f"Wrote task selection CSV: {_rel(csv_path)}")
    return 0


def _task_meta(index: int, task: Any, coverage: dict[str, Any], *, executable_tier: str) -> dict[str, Any]:
    spec = task.to_dict() if hasattr(task, "to_dict") else {}
    expectation = task.expectation if isinstance(task.expectation, dict) else {}
    request = spec.get("request") if isinstance(spec.get("request"), dict) else {}
    anchor = spec.get("anchor") if isinstance(spec.get("anchor"), dict) else {}
    object_id = str(request.get("object_id") or getattr(task.request, "object_id", "") or "")
    obj = _anchor_object(anchor, object_id)
    category = str((obj or {}).get("category") or _category_from_tags(task.tags) or "unknown")
    affordance = str(
        expectation.get("grasp_affordance")
        or ((request.get("metadata") or {}).get("preferred_grasp_affordance"))
        or "unknown"
    )
    demo = str(expectation.get("demo_kind") or ("box" if affordance == "bimanual_clamp" else "ball"))
    executability = task_executability(task, tier=executable_tier)
    pose_base = executability.pose_base
    return {
        "order": index + 1,
        "task_id": task.task_id,
        "demo_kind": demo,
        "category": category,
        "affordance": affordance,
        "executable": bool(executability.executable),
        "ineligible_reason": "" if executability.executable else executability.reason,
        "pose_base": "" if pose_base is None else ",".join(f"{value:.3f}" for value in pose_base),
        "execution_object_y": "" if executability.execution_object_y is None else f"{executability.execution_object_y:.3f}",
        "verb": str(request.get("verb") or getattr(task.request, "verb", "")),
        "scene": str((spec.get("scene") or {}).get("scene_xml") or ""),
        "tags": list(task.tags),
        "covered": int(coverage.get("count") or 0) > 0,
        "coverage_count": int(coverage.get("count") or 0),
        "success_count": int(coverage.get("success") or 0),
        "last_issue": str(coverage.get("last_issue") or ""),
        "bucket": "|".join([demo, affordance, category]),
    }


def _coverage_by_task(path: str | Path) -> dict[str, dict[str, Any]]:
    p = _repo_path(path)
    out: dict[str, dict[str, Any]] = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            match = row.get("match") if isinstance(row.get("match"), dict) else {}
            rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
            outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
            task_id = str(rollout.get("task_id") or "")
            if not task_id or match.get("type") not in {"task_id_exact", "compatible_demo_kind"}:
                continue
            bucket = out.setdefault(task_id, {"count": 0, "success": 0, "last_issue": ""})
            bucket["count"] += 1
            if outcome.get("success"):
                bucket["success"] += 1
            issue = outcome.get("primary_issue")
            if issue:
                bucket["last_issue"] = str(issue)
    return out


def _select(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    include_covered: bool,
    max_per_bucket: int,
    balance_demo: bool,
) -> list[dict[str, Any]]:
    pool = [item for item in candidates if include_covered or not item["covered"]]
    if not pool:
        pool = list(candidates)
    if balance_demo:
        per_demo = {
            demo: _select_unbalanced(items, count=count, max_per_bucket=max_per_bucket)
            for demo, items in _group_by_demo(pool).items()
        }
        selected: list[dict[str, Any]] = []
        while len(selected) < count:
            progressed = False
            for demo in sorted(per_demo):
                items = per_demo[demo]
                if items and len(selected) < count:
                    selected.append(items.pop(0))
                    progressed = True
            if not progressed:
                break
        selected.sort(key=lambda item: item["order"])
        return selected[:count]
    selected = _select_unbalanced(pool, count=count, max_per_bucket=max_per_bucket)
    selected.sort(key=lambda item: item["order"])
    return selected[:count]


def _select_unbalanced(
    pool: list[dict[str, Any]],
    *,
    count: int,
    max_per_bucket: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in pool:
        buckets.setdefault(item["bucket"], []).append(item)
    for items in buckets.values():
        items.sort(key=lambda item: (item["coverage_count"], item["order"]))

    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    while len(selected) < count:
        progressed = False
        for key in sorted(buckets):
            if len(selected) >= count:
                break
            if bucket_counts.get(key, 0) >= max(1, max_per_bucket):
                continue
            items = buckets[key]
            if not items:
                continue
            selected.append(items.pop(0))
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            progressed = True
        if not progressed:
            break
    if len(selected) < count:
        already = {item["task_id"] for item in selected}
        leftovers = [item for item in pool if item["task_id"] not in already]
        leftovers.sort(key=lambda item: (item["coverage_count"], item["order"]))
        selected.extend(leftovers[: count - len(selected)])
    return selected[:count]


def _group_by_demo(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        out.setdefault(str(item.get("demo_kind") or "unknown"), []).append(item)
    return out


def _rollout_command(args: argparse.Namespace, selected: list[dict[str, Any]], *, tag: str) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/tools/task_suite_rollout.py",
        "--suite",
        str(args.suite),
        "--tag",
        str(tag),
        "--runs-per-task",
        str(args.runs_per_task),
        "--python",
        str(args.python),
    ]
    if args.executable_tier == "current":
        cmd.extend(["--executable-tier", "current"])
    for item in selected:
        cmd.extend(["--task", item["task_id"]])
    if not args.gui:
        cmd.append("--headless")
    if args.camera:
        cmd.append("--camera")
    if not args.continue_state:
        cmd.append("--reset-each-rollout")
    else:
        cmd.append("--continue-state")
    if args.policy_action_json and args.policy_action_apply != "off":
        cmd.extend(
            [
                "--policy-action-json",
                str(args.policy_action_json),
                "--policy-action-apply",
                str(args.policy_action_apply),
            ]
        )
    return cmd


def _coverage_summary(
    items: list[dict[str, Any]],
    *,
    filtered_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    covered = [item for item in items if item["covered"]]
    executable = [item for item in items if item.get("executable")]
    selected_pool = filtered_candidates if filtered_candidates is not None else items
    demos: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = demos.setdefault(item["demo_kind"], {"total": 0, "covered": 0, "executable": 0})
        bucket["total"] += 1
        if item["covered"]:
            bucket["covered"] += 1
        if item.get("executable"):
            bucket["executable"] += 1
    return {
        "task_count": len(items),
        "covered_task_count": len(covered),
        "uncovered_task_count": len(items) - len(covered),
        "executable_task_count": len(executable),
        "selection_pool_count": len(selected_pool),
        "by_demo": dict(sorted(demos.items())),
    }


def _anchor_object(anchor: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in anchor.get("objects") or []:
        if isinstance(obj, dict) and str(obj.get("object_id") or "") == object_id:
            return obj
    return None


def _category_from_tags(tags: tuple[str, ...] | list[str]) -> str:
    skip = {
        "move",
        "pick",
        "place",
        "short",
        "tabletop",
        "clutter",
        "nav",
        "sequence",
        "single_hand_pinch",
        "side_grasp",
        "top_grasp",
        "bimanual_clamp",
    }
    for tag in tags:
        if str(tag) not in skip:
            return str(tag)
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "order",
        "task_id",
        "demo_kind",
        "category",
        "affordance",
        "executable",
        "ineligible_reason",
        "pose_base",
        "execution_object_y",
        "verb",
        "covered",
        "coverage_count",
        "success_count",
        "last_issue",
        "bucket",
        "scene",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_table(rows: list[dict[str, Any]], *, coverage_summary: dict[str, Any]) -> None:
    print(
        "task_selection="
        f"{len(rows)} coverage={coverage_summary['covered_task_count']}/{coverage_summary['task_count']} "
        f"uncovered={coverage_summary['uncovered_task_count']} "
        f"executable={coverage_summary['executable_task_count']} "
        f"pool={coverage_summary['selection_pool_count']}"
    )
    print(f"{'task_id':34s} {'demo':5s} {'affordance':18s} {'cat':12s} {'seen':>4s} {'pose_base':>18s} issue")
    for row in rows:
        print(
            f"{row['task_id'][:34]:34s} "
            f"{row['demo_kind'][:5]:5s} "
            f"{row['affordance'][:18]:18s} "
            f"{row['category'][:12]:12s} "
            f"{int(row['coverage_count']):>4d} "
            f"{str(row.get('pose_base') or '-')[:18]:>18s} "
            f"{row['last_issue'] or '-'}"
        )


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _shell_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%"
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
