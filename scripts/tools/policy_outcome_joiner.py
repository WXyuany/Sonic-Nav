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


DEFAULT_POLICY = "reports/policy_data/sonic_general_v0_heuristic.jsonl"
DEFAULT_ROLLOUTS = "reports/rollouts"
DEFAULT_OUTPUT = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join task/skill policy samples with real rollout outcomes. "
            "The result is a high-level policy dataset; it never contains raw SONIC joint targets."
        )
    )
    parser.add_argument("--policy-jsonl", default=DEFAULT_POLICY)
    parser.add_argument("--rollouts", nargs="*", default=[DEFAULT_ROLLOUTS])
    parser.add_argument("--run-id-prefix", action="append", help="Only include rollouts whose run_id starts with this prefix.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", help="CSV summary path. Defaults to <output>.csv.")
    parser.add_argument("--include-events", action="store_true", help="Embed compact rollout events in each JSONL row.")
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--require-policy-match", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_samples = _read_policy_samples(args.policy_jsonl)
    events = list(_read_events(args.rollouts))
    summaries = summarize(events)
    if args.run_id_prefix:
        prefixes = tuple(str(item) for item in args.run_id_prefix)
        summaries = [row for row in summaries if str(row.get("run_id") or "").startswith(prefixes)]
    grouped_events = _events_by_run(events)
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for summary in summaries:
        sample, match = _match_policy_sample(summary, policy_samples)
        if sample is None and args.require_policy_match:
            continue
        outcome = _outcome_from_summary(summary)
        row = {
            "schema": "task_skill_policy_outcome_v0",
            "link_id": _link_id(sample, summary),
            "match": match,
            "observation": (sample or {}).get("observation"),
            "teacher_action": (sample or {}).get("action"),
            "outcome": outcome,
            "rollout": {
                "run_id": summary.get("run_id"),
                "demo_kind": summary.get("demo_kind"),
                "task_id": summary.get("task_id"),
                "scene": summary.get("scene"),
                "summary": summary,
            },
            "metadata": {
                "controller_boundary": "frozen_sonic_low_level",
                "training_scope": "task_and_skill_policy_only",
                "source_policy_jsonl": _rel(_repo_path(args.policy_jsonl)),
                "source_rollouts": [_rel(_repo_path(path)) for path in args.rollouts],
            },
        }
        if args.include_events:
            row["rollout"]["events"] = _compact_events(
                grouped_events.get(str(summary.get("run_id") or "unknown"), []),
                limit=max(0, int(args.max_events)),
            )
        rows.append(row)
        csv_rows.append(_csv_row(row))

    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_path, csv_rows)
    if args.print_json:
        print(json.dumps({"outcomes": rows}, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(csv_rows)
    print(f"\nWrote policy outcome JSONL: {_rel(output)}")
    print(f"Wrote policy outcome summary: {_rel(summary_path)}")
    return 0


def _read_policy_samples(path: str | Path) -> list[dict[str, Any]]:
    p = _repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy JSONL not found: {p}")
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"bad policy JSONL row at {p}:{line_no}")
            out.append(payload)
    if not out:
        raise ValueError(f"policy JSONL has no samples: {p}")
    return out


def _events_by_run(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("run_id") or "unknown"), []).append(event)
    for items in grouped.values():
        items.sort(key=lambda event: float(event.get("monotonic") or event.get("stamp") or 0.0))
    return grouped


def _match_policy_sample(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rollout_task = str(summary.get("task_id") or "")
    rollout_kind = str(summary.get("demo_kind") or "")
    exact = [sample for sample in samples if _sample_task_id(sample) == rollout_task]
    if exact:
        return exact[0], {"type": "task_id_exact", "score": 100, "reason": rollout_task}

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for sample in samples:
        action = sample.get("action") if isinstance(sample.get("action"), dict) else {}
        intent = action.get("task_intent") if isinstance(action.get("task_intent"), dict) else {}
        metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
        sample_kind = str(intent.get("demo_kind") or "")
        grasp = str(metadata.get("grasp_affordance") or "")
        score = 0
        reasons: list[str] = []
        if rollout_kind and sample_kind == rollout_kind:
            score += 50
            reasons.append("demo_kind")
        if rollout_task == "ball_demo" and grasp == "single_hand_pinch":
            score += 20
            reasons.append("ball_demo_single_hand_pinch")
        if rollout_task == "box_demo" and grasp == "bimanual_clamp":
            score += 20
            reasons.append("box_demo_bimanual_clamp")
        if score:
            scored.append((score, ",".join(reasons), sample))

    if not scored:
        return None, {"type": "none", "score": 0, "reason": "no compatible policy sample"}
    scored.sort(key=lambda item: (-item[0], _sample_task_id(item[2])))
    score, reason, sample = scored[0]
    return sample, {
        "type": "compatible_demo_kind",
        "score": score,
        "reason": reason,
        "matched_sample_id": sample.get("sample_id"),
        "matched_task_id": _sample_task_id(sample),
    }


def _outcome_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    final_status = str(summary.get("final_status") or "unknown")
    retry_count = int(summary.get("retry_count") or 0)
    lift_success = _as_bool(summary.get("lift_success"))
    lift_failed = _as_bool(summary.get("lift_failed"))
    fail_stage = str(summary.get("fail_stage") or "")
    fail_reason = str(summary.get("fail_reason") or "")
    retry_stage = str(summary.get("retry_stage") or "")
    retry_reason = str(summary.get("retry_reason") or "")
    stage_labels = {
        stage: {
            "events": int(summary.get(f"{stage}_events") or 0),
            "failures": int(summary.get(f"{stage}_failures") or 0),
            "retries": int(summary.get(f"{stage}_retries") or 0),
        }
        for stage in STAGES
    }
    success = final_status == "success"
    terminal_stage = fail_stage or ("done" if success else "unknown")
    primary_issue = fail_reason or retry_reason or fail_stage or retry_stage or ""
    return {
        "final_status": final_status,
        "supervision": _supervision_label(success, retry_count, final_status),
        "quality": _quality_label(success, retry_count, final_status),
        "success": success,
        "clean_success": success and retry_count == 0,
        "recovered_success": success and retry_count > 0,
        "lift_success": lift_success,
        "lift_failed": lift_failed,
        "retry_count": retry_count,
        "terminal_stage": terminal_stage,
        "fail_stage": fail_stage,
        "fail_reason": fail_reason,
        "retry_stage": retry_stage,
        "retry_reason": retry_reason,
        "primary_issue": primary_issue,
        "stage_labels": stage_labels,
        "dense_score": _dense_score(success, lift_success, stage_labels),
        "correction_targets": _correction_targets(stage_labels, terminal_stage, primary_issue),
    }


def _supervision_label(success: bool, retry_count: int, final_status: str) -> str:
    if success and retry_count == 0:
        return "positive_clean"
    if success:
        return "positive_with_recovery"
    if final_status in {"unknown", ""}:
        return "unknown"
    return "negative"


def _quality_label(success: bool, retry_count: int, final_status: str) -> str:
    if final_status in {"unknown", "", "skipped", "interrupted"}:
        return final_status or "unknown"
    if not success:
        return "failed"
    if retry_count == 0:
        return "clean_success"
    if retry_count <= 2:
        return "minor_recovery_success"
    if retry_count <= 6:
        return "rough_success"
    return "poor_success"


def _dense_score(success: bool, lift_success: bool, stage_labels: dict[str, dict[str, int]]) -> float:
    score = 0.0
    stage_weights = {
        "approach": 0.12,
        "workspace": 0.14,
        "grasp": 0.20,
        "lift": 0.22,
        "transport": 0.12,
        "place": 0.12,
        "done": 0.08,
    }
    for stage, weight in stage_weights.items():
        label = stage_labels.get(stage) or {}
        if int(label.get("events") or 0) > 0 and int(label.get("failures") or 0) == 0:
            score += weight
    if lift_success:
        score += 0.06
    if success:
        score += 0.14
    raw_score = min(1.0, score)
    retry_penalty = min(0.60, 0.045 * sum(int(v.get("retries") or 0) for v in stage_labels.values()))
    failure_penalty = min(0.70, 0.16 * sum(int(v.get("failures") or 0) for v in stage_labels.values()))
    return round(max(0.0, min(1.0, raw_score - retry_penalty - failure_penalty)), 4)


def _correction_targets(
    stage_labels: dict[str, dict[str, int]],
    terminal_stage: str,
    primary_issue: str,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    stage_to_outputs = {
        "approach": ["base_goal", "object_target_anchors", "grasp_offsets"],
        "workspace": ["base_goal", "hand_pose_target", "grasp_offsets"],
        "grasp": ["hand_pose_target", "wrist_target", "grasp_close_ratio", "grasp_offsets"],
        "lift": ["grasp_close_ratio", "lift_place_targets", "hand_pose_target"],
        "transport": ["base_goal", "lift_place_targets"],
        "place": ["lift_place_targets", "hand_pose_target"],
        "fall": ["recovery_decision", "base_goal", "lift_place_targets"],
        "unknown": ["recovery_decision"],
    }
    for stage, label in stage_labels.items():
        retries = int(label.get("retries") or 0)
        failures = int(label.get("failures") or 0)
        if retries == 0 and failures == 0 and stage != terminal_stage:
            continue
        if stage not in stage_to_outputs:
            continue
        severity = "failure" if failures else ("terminal" if stage == terminal_stage and primary_issue else "retry")
        targets.append(
            {
                "stage": stage,
                "severity": severity,
                "retry_count": retries,
                "failure_count": failures,
                "affected_outputs": stage_to_outputs[stage],
                "issue": primary_issue if stage == terminal_stage or failures else "",
            }
        )
    if not targets and terminal_stage in stage_to_outputs and primary_issue:
        targets.append(
            {
                "stage": terminal_stage,
                "severity": "terminal",
                "retry_count": 0,
                "failure_count": 0,
                "affected_outputs": stage_to_outputs[terminal_stage],
                "issue": primary_issue,
            }
        )
    return targets


def _compact_events(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events[:limit]:
        out.append(
            {
                "event": event.get("event"),
                "phase": event.get("phase"),
                "primitive_stage": event.get("primitive_stage"),
                "skill_name": event.get("skill_name"),
                "status": event.get("status"),
                "reason": event.get("reason"),
                "metrics": event.get("metrics") or {},
                "metadata": event.get("metadata") or {},
                "source_file": event.get("_source_file"),
            }
        )
    return out


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    outcome = row["outcome"]
    match = row["match"]
    rollout = row["rollout"]
    teacher = row.get("teacher_action") or {}
    metadata = teacher.get("metadata") if isinstance(teacher.get("metadata"), dict) else {}
    return {
        "link_id": row["link_id"],
        "run_id": rollout.get("run_id"),
        "rollout_task_id": rollout.get("task_id"),
        "demo_kind": rollout.get("demo_kind"),
        "policy_task_id": teacher.get("task_id"),
        "match_type": match.get("type"),
        "match_score": match.get("score"),
        "grasp_affordance": metadata.get("grasp_affordance"),
        "final_status": outcome.get("final_status"),
        "supervision": outcome.get("supervision"),
        "quality": outcome.get("quality"),
        "dense_score": outcome.get("dense_score"),
        "retry_count": outcome.get("retry_count"),
        "terminal_stage": outcome.get("terminal_stage"),
        "primary_issue": outcome.get("primary_issue"),
        "correction_targets": ",".join(
            target["stage"] + ":" + "|".join(target["affected_outputs"])
            for target in outcome.get("correction_targets", [])
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "link_id",
        "run_id",
        "rollout_task_id",
        "demo_kind",
        "policy_task_id",
        "match_type",
        "match_score",
        "grasp_affordance",
        "final_status",
        "supervision",
        "quality",
        "dense_score",
        "retry_count",
        "terminal_stage",
        "primary_issue",
        "correction_targets",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"policy_outcomes={len(rows)}")
    print(
        f"{'run_id':28s} {'demo':8s} {'policy_task':28s} {'match':18s} "
        f"{'status':10s} {'score':>5s} {'retry':>5s} stage issue"
    )
    for row in rows:
        print(
            f"{str(row.get('run_id') or '-')[:28]:28s} "
            f"{str(row.get('demo_kind') or '-')[:8]:8s} "
            f"{str(row.get('policy_task_id') or '-')[:28]:28s} "
            f"{str(row.get('match_type') or '-')[:18]:18s} "
            f"{str(row.get('final_status') or '-')[:10]:10s} "
            f"{float(row.get('dense_score') or 0.0):5.2f} "
            f"{int(row.get('retry_count') or 0):5d} "
            f"{str(row.get('terminal_stage') or '-')[:10]:10s} "
            f"{row.get('primary_issue') or '-'}"
        )


def _link_id(sample: dict[str, Any] | None, summary: dict[str, Any]) -> str:
    sample_id = str((sample or {}).get("sample_id") or "unmatched_policy")
    run_id = str(summary.get("run_id") or "unknown_run")
    return f"{sample_id}::{run_id}"


def _sample_task_id(sample: dict[str, Any]) -> str:
    action = sample.get("action") if isinstance(sample.get("action"), dict) else {}
    return str(action.get("task_id") or sample.get("sample_id") or "")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


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
