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


DEFAULT_INPUT = "reports/policy_outcomes/sonic_policy_outcomes.jsonl"
DEFAULT_OUTPUT = "reports/policy_outcomes/sonic_feedback_profile.json"


RULES: dict[str, dict[str, Any]] = {
    "approach_still_far": {
        "affected_outputs": ["base_goal", "object_target_anchors"],
        "adjustments": {
            "standoff_delta_m": -0.04,
            "walk_duration_delta_s": 0.35,
            "require_runtime_anchor_refresh": True,
        },
        "note": "Robot stops too far from the object; close the base standoff or extend approach.",
    },
    "workspace_alignment_residual": {
        "affected_outputs": ["base_goal", "hand_pose_target", "grasp_offsets"],
        "adjustments": {
            "enable_lateral_micro_step": True,
            "max_lateral_step_m": 0.06,
            "recompute_hand_target_from_latest_anchor": True,
        },
        "note": "Object is in view but not centered in manipulation workspace.",
    },
    "capture_contact_not_ready": {
        "affected_outputs": ["hand_pose_target", "wrist_target", "grasp_close_ratio", "grasp_offsets"],
        "adjustments": {
            "contact_x_delta_m": 0.025,
            "contact_z_delta_m": -0.012,
            "secure_aperture_delta_m": -0.006,
            "close_ratio_delta": 0.06,
        },
        "note": "Hand reaches near the object but contact geometry is not ready.",
    },
    "palm_pocket_not_ready": {
        "affected_outputs": ["hand_pose_target", "wrist_target", "grasp_offsets"],
        "adjustments": {
            "palm_x_delta_m": 0.02,
            "palm_z_delta_m": -0.008,
            "wrist_pitch_delta_rad": -0.06,
        },
        "note": "Object is not settled in the palm/finger pocket before closing.",
    },
    "lift_delta_below_threshold": {
        "affected_outputs": ["grasp_close_ratio", "lift_place_targets", "hand_pose_target"],
        "adjustments": {
            "close_ratio_delta": 0.08,
            "hold_before_lift_delta_s": 0.25,
            "lift_z_delta_m": 0.03,
            "abort_if_contact_not_confirmed": True,
        },
        "note": "Object did not move with the hand during lift.",
    },
    "missing_or_implausible_anchor": {
        "affected_outputs": ["object_target_anchors", "recovery_decision"],
        "adjustments": {
            "request_scene_reset": True,
            "request_reobserve_from_current_view": True,
            "do_not_use_as_positive_action_sample": True,
        },
        "note": "Object anchor is missing or physically implausible; this is perception/state recovery first.",
    },
}

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "approach": {
        "affected_outputs": ["base_goal", "object_target_anchors"],
        "adjustments": {"standoff_delta_m": -0.03},
    },
    "workspace": {
        "affected_outputs": ["base_goal", "hand_pose_target", "grasp_offsets"],
        "adjustments": {"enable_lateral_micro_step": True},
    },
    "grasp": {
        "affected_outputs": ["hand_pose_target", "wrist_target", "grasp_close_ratio", "grasp_offsets"],
        "adjustments": {"close_ratio_delta": 0.04},
    },
    "lift": {
        "affected_outputs": ["grasp_close_ratio", "lift_place_targets", "hand_pose_target"],
        "adjustments": {"close_ratio_delta": 0.06},
    },
    "place": {
        "affected_outputs": ["lift_place_targets", "hand_pose_target"],
        "adjustments": {"place_z_delta_m": 0.02},
    },
    "fall": {
        "affected_outputs": ["recovery_decision", "base_goal", "lift_place_targets"],
        "adjustments": {"lower_carry_height": True, "abort_if_unstable": True},
    },
    "unknown": {
        "affected_outputs": ["recovery_decision"],
        "adjustments": {"manual_review": True},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate policy outcome JSONL into a high-level feedback profile."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", help="CSV output path. Defaults to <output>.csv.")
    parser.add_argument("--markdown", help="Markdown output path. Defaults to <output>.md.")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.input)
    profile = build_profile(rows, min_count=max(1, int(args.min_count)))
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    issue_rows = profile["failure_modes"]

    csv_path = _repo_path(args.csv) if args.csv else output.with_suffix(".csv")
    _write_csv(csv_path, issue_rows)
    md_path = _repo_path(args.markdown) if args.markdown else output.with_suffix(".md")
    md_path.write_text(_markdown(profile), encoding="utf-8")

    if args.print_json:
        print(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(profile)
    print(f"\nWrote feedback profile: {_rel(output)}")
    print(f"Wrote feedback CSV: {_rel(csv_path)}")
    print(f"Wrote feedback Markdown: {_rel(md_path)}")
    return 0


def build_profile(rows: list[dict[str, Any]], *, min_count: int) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    qualities: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    matched = 0
    scores: list[float] = []
    for row in rows:
        rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        demo = str(rollout.get("demo_kind") or "unknown")
        quality = str(outcome.get("quality") or "unknown")
        demo_counts[demo] = demo_counts.get(demo, 0) + 1
        qualities[quality] = qualities.get(quality, 0) + 1
        if match.get("type") != "none":
            matched += 1
        scores.append(float(outcome.get("dense_score") or 0.0))

        issue = str(outcome.get("primary_issue") or "")
        terminal_stage = str(outcome.get("terminal_stage") or "unknown")
        corrections = outcome.get("correction_targets")
        if not isinstance(corrections, list) or not corrections:
            if issue:
                corrections = [{"stage": terminal_stage, "issue": issue, "retry_count": 0, "failure_count": 0}]
            else:
                continue
        for correction in corrections:
            if not isinstance(correction, dict):
                continue
            stage = str(correction.get("stage") or terminal_stage or "unknown")
            local_issue = str(correction.get("issue") or issue or stage)
            key = (demo, stage, local_issue)
            bucket = buckets.setdefault(
                key,
                {
                    "demo_kind": demo,
                    "stage": stage,
                    "issue": local_issue,
                    "count": 0,
                    "retry_count": 0,
                    "failure_count": 0,
                    "run_ids": [],
                    "dense_scores": [],
                    "qualities": {},
                    "affected_outputs": set(),
                },
            )
            bucket["count"] += 1
            bucket["retry_count"] += int(correction.get("retry_count") or 0)
            bucket["failure_count"] += int(correction.get("failure_count") or 0)
            bucket["run_ids"].append(rollout.get("run_id"))
            bucket["dense_scores"].append(float(outcome.get("dense_score") or 0.0))
            bucket["qualities"][quality] = bucket["qualities"].get(quality, 0) + 1
            for name in correction.get("affected_outputs") or []:
                bucket["affected_outputs"].add(str(name))

    failure_modes = []
    for bucket in buckets.values():
        if int(bucket["count"]) < min_count:
            continue
        failure_modes.append(_finalize_bucket(bucket))
    failure_modes.sort(key=lambda item: (-item["count"], item["demo_kind"], item["stage"], item["issue"]))
    return {
        "schema": "task_skill_feedback_profile_v0",
        "controller_boundary": "frozen_sonic_low_level",
        "training_scope": "task_and_skill_policy_only",
        "summary": {
            "outcome_count": len(rows),
            "matched_policy_count": matched,
            "avg_dense_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "quality_counts": dict(sorted(qualities.items())),
            "demo_counts": dict(sorted(demo_counts.items())),
            "failure_mode_count": len(failure_modes),
        },
        "failure_modes": failure_modes,
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    issue = str(bucket["issue"])
    stage = str(bucket["stage"])
    rule = RULES.get(issue) or STAGE_DEFAULTS.get(stage) or STAGE_DEFAULTS["unknown"]
    affected = sorted(set(bucket["affected_outputs"]) | set(rule.get("affected_outputs") or []))
    scores = bucket["dense_scores"]
    return {
        "demo_kind": bucket["demo_kind"],
        "stage": stage,
        "issue": issue,
        "count": int(bucket["count"]),
        "retry_count": int(bucket["retry_count"]),
        "failure_count": int(bucket["failure_count"]),
        "avg_dense_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "qualities": dict(sorted(bucket["qualities"].items())),
        "affected_outputs": affected,
        "proposed_adjustments": rule.get("adjustments") or {},
        "note": rule.get("note") or "",
        "example_run_ids": [run_id for run_id in bucket["run_ids"][:8] if run_id],
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = _repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"input JSONL not found: {p}")
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"bad JSONL row at {p}:{line_no}")
            out.append(payload)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "demo_kind",
        "stage",
        "issue",
        "count",
        "retry_count",
        "failure_count",
        "avg_dense_score",
        "affected_outputs",
        "proposed_adjustments",
        "example_run_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "demo_kind": row.get("demo_kind"),
                    "stage": row.get("stage"),
                    "issue": row.get("issue"),
                    "count": row.get("count"),
                    "retry_count": row.get("retry_count"),
                    "failure_count": row.get("failure_count"),
                    "avg_dense_score": row.get("avg_dense_score"),
                    "affected_outputs": ",".join(row.get("affected_outputs") or []),
                    "proposed_adjustments": json.dumps(row.get("proposed_adjustments") or {}, sort_keys=True),
                    "example_run_ids": ",".join(str(item) for item in row.get("example_run_ids") or []),
                }
            )


def _markdown(profile: dict[str, Any]) -> str:
    lines = [
        f"# {profile['schema']}",
        "",
        f"- outcome_count: {profile['summary']['outcome_count']}",
        f"- matched_policy_count: {profile['summary']['matched_policy_count']}",
        f"- avg_dense_score: {profile['summary']['avg_dense_score']}",
        "",
        "| demo | stage | issue | count | avg_score | affected_outputs |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in profile["failure_modes"]:
        lines.append(
            "| {demo_kind} | {stage} | {issue} | {count} | {avg_dense_score:.2f} | {affected} |".format(
                affected=", ".join(row.get("affected_outputs") or []),
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _print_table(profile: dict[str, Any]) -> None:
    summary = profile["summary"]
    print(
        f"feedback_modes={summary['failure_mode_count']} outcomes={summary['outcome_count']} "
        f"matched={summary['matched_policy_count']} avg_score={summary['avg_dense_score']:.2f}"
    )
    print(f"{'demo':8s} {'stage':10s} {'count':>5s} {'score':>5s} {'issue':30s} affected_outputs")
    for row in profile["failure_modes"]:
        print(
            f"{str(row['demo_kind'])[:8]:8s} "
            f"{str(row['stage'])[:10]:10s} "
            f"{int(row['count']):5d} "
            f"{float(row['avg_dense_score']):5.2f} "
            f"{str(row['issue'])[:30]:30s} "
            f"{','.join(row.get('affected_outputs') or [])}"
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
