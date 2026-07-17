#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
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


DEFAULT_POLICY = "reports/policy_data/sonic_general_v0_heuristic.jsonl"
DEFAULT_FEEDBACK = "reports/policy_outcomes/sonic_feedback_profile.json"
DEFAULT_OUTPUT = "reports/policy_data/sonic_general_v0_feedback_adjusted.jsonl"
SKIP_ACTION_ISSUES = {"missing_or_implausible_anchor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply rollout feedback profile to high-level policy samples. "
            "This adjusts task/skill parameters only; SONIC low-level control remains frozen."
        )
    )
    parser.add_argument("--policy-jsonl", default=DEFAULT_POLICY)
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", help="CSV summary path. Defaults to <output>.csv.")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = _read_jsonl(args.policy_jsonl)
    feedback = json.loads(_repo_path(args.feedback).read_text(encoding="utf-8"))
    adjusted: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = apply_feedback(
            sample,
            feedback,
            min_count=max(1, int(args.min_count)),
            max_modes=max(1, int(args.max_modes)),
        )
        adjusted.append(row)
        summary_rows.append(_summary_row(sample, row))

    output = _repo_path(args.output)
    summary_path = _repo_path(args.summary) if args.summary else output.with_suffix(".csv")
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for sample in adjusted:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        _write_csv(summary_path, summary_rows)

    _print_table(summary_rows, dry_run=bool(args.dry_run))
    if not args.dry_run:
        print(f"\nWrote adjusted policy JSONL: {_rel(output)}")
        print(f"Wrote adjusted policy summary: {_rel(summary_path)}")
    return 0


def apply_feedback(
    sample: dict[str, Any],
    feedback: dict[str, Any],
    *,
    min_count: int,
    max_modes: int,
) -> dict[str, Any]:
    out = copy.deepcopy(sample)
    action = out.get("action") if isinstance(out.get("action"), dict) else {}
    demo_kind = str(((action.get("task_intent") or {}).get("demo_kind")) or "")
    modes = _select_modes(feedback, demo_kind=demo_kind, min_count=min_count, max_modes=max_modes)
    applied: list[dict[str, Any]] = []
    for mode in modes:
        adjustments = mode.get("proposed_adjustments") if isinstance(mode.get("proposed_adjustments"), dict) else {}
        if not adjustments:
            continue
        applied.append(_apply_adjustments(action, adjustments, mode))
    metadata = action.setdefault("metadata", {})
    metadata["feedback_policy"] = {
        "source": "task_skill_feedback_profile_v0",
        "applied_mode_count": len([item for item in applied if item]),
        "applied_modes": [item for item in applied if item],
        "skipped_issues": sorted(SKIP_ACTION_ISSUES),
        "controller_boundary": "frozen_sonic_low_level",
    }
    action["policy_id"] = f"{action.get('policy_id', 'policy')}_feedback_adjusted"
    out["metadata"] = {
        **dict(out.get("metadata") or {}),
        "feedback_adjusted": True,
        "feedback_source": feedback.get("schema", "unknown"),
    }
    return out


def _select_modes(
    feedback: dict[str, Any],
    *,
    demo_kind: str,
    min_count: int,
    max_modes: int,
) -> list[dict[str, Any]]:
    modes = []
    for mode in feedback.get("failure_modes") or []:
        if not isinstance(mode, dict):
            continue
        if str(mode.get("demo_kind") or "") != demo_kind:
            continue
        if str(mode.get("issue") or "") in SKIP_ACTION_ISSUES:
            continue
        if int(mode.get("count") or 0) < min_count:
            continue
        modes.append(mode)
    modes.sort(key=lambda item: (-int(item.get("count") or 0), float(item.get("avg_dense_score") or 0.0), str(item.get("issue") or "")))
    unique: list[dict[str, Any]] = []
    seen_issues: set[str] = set()
    for mode in modes:
        issue = str(mode.get("issue") or "")
        if issue in seen_issues:
            continue
        seen_issues.add(issue)
        unique.append(mode)
        if len(unique) >= max_modes:
            break
    return unique


def _apply_adjustments(
    action: dict[str, Any],
    adjustments: dict[str, Any],
    mode: dict[str, Any],
) -> dict[str, Any]:
    applied: dict[str, Any] = {
        "demo_kind": mode.get("demo_kind"),
        "stage": mode.get("stage"),
        "issue": mode.get("issue"),
        "count": mode.get("count"),
        "avg_dense_score": mode.get("avg_dense_score"),
        "fields": [],
    }
    base_goal = action.get("base_goal") if isinstance(action.get("base_goal"), dict) else None
    if base_goal is not None and "standoff_delta_m" in adjustments:
        old = _finite(base_goal.get("standoff"), 0.0)
        base_goal["standoff"] = round(max(0.32, old + float(adjustments["standoff_delta_m"])), 4)
        applied["fields"].append({"field": "base_goal.standoff", "old": old, "new": base_goal["standoff"]})

    hand = action.get("hand_pose_target") if isinstance(action.get("hand_pose_target"), dict) else None
    if hand is not None:
        if "contact_x_delta_m" in adjustments:
            _shift_hand_point(hand, "contact", "palm", axis=0, delta=float(adjustments["contact_x_delta_m"]), applied=applied)
            _shift_hand_point(hand, "hold", "palm", axis=0, delta=float(adjustments["contact_x_delta_m"]) * 0.5, applied=applied)
        if "contact_z_delta_m" in adjustments:
            _shift_hand_point(hand, "contact", "palm", axis=2, delta=float(adjustments["contact_z_delta_m"]), applied=applied)
            _shift_hand_point(hand, "pregrasp", "palm", axis=2, delta=float(adjustments["contact_z_delta_m"]) * 0.5, applied=applied)
        if "palm_x_delta_m" in adjustments:
            _shift_hand_point(hand, "contact", "palm", axis=0, delta=float(adjustments["palm_x_delta_m"]), applied=applied)
        if "palm_z_delta_m" in adjustments:
            _shift_hand_point(hand, "contact", "palm", axis=2, delta=float(adjustments["palm_z_delta_m"]), applied=applied)

    wrist = action.get("wrist_target") if isinstance(action.get("wrist_target"), dict) else None
    if wrist is not None and "wrist_pitch_delta_rad" in adjustments and "pitch" in wrist:
        old = _finite(wrist.get("pitch"), 0.0)
        wrist["pitch"] = round(old + float(adjustments["wrist_pitch_delta_rad"]), 4)
        applied["fields"].append({"field": "wrist_target.pitch", "old": old, "new": wrist["pitch"]})

    close = action.get("grasp_close_ratio") if isinstance(action.get("grasp_close_ratio"), dict) else None
    if close is not None:
        if "close_ratio_delta" in adjustments and "close_ratio" in close:
            old = _finite(close.get("close_ratio"), 0.0)
            close["close_ratio"] = round(_clamp(old + float(adjustments["close_ratio_delta"]), 0.05, 0.98), 4)
            applied["fields"].append({"field": "grasp_close_ratio.close_ratio", "old": old, "new": close["close_ratio"]})
        if "secure_aperture_delta_m" in adjustments and "secure_aperture" in close:
            old = _finite(close.get("secure_aperture"), 0.0)
            close["secure_aperture"] = round(max(0.018, old + float(adjustments["secure_aperture_delta_m"])), 4)
            applied["fields"].append({"field": "grasp_close_ratio.secure_aperture", "old": old, "new": close["secure_aperture"]})

    lift = action.get("lift_place_targets") if isinstance(action.get("lift_place_targets"), dict) else None
    if lift is not None and "lift_z_delta_m" in adjustments:
        for key in ("lift_pose_base", "carry_pose_base"):
            if isinstance(lift.get(key), list) and len(lift[key]) >= 3:
                old = _finite(lift[key][2], 0.0)
                lift[key][2] = round(old + float(adjustments["lift_z_delta_m"]), 4)
                applied["fields"].append({"field": f"lift_place_targets.{key}.z", "old": old, "new": lift[key][2]})

    recovery = action.get("recovery_decision") if isinstance(action.get("recovery_decision"), dict) else {}
    recovery.setdefault("profile_hints", [])
    recovery["profile_hints"].append(
        {
            "issue": mode.get("issue"),
            "stage": mode.get("stage"),
            "adjustments": adjustments,
        }
    )
    action["recovery_decision"] = recovery
    return applied


def _shift_hand_point(
    hand: dict[str, Any],
    phase: str,
    key: str,
    *,
    axis: int,
    delta: float,
    applied: dict[str, Any],
) -> None:
    payload = hand.get(phase)
    if not isinstance(payload, dict):
        return
    point = payload.get(key)
    if not isinstance(point, list) or len(point) < 3:
        return
    old = _finite(point[axis], 0.0)
    point[axis] = round(old + delta, 4)
    applied["fields"].append({"field": f"hand_pose_target.{phase}.{key}.{axis}", "old": old, "new": point[axis]})


def _summary_row(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    action_before = before.get("action") if isinstance(before.get("action"), dict) else {}
    action_after = after.get("action") if isinstance(after.get("action"), dict) else {}
    feedback = ((action_after.get("metadata") or {}).get("feedback_policy") or {})
    fields = []
    for mode in feedback.get("applied_modes") or []:
        fields.extend(item.get("field") for item in mode.get("fields") or [] if item.get("field"))
    return {
        "sample_id": after.get("sample_id"),
        "task_id": action_after.get("task_id"),
        "demo_kind": ((action_after.get("task_intent") or {}).get("demo_kind")),
        "policy_before": action_before.get("policy_id"),
        "policy_after": action_after.get("policy_id"),
        "applied_mode_count": feedback.get("applied_mode_count", 0),
        "applied_fields": ",".join(str(field) for field in fields),
    }


def _print_table(rows: list[dict[str, Any]], *, dry_run: bool) -> None:
    prefix = "DRY-RUN " if dry_run else ""
    print(f"{prefix}feedback_adjusted_samples={len(rows)}")
    print(f"{'task_id':32s} {'kind':8s} {'modes':>5s} fields")
    for row in rows:
        print(
            f"{str(row.get('task_id') or '-')[:32]:32s} "
            f"{str(row.get('demo_kind') or '-')[:8]:8s} "
            f"{int(row.get('applied_mode_count') or 0):5d} "
            f"{row.get('applied_fields') or '-'}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sample_id", "task_id", "demo_kind", "policy_before", "policy_after", "applied_mode_count", "applied_fields"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"bad JSONL row at {p}:{line_no}")
            out.append(payload)
    return out


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
