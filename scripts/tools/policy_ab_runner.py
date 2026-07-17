#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent


DEFAULT_BASELINE_POLICY = "reports/policy_data/sonic_general_v0_heuristic.jsonl"
DEFAULT_ADJUSTED_POLICY = "reports/policy_data/sonic_general_v0_feedback_adjusted.jsonl"
DEFAULT_REPORT_PATH = "reports/rollouts"
DEFAULT_OUTPUT_DIR = "reports/policy_ab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-vs-feedback-adjusted task/skill policy rollouts and compare outcomes. "
            "This evaluates high-level policy parameters only; SONIC low-level control remains frozen."
        )
    )
    parser.add_argument("demo", choices=["ball"], default="ball", nargs="?")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tag", default=None, help="Experiment tag. Defaults to timestamp.")
    parser.add_argument("--policy-task-id", default="ball_left_to_tray")
    parser.add_argument("--baseline-policy", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--adjusted-policy", default=DEFAULT_ADJUSTED_POLICY)
    parser.add_argument("--adjusted-apply", choices=["safe", "full"], default="safe")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--camera", action="store_true", default=False)
    parser.add_argument("--reset-each-rollout", action="store_true", default=True)
    parser.add_argument("--no-reset-each-rollout", dest="reset_each_rollout", action="store_false")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-rollout-fail", action="store_true")
    parser.add_argument(
        "--baseline-policy-json",
        action="store_true",
        help="Also pass the baseline teacher JSONL to the demo. Default baseline uses old demo behavior.",
    )
    args, demo_args = parser.parse_known_args()
    args.demo_args = _clean_extra_args(demo_args)
    return args


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    output_dir = _repo_path(args.output_dir) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "baseline",
            "policy": _repo_path(args.baseline_policy),
            "prefix": f"ab_{tag}_baseline",
            "demo_args": _baseline_demo_args(args),
        },
        {
            "name": "adjusted",
            "policy": _repo_path(args.adjusted_policy),
            "prefix": f"ab_{tag}_adjusted",
            "demo_args": [
                "--policy-action-json",
                str(_repo_path(args.adjusted_policy)),
                "--policy-action-task-id",
                str(args.policy_task_id),
                "--policy-action-apply",
                str(args.adjusted_apply),
            ],
        },
    ]

    results: list[dict[str, Any]] = []
    for variant in variants:
        code = _run_variant(args, variant)
        outcome_jsonl = output_dir / f"{variant['name']}_outcomes.jsonl"
        outcome_csv = output_dir / f"{variant['name']}_outcomes.csv"
        join_code = 0
        if not args.dry_run:
            join_code = _join_variant_outcomes(args, variant, outcome_jsonl, outcome_csv)
        metrics = _variant_metrics(outcome_csv) if outcome_csv.exists() else {"rollouts": 0}
        results.append(
            {
                "variant": variant["name"],
                "prefix": variant["prefix"],
                "policy": _rel(variant["policy"]),
                "rollout_exit_code": code,
                "join_exit_code": join_code,
                **metrics,
            }
        )

    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "summary.json"
    if not args.dry_run:
        _write_summary(summary_path, results)
        report_path.write_text(json.dumps({"tag": tag, "variants": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(results, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nWrote A/B summary: {_rel(summary_path)}")
        print(f"Wrote A/B report: {_rel(report_path)}")
    if any(int(row.get("rollout_exit_code") or 0) != 0 for row in results):
        return 1
    return 0


def _baseline_demo_args(args: argparse.Namespace) -> list[str]:
    if not args.baseline_policy_json:
        return []
    return [
        "--policy-action-json",
        str(_repo_path(args.baseline_policy)),
        "--policy-action-task-id",
        str(args.policy_task_id),
        "--policy-action-apply",
        "safe",
    ]


def _run_variant(args: argparse.Namespace, variant: dict[str, Any]) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rollout_batch.py"),
        args.demo,
        "--runs",
        str(args.runs),
        "--prefix",
        str(variant["prefix"]),
        "--start-index",
        "1",
        "--width",
        "3",
        "--report-path",
        str(args.report_path),
        "--python",
        str(args.python),
    ]
    if args.headless:
        cmd.append("--headless")
    if not args.camera:
        cmd.append("--no-camera")
    if args.reset_each_rollout:
        cmd.append("--reset-each-rollout")
    if args.fail_on_rollout_fail:
        cmd.append("--fail-on-rollout-fail")
    if args.dry_run:
        cmd.append("--dry-run")
    extra = _clean_extra_args(args.demo_args)
    if variant["demo_args"] or extra:
        cmd.append("--")
        cmd.extend(variant["demo_args"])
        cmd.extend(extra)
    print()
    print("=" * 72)
    print(f"[AB] variant={variant['name']} prefix={variant['prefix']}")
    print("[AB] " + " ".join(_shell_quote(part) for part in cmd))
    print("=" * 72)
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO)


def _join_variant_outcomes(
    args: argparse.Namespace,
    variant: dict[str, Any],
    output_jsonl: Path,
    output_csv: Path,
) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "policy_outcome_joiner.py"),
        "--policy-jsonl",
        str(variant["policy"]),
        "--rollouts",
        str(args.report_path),
        "--run-id-prefix",
        str(variant["prefix"]),
        "--output",
        str(output_jsonl),
        "--summary",
        str(output_csv),
    ]
    print()
    print(f"[AB] Joining outcomes for {variant['name']}...")
    return subprocess.call(cmd, cwd=REPO)


def _variant_metrics(csv_path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rollouts": 0}
    scores = [float(row.get("dense_score") or 0.0) for row in rows]
    successes = [row for row in rows if row.get("final_status") == "success"]
    clean = [row for row in rows if row.get("quality") == "clean_success"]
    rough = [row for row in rows if row.get("quality") in {"rough_success", "poor_success"}]
    failures = [row for row in rows if row.get("final_status") == "failed"]
    retries = [int(row.get("retry_count") or 0) for row in rows]
    issues: dict[str, int] = {}
    for row in rows:
        issue = row.get("primary_issue") or "-"
        issues[issue] = issues.get(issue, 0) + 1
    return {
        "rollouts": len(rows),
        "success": len(successes),
        "clean_success": len(clean),
        "rough_success": len(rough),
        "failed": len(failures),
        "avg_dense_score": round(sum(scores) / len(scores), 4),
        "avg_retry_count": round(sum(retries) / len(retries), 3),
        "top_issues": ";".join(f"{k}:{v}" for k, v in sorted(issues.items(), key=lambda item: (-item[1], item[0]))[:5]),
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "prefix",
        "policy",
        "rollouts",
        "success",
        "clean_success",
        "rough_success",
        "failed",
        "avg_dense_score",
        "avg_retry_count",
        "top_issues",
        "rollout_exit_code",
        "join_exit_code",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_summary(rows: list[dict[str, Any]], *, dry_run: bool) -> None:
    prefix = "DRY-RUN " if dry_run else ""
    print()
    print(f"{prefix}policy_ab_variants={len(rows)}")
    print(f"{'variant':10s} {'runs':>4s} {'succ':>4s} {'clean':>5s} {'fail':>4s} {'score':>6s} {'retry':>6s} issues")
    for row in rows:
        print(
            f"{str(row.get('variant') or '-')[:10]:10s} "
            f"{int(row.get('rollouts') or 0):4d} "
            f"{int(row.get('success') or 0):4d} "
            f"{int(row.get('clean_success') or 0):5d} "
            f"{int(row.get('failed') or 0):4d} "
            f"{float(row.get('avg_dense_score') or 0.0):6.3f} "
            f"{float(row.get('avg_retry_count') or 0.0):6.2f} "
            f"{row.get('top_issues') or '-'}"
        )


def _clean_extra_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _shell_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@")
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
