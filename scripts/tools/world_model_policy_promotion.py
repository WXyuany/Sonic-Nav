#!/usr/bin/env python3
"""Gate a learned world-model policy on physical episode evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate physical-policy promotion thresholds from episode JSONL logs.")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate episode JSONL file or directory; repeatable.")
    parser.add_argument("--baseline", action="append", default=[], help="Optional baseline episode JSONL file or directory.")
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument("--min-sequence-success-rate", type=float, default=0.20)
    parser.add_argument("--min-stage-effect-rate", type=float, default=0.60)
    parser.add_argument("--max-recovery-per-episode", type=float, default=3.0)
    parser.add_argument("--max-baseline-regression", type=float, default=0.05)
    parser.add_argument("--output", default="reports/leaderboards/policy_promotion_latest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = _metrics(_episodes(args.candidate))
    baseline = _metrics(_episodes(args.baseline)) if args.baseline else None
    checks = {
        "enough_episodes": candidate["full_sequence_episode_count"] >= max(1, int(args.min_episodes)),
        "sequence_success": candidate["sequence_success_rate"] >= float(args.min_sequence_success_rate),
        "stage_effect": candidate["stage_effect_rate"] >= float(args.min_stage_effect_rate),
        "recovery_budget": candidate["avg_recoveries"] <= float(args.max_recovery_per_episode),
    }
    if baseline is not None:
        checks["baseline_non_regression"] = candidate["sequence_success_rate"] + float(args.max_baseline_regression) >= baseline["sequence_success_rate"]
    report = {
        "schema": "sonic_world_model_policy_promotion_v0",
        "decision": "promote" if all(checks.values()) else "hold",
        "checks": checks,
        "thresholds": {
            "min_episodes": int(args.min_episodes), "min_sequence_success_rate": float(args.min_sequence_success_rate),
            "min_stage_effect_rate": float(args.min_stage_effect_rate), "max_recovery_per_episode": float(args.max_recovery_per_episode),
            "max_baseline_regression": float(args.max_baseline_regression),
        },
        "candidate": candidate,
        "baseline": baseline,
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"promotion={report['decision']} checks={sum(checks.values())}/{len(checks)}")
    print(_relative(output))
    return 0 if report["decision"] == "promote" else 2


def _episodes(inputs: list[str]) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for raw in inputs:
        path = _repo_path(raw)
        # Keep candidate evidence complete when a curriculum runner stores one
        # log per stage in nested trial directories.
        paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl")) if path.is_dir() else []
        for item in paths:
            events = []
            for line in item.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("schema") == "sonic_world_model_episode_event_v0":
                    events.append(event)
            if any(event.get("event") == "episode_terminal" for event in events):
                output.append(events)
    return output


def _metrics(episodes: list[list[dict[str, Any]]]) -> dict[str, Any]:
    sequence_success = full_sequence_count = 0
    effect_count = effect_success = recoveries = 0
    for events in episodes:
        terminal = next(event for event in reversed(events) if event.get("event") == "episode_terminal")
        is_full_sequence = str(terminal.get("episode_scope") or "") == "full_sequence"
        full_sequence_count += int(is_full_sequence)
        sequence_success += int(is_full_sequence and terminal.get("status") == "succeeded")
        recoveries += sum(event.get("event") == "recovery_status" for event in events)
        for event in events:
            evidence = event.get("effect_evidence") if event.get("event") == "primitive_status" else None
            if isinstance(evidence, dict):
                effect_count += 1
                effect_success += int(bool(evidence.get("passed")))
    count = len(episodes)
    return {
        "episode_count": count, "full_sequence_episode_count": full_sequence_count, "sequence_success_count": sequence_success,
        "sequence_success_rate": round(sequence_success / full_sequence_count, 4) if full_sequence_count else 0.0,
        "stage_effect_count": effect_count, "stage_effect_success_count": effect_success,
        "stage_effect_rate": round(effect_success / effect_count, 4) if effect_count else 0.0,
        "recovery_count": recoveries, "avg_recoveries": round(recoveries / count, 4) if count else 0.0,
    }


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
