#!/usr/bin/env python3
"""Compare two policy checkpoints on an identical physical curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "scripts" / "tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched baseline/candidate physical curriculum trials.")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--sequence", default="set_table_sequence")
    parser.add_argument("--suite", default="configs/world_model/task_suites/sonic_general_v0.yaml")
    parser.add_argument("--demo", default="ball", choices=("ball", "box"))
    parser.add_argument("--stages", default="")
    parser.add_argument("--trials-per-stage", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--output-dir", default="reports/ab/latest")
    parser.add_argument("--min-stage-success-rate", type=float, default=0.60)
    parser.add_argument("--max-baseline-regression", type=float, default=0.05)
    parser.add_argument("--rollout-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _path(args.output_dir)
    shared_scene, shared_manifest = root / "shared" / "scenes" / "scene_sonic_episode_shared.xml", root / "shared" / "manifests" / "shared.json"
    materialize = [
        args.ros_python, str(TOOLS / "world_model_episode_materializer.py"), "--suite", args.suite, "--sequence", args.sequence,
        "--scene-output-dir", str(shared_scene.parent), "--manifest-output-dir", str(shared_manifest.parent), "--name", "shared", "--overwrite",
    ]
    print("[CURRICULUM_AB] shared materialization: " + " ".join(materialize))
    if not args.dry_run:
        code = subprocess.run(materialize, cwd=REPO, check=False).returncode
        if code:
            raise RuntimeError(f"shared episode materialization failed with code {code}")
    commands = {
        "baseline": _batch_command(args, root / "baseline", args.baseline_model, shared_scene, shared_manifest),
        "candidate": _batch_command(args, root / "candidate", args.candidate_model, shared_scene, shared_manifest),
    }
    for name, command in commands.items():
        print(f"[CURRICULUM_AB] {name}: " + " ".join(command))
        if not args.dry_run:
            code = subprocess.run(command, cwd=REPO, check=False).returncode
            if code:
                raise RuntimeError(f"{name} curriculum batch failed with code {code}")
    if args.dry_run:
        return 0
    baseline = _read(root / "baseline" / "summary.json")
    candidate = _read(root / "candidate" / "summary.json")
    report = compare_curricula(baseline, candidate, min_stage_success_rate=float(args.min_stage_success_rate), max_baseline_regression=float(args.max_baseline_regression))
    report["schema"] = "sonic_world_model_curriculum_ab_v0"
    report["evidence"] = {
        "baseline_summary": _relative(root / "baseline" / "summary.json"),
        "candidate_summary": _relative(root / "candidate" / "summary.json"),
        "matched_seed_schedule": baseline.get("config", {}).get("base_seed") == candidate.get("config", {}).get("base_seed"),
        "matched_manifest": baseline.get("config", {}).get("manifest_sha256") == candidate.get("config", {}).get("manifest_sha256"),
        "matched_scene": baseline.get("config", {}).get("scene_sha256") == candidate.get("config", {}).get("scene_sha256"),
    }
    output = root / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"curriculum_ab={report['decision']} candidate={report['candidate_stage_success_rate']:.1%} baseline={report['baseline_stage_success_rate']:.1%}")
    print(_relative(output))
    return 0 if report["decision"] == "advance_to_sequence_eval" else 2


def compare_curricula(baseline: dict[str, Any], candidate: dict[str, Any], *, min_stage_success_rate: float, max_baseline_regression: float) -> dict[str, Any]:
    baseline_summary = baseline.get("summary") if isinstance(baseline.get("summary"), dict) else {}
    candidate_summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
    base_trials, candidate_trials = int(baseline_summary.get("trial_count") or 0), int(candidate_summary.get("trial_count") or 0)
    base_rate = float(baseline_summary.get("stage_success_rate") or 0.0)
    candidate_rate = float(candidate_summary.get("stage_success_rate") or 0.0)
    checks = {
        "matched_trial_count": base_trials > 0 and base_trials == candidate_trials,
        "candidate_stage_floor": candidate_rate >= min_stage_success_rate,
        "baseline_non_regression": candidate_rate + max_baseline_regression >= base_rate,
    }
    return {
        "decision": "advance_to_sequence_eval" if all(checks.values()) else "hold",
        "checks": checks,
        "baseline_trial_count": base_trials,
        "candidate_trial_count": candidate_trials,
        "baseline_stage_success_rate": base_rate,
        "candidate_stage_success_rate": candidate_rate,
        "absolute_stage_success_delta": round(candidate_rate - base_rate, 4),
        "thresholds": {"min_stage_success_rate": min_stage_success_rate, "max_baseline_regression": max_baseline_regression},
    }


def _batch_command(args: argparse.Namespace, output: Path, model: str, scene: Path, manifest: Path) -> list[str]:
    command = [
        args.ros_python, str(TOOLS / "world_model_curriculum_batch.py"), "--sequence", args.sequence, "--suite", args.suite,
        "--demo", args.demo, "--output-dir", str(output), "--stages", args.stages,
        "--trials-per-stage", str(int(args.trials_per_stage)), "--seed", str(int(args.seed)), "--ros-python", args.ros_python,
        "--policy-backend", "learned", "--policy-model", str(model), "--scene", str(scene), "--manifest", str(manifest),
    ]
    for item in args.rollout_arg:
        command.extend(["--rollout-arg", str(item)])
    return command


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
