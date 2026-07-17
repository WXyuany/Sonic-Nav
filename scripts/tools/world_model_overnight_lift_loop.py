#!/usr/bin/env python3
"""Overnight teacher-data, AWR, and non-assisted lift evaluation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "scripts" / "tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable overnight world-model lift training.")
    parser.add_argument("--output-dir", default="reports/overnight/lift_latest")
    parser.add_argument("--teacher-trials", type=int, default=80)
    parser.add_argument("--eval-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--checkpoint", default="reports/policy_models/world_model_hybrid_ppo_lift_aw_teacher_v1.pt")
    parser.add_argument("--train-epochs", type=int, default=160)
    parser.add_argument("--timeout-per-stage", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.teacher_trials < 1 or args.eval_trials < 1:
        raise SystemExit("teacher/eval trials must be positive")
    root = _path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    teacher = root / "teacher_collect"; eval_dir = root / "non_assisted_eval"
    dataset = root / "lift_teacher_physical.jsonl"; candidate = root / "world_model_hybrid_ppo_lift_overnight.pt"
    report = {"schema": "sonic_world_model_overnight_lift_v0", "started_at": time.time(), "root": _relative(root), "stages": {}}
    _write(root / "run.json", report)

    if not args.resume or not (teacher / "summary.json").is_file():
        _run("teacher_collect", [
            "/usr/bin/python3", str(TOOLS / "world_model_curriculum_batch.py"), "--sequence", "set_table_sequence", "--demo", "ball",
            "--stages", "1", "--trials-per-stage", str(args.teacher_trials), "--seed", str(args.seed), "--output-dir", str(teacher),
            "--policy-model", args.checkpoint, "--episode-start-settle", "5", "--episode-timeout-per-stage", str(args.timeout_per_stage),
            "--rollout-arg=--world-teacher-ball-attach", "--rollout-arg=--world-runtime-override-file",
            "--rollout-arg=configs/world_model/contact_profiles/stage1_center_contact_v1.json",
        ], report)
    report["stages"]["teacher_collect"] = _summary(teacher); _write(root / "run.json", report)

    _run("dataset", ["/usr/bin/python3", str(TOOLS / "world_model_episode_dataset.py"), "--input", str(teacher / "runs"), "--output", str(dataset)], report)
    report["stages"]["dataset"] = {"path": _relative(dataset), "bytes": dataset.stat().st_size if dataset.exists() else 0}; _write(root / "run.json", report)

    _run("awr", [
        "/home/wxy/miniconda3/envs/gr00t-wbc-sim/bin/python",
        str(TOOLS / "train_world_model_residual_offline.py"), "--dataset", str(dataset), "--checkpoint", args.checkpoint,
        "--output", str(candidate), "--skill", "manip.lift_object", "--component", "joint", "--epochs", str(args.train_epochs), "--min-samples", "8",
        "--positive-effect-only",
    ], report)
    report["stages"]["awr"] = {"path": _relative(candidate), "exists": candidate.is_file()}; _write(root / "run.json", report)

    if candidate.is_file():
        _run("non_assisted_eval", [
            "/usr/bin/python3", str(TOOLS / "world_model_curriculum_batch.py"), "--sequence", "set_table_sequence", "--demo", "ball",
            "--stages", "1", "--trials-per-stage", str(args.eval_trials), "--seed", str(args.seed + 100000), "--output-dir", str(eval_dir),
            "--policy-model", str(candidate), "--episode-start-settle", "5", "--episode-timeout-per-stage", str(args.timeout_per_stage),
            "--rollout-arg=--world-runtime-override-file", "--rollout-arg=configs/world_model/contact_profiles/stage1_center_contact_v1.json",
        ], report)
    report["stages"]["non_assisted_eval"] = _summary(eval_dir); report["finished_at"] = time.time(); _write(root / "run.json", report)
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


def _run(name: str, command: list[str], report: dict) -> None:
    print("[OVERNIGHT] " + name + ": " + " ".join(command), flush=True)
    started = time.time(); code = subprocess.run(command, cwd=REPO, check=False).returncode
    report["stages"][name] = {"return_code": int(code), "elapsed_s": round(time.time() - started, 3)}
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}")


def _summary(directory: Path) -> dict:
    path = directory / "summary.json"
    if not path.is_file(): return {"available": False}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError: return {"available": False, "error": "invalid summary JSON"}


def _path(value: str) -> Path:
    path = Path(value).expanduser(); return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try: return path.resolve().relative_to(REPO).as_posix()
    except ValueError: return str(path)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
