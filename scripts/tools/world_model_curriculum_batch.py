#!/usr/bin/env python3
"""Run reproducible, stage-isolated physical curriculum trials.

Each trial uses the normal executor-backed rollout stack, but restores the
objects for exactly one manifest stage before execution.  This makes failed
trials useful training data without pretending that they are carry-state
sequence successes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "scripts" / "tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect reproducible stage-isolated physical curriculum evidence.")
    parser.add_argument("--sequence", default="set_table_sequence")
    parser.add_argument("--suite", default="configs/world_model/task_suites/sonic_general_v0.yaml")
    parser.add_argument("--demo", default="ball", choices=("ball", "box"))
    parser.add_argument("--scene", default="", help="Reuse a pre-materialized carry-state scene; requires --manifest.")
    parser.add_argument("--manifest", default="", help="Reuse a pre-materialized episode manifest; requires --scene.")
    parser.add_argument("--output-dir", default="reports/curriculum/latest")
    parser.add_argument("--name", default="", help="Artifact tag; defaults to sequence.")
    parser.add_argument("--stages", default="", help="Comma-separated manifest stage indices; default is all stages.")
    parser.add_argument("--trials-per-stage", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--episode-start-settle", type=float, default=5.0)
    parser.add_argument("--episode-timeout-per-stage", type=float, default=120.0)
    parser.add_argument("--policy-backend", default="learned", choices=("heuristic", "learned", "memory"))
    parser.add_argument("--policy-model", default="reports/policy_models/world_model_hybrid_ppo_physical_aw_v0.pt")
    parser.add_argument("--rollout-arg", action="append", default=[], help="Extra rollout_batch.py option; repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.trials_per_stage) < 1:
        raise SystemExit("--trials-per-stage must be positive")
    if bool(args.scene) != bool(args.manifest):
        raise SystemExit("--scene and --manifest must be supplied together")
    root = _path(args.output_dir)
    artifact = _safe_name(args.name or args.sequence)
    if args.scene:
        scene, manifest = _path(args.scene), _path(args.manifest)
        if not scene.is_file() or not manifest.is_file():
            raise SystemExit(f"shared scene/manifest must exist: scene={scene.is_file()} manifest={manifest.is_file()}")
        materialization = "shared_input"
    else:
        materialized = root / "materialized"
        scene_dir, manifest_dir = materialized / "scenes", materialized / "manifests"
        manifest = manifest_dir / f"{artifact}.json"
        scene = scene_dir / f"scene_sonic_episode_{artifact}.xml"
        materialize_cmd = [
            args.ros_python, str(TOOLS / "world_model_episode_materializer.py"), "--suite", str(args.suite),
            "--sequence", str(args.sequence), "--scene-output-dir", str(scene_dir), "--manifest-output-dir", str(manifest_dir),
            "--name", artifact, "--overwrite",
        ]
        if args.dry_run:
            _print("materialize", materialize_cmd)
            return 0
        _run(materialize_cmd)
        materialization = "generated"
    if not manifest.is_file() or not scene.is_file():
        raise RuntimeError("episode materializer did not create scene and manifest")
    payload = _json(manifest)
    stages = _stage_indices(payload, args.stages)
    config = {
        "schema": "sonic_world_model_curriculum_batch_v0",
        "sequence": args.sequence,
        "demo": args.demo,
        "stages": stages,
        "trials_per_stage": int(args.trials_per_stage),
        "base_seed": int(args.seed),
        "policy_backend": args.policy_backend,
        "policy_model": str(args.policy_model),
        "scene": _relative(scene),
        "manifest": _relative(manifest),
        "manifest_sha256": _sha256(manifest),
        "scene_sha256": _sha256(scene),
        "materialization": materialization,
        "created_at": time.time(),
    }
    _write(root / "batch_config.json", config)
    runs: list[dict[str, Any]] = []
    for stage in stages:
        for trial in range(1, int(args.trials_per_stage) + 1):
            seed = int(args.seed) + stage * 1000 + trial
            output = root / "runs" / f"stage_{stage:02d}" / f"trial_{trial:03d}.jsonl"
            command = [
                args.ros_python, str(TOOLS / "rollout_batch.py"), args.demo,
                "--scene", str(scene), "--episode-manifest", str(manifest), "--episode-output-jsonl", str(output),
                "--episode-curriculum-stage", str(stage), "--episode-stage-start", str(stage), "--episode-stage-stop", str(stage),
                "--episode-start-settle", str(float(args.episode_start_settle)), "--episode-timeout-per-stage", str(float(args.episode_timeout_per_stage)),
                "--world-policy-backend", str(args.policy_backend), "--world-policy-model", str(args.policy_model),
                "--headless", "--no-report", "--fail-on-rollout-fail", "--prefix", f"curriculum_{artifact}_s{stage:02d}_t{trial:03d}",
            ]
            command.extend(str(item) for item in args.rollout_arg)
            started = time.time()
            _print(f"stage={stage} trial={trial} seed={seed}", command)
            code = _run(command, check=False)
            terminal = _terminal(output)
            liveness_error = "missing_episode_terminal" if terminal is None else None
            runs.append({
                "stage_index": stage, "trial_index": trial, "seed": seed, "return_code": code,
                "source_log": _relative(output), "elapsed_s": round(time.time() - started, 3), "terminal": terminal,
                "liveness_error": liveness_error,
            })
            _write(root / "runs.json", {"schema": "sonic_world_model_curriculum_runs_v0", "config": config, "runs": runs})
    dataset = root / "policy_data" / "transitions.jsonl"
    if not args.skip_dataset:
        _run([args.ros_python, str(TOOLS / "world_model_episode_dataset.py"), "--input", str(root / "runs"), "--output", str(dataset)], check=False)
    summary = _summary(runs)
    report = {"schema": "sonic_world_model_curriculum_batch_report_v0", "config": config, "summary": summary, "runs": runs, "dataset": _relative(dataset) if dataset.exists() else None}
    _write(root / "summary.json", report)
    print(json.dumps(summary, sort_keys=True))
    print(_relative(root / "summary.json"))
    return 0


def _stage_indices(manifest: dict[str, Any], raw: str) -> list[int]:
    available = [int(item.get("stage_index") or 0) for item in manifest.get("stages", []) if isinstance(item, dict) and int(item.get("stage_index") or 0) > 0]
    requested = [int(item.strip()) for item in raw.split(",") if item.strip()] if raw else available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"requested stages are absent from manifest: {missing}; available={available}")
    return list(dict.fromkeys(requested))


def _terminal(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("schema") == "sonic_world_model_episode_event_v0" and event.get("event") == "episode_terminal":
            rows.append(event)
    return rows[-1] if rows else None


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [row for row in runs if isinstance(row.get("terminal"), dict)]
    succeeded = [row for row in terminal if str((row.get("terminal") or {}).get("status")) == "succeeded"]
    by_stage: dict[str, dict[str, int]] = {}
    for row in runs:
        item = by_stage.setdefault(str(row["stage_index"]), {"trials": 0, "terminal": 0, "succeeded": 0})
        item["trials"] += 1
        item["terminal"] += int(isinstance(row.get("terminal"), dict))
        item["succeeded"] += int(str((row.get("terminal") or {}).get("status")) == "succeeded")
    return {"trial_count": len(runs), "terminal_count": len(terminal), "stage_success_count": len(succeeded), "stage_success_rate": round(len(succeeded) / len(runs), 4) if runs else 0.0, "by_stage": by_stage}


def _run(command: list[str], *, check: bool = True) -> int:
    result = subprocess.run(command, cwd=REPO, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return int(result.returncode)


def _print(label: str, command: list[str]) -> None:
    print(f"[CURRICULUM] {label}: " + " ".join(command))


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
