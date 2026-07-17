#!/usr/bin/env python3
"""Run reproducible, fixed-scene contact parameter scans for physical skills.

The sweep deliberately evaluates stage-isolated rollouts.  It changes only the
side-grasp contact pose/closure overrides and records the primitive evidence,
so a winning setting can be promoted into a policy-data collection run without
claiming that a full carry-state sequence succeeded.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "scripts" / "tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan side-grasp contact parameters on one physical curriculum stage.")
    parser.add_argument("--sequence", default="set_table_sequence")
    parser.add_argument("--suite", default="configs/world_model/task_suites/sonic_general_v0.yaml")
    parser.add_argument("--demo", default="ball", choices=("ball", "box"))
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/contact_sweeps/latest")
    parser.add_argument("--close-ratios", default="0.55,0.63,0.71")
    parser.add_argument("--contact-x-deltas", default="-0.008,0.0,0.008")
    parser.add_argument("--contact-z-deltas", default="-0.012,-0.006,0.0")
    parser.add_argument("--grasp-wrist-pitches", default="-0.05")
    parser.add_argument("--trials-per-config", type=int, default=1)
    parser.add_argument("--max-configs", type=int, default=0, help="Run at most this many configs (0 means all).")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--episode-start-settle", type=float, default=5.0)
    parser.add_argument("--episode-timeout-per-stage", type=float, default=180.0)
    parser.add_argument("--policy-backend", default="learned", choices=("heuristic", "learned", "memory"))
    parser.add_argument("--policy-model", default="reports/policy_models/world_model_hybrid_ppo_curriculum_stage1_v4.pt")
    parser.add_argument("--lift-hold-close-ratio", type=float, default=0.88)
    parser.add_argument("--lift-squeeze-close-ratio", type=float, default=0.81)
    parser.add_argument("--lift-z-lead", type=float, default=0.032)
    parser.add_argument("--lift-z", type=float, default=0.18)
    parser.add_argument("--lift-duration", type=float, default=1.6)
    parser.add_argument("--low-hold-duration", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials_per_config < 1:
        raise SystemExit("--trials-per-config must be positive")
    root = _path(args.output_dir)
    scene, manifest = _materialize(args, root)
    configs = _grid(args.close_ratios, args.contact_x_deltas, args.contact_z_deltas, args.grasp_wrist_pitches)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    if not configs:
        raise SystemExit("the contact grid is empty")

    results: list[dict[str, Any]] = []
    for index, override in enumerate(configs, start=1):
        name = (
            f"contact_{index:02d}_c{override['close_ratio']:.3f}_x{override['contact_x_delta_m']:+.3f}"
            f"_z{override['contact_z_delta_m']:+.3f}_p{override['grasp_wrist_pitch']:+.3f}"
        )
        config_path = root / "configs" / f"{name}.json"
        _write(config_path, _override_payload(override, args))
        output = root / "runs" / name
        command = [
            args.ros_python, str(TOOLS / "world_model_curriculum_batch.py"),
            "--sequence", args.sequence, "--suite", args.suite, "--demo", args.demo,
            "--scene", str(scene), "--manifest", str(manifest), "--stages", str(args.stage),
            "--trials-per-stage", str(args.trials_per_config),
            "--seed", str(args.seed + index * 100), "--output-dir", str(output),
            "--name", name, "--ros-python", args.ros_python,
            "--episode-start-settle", str(args.episode_start_settle),
            "--episode-timeout-per-stage", str(args.episode_timeout_per_stage),
            "--policy-backend", args.policy_backend, "--policy-model", args.policy_model,
            "--rollout-arg=--world-runtime-override-file", f"--rollout-arg={config_path}",
        ]
        started = time.time()
        if args.dry_run:
            print("[CONTACT-SWEEP] " + " ".join(command))
            code = 0
        else:
            print(f"[CONTACT-SWEEP] {index}/{len(configs)} {name}: " + " ".join(command))
            code = subprocess.run(command, cwd=REPO, check=False).returncode
        evidence = _collect_evidence(output / "runs")
        result = {
            "config_index": index,
            "name": name,
            "override": override,
            "override_file": _relative(config_path),
            "return_code": int(code),
            "elapsed_s": round(time.time() - started, 3),
            **evidence,
        }
        results.append(result)
        _write(root / "sweep.json", _report(args, scene, manifest, results))
    report = _report(args, scene, manifest, results)
    _write(root / "sweep.json", report)
    print(json.dumps(report["summary"], sort_keys=True))
    print(_relative(root / "sweep.json"))
    return 0


def _materialize(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    scene_dir, manifest_dir = root / "materialized" / "scenes", root / "materialized" / "manifests"
    scene = scene_dir / f"scene_sonic_episode_{args.sequence}.xml"
    manifest = manifest_dir / f"{args.sequence}.json"
    command = [
        args.ros_python, str(TOOLS / "world_model_episode_materializer.py"),
        "--suite", args.suite, "--sequence", args.sequence,
        "--scene-output-dir", str(scene_dir), "--manifest-output-dir", str(manifest_dir),
        "--name", args.sequence, "--overwrite",
    ]
    if args.dry_run:
        print("[CONTACT-SWEEP] materialize: " + " ".join(command))
        return scene, manifest
    if subprocess.run(command, cwd=REPO, check=False).returncode:
        raise RuntimeError("episode materialization failed")
    if not scene.is_file() or not manifest.is_file():
        raise RuntimeError("episode materializer did not create the shared scene and manifest")
    return scene, manifest


def _grid(close_ratios: str, x_deltas: str, z_deltas: str, wrist_pitches: str = "-0.05") -> list[dict[str, float]]:
    values = [
        _float_list(close_ratios, "close ratios"),
        _float_list(x_deltas, "contact x deltas"),
        _float_list(z_deltas, "contact z deltas"),
        _float_list(wrist_pitches, "grasp wrist pitches"),
    ]
    grid = []
    for close, x_delta, z_delta, wrist_pitch in itertools.product(*values):
        grid.append({
            "close_ratio": round(_bounded(close, 0.20, 0.95), 5),
            "contact_x_delta_m": round(_bounded(x_delta, -0.025, 0.025), 5),
            "contact_z_delta_m": round(_bounded(z_delta, -0.015, 0.015), 5),
            "grasp_wrist_pitch": round(_bounded(wrist_pitch, -0.45, 0.20), 5),
        })
    return grid


def _override_payload(override: dict[str, float], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    return {
        "manip.side_grasp": dict(override),
        "manip.lift_object": {
            "hold_close_ratio": round(_bounded(args.lift_hold_close_ratio, 0.20, 0.98), 5),
            "squeeze_close_ratio": round(_bounded(args.lift_squeeze_close_ratio, 0.20, 0.98), 5),
            "servo_lift_z_lead": round(_bounded(args.lift_z_lead, 0.01, 0.10), 5),
            "lift_z": round(_bounded(args.lift_z, 0.10, 0.30), 5),
            "lift_duration": round(_bounded(args.lift_duration, 0.4, 5.0), 5),
            "low_hold_duration": round(_bounded(args.low_hold_duration, 0.0, 4.0), 5),
        },
    }


def _collect_evidence(root: Path) -> dict[str, Any]:
    side_grasp: list[dict[str, Any]] = []
    lift: list[dict[str, Any]] = []
    terminals: list[str] = []
    low_hold_guard_failures = 0
    for path in sorted(root.rglob("*.jsonl")) if root.exists() else []:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "episode_terminal":
                terminals.append(str(row.get("status") or ""))
            if row.get("event") == "primitive_status" and "low_hold_contact_lost" in str(row.get("detail") or ""):
                low_hold_guard_failures += 1
            if row.get("event") != "primitive_status" or row.get("status") not in {"success", "failed"}:
                continue
            skill = str(row.get("skill_name") or "")
            evidence = row.get("effect_evidence") if isinstance(row.get("effect_evidence"), dict) else {}
            effects = evidence.get("effects") if isinstance(evidence.get("effects"), dict) else {}
            contact = effects.get("object_contact_ready") if isinstance(effects.get("object_contact_ready"), dict) else {}
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            ik = metrics.get("ik") if isinstance(metrics.get("ik"), dict) else {}
            item = {
                "status": row.get("status"),
                "effect_passed": bool(evidence.get("passed")),
                "reason": evidence.get("reason") or row.get("detail"),
                "contact_count": _finite_or_none(contact.get("contact_count")),
                "critical_ik_error": _finite_or_none(ik.get("critical_max_error")),
            }
            if skill == "manip.side_grasp":
                side_grasp.append(item)
            elif skill == "manip.lift_object":
                lift.append(item)
    return {
        "terminal_statuses": terminals,
        "side_grasp_attempt_count": len(side_grasp),
        "side_grasp_effect_success_count": sum(int(item["effect_passed"]) for item in side_grasp),
        "side_grasp_contact_count_max": max((item["contact_count"] or 0 for item in side_grasp), default=0),
        "side_grasp_critical_ik_error_min": _min_or_none(item["critical_ik_error"] for item in side_grasp),
        "lift_attempt_count": len(lift),
        "lift_effect_success_count": sum(int(item["effect_passed"]) for item in lift),
        "low_hold_guard_failure_count": low_hold_guard_failures,
    }


def _report(args: argparse.Namespace, scene: Path, manifest: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        results,
        key=lambda item: (
            item["lift_effect_success_count"],
            item["side_grasp_effect_success_count"],
            item["side_grasp_contact_count_max"],
            -(item["side_grasp_critical_ik_error_min"] if item["side_grasp_critical_ik_error_min"] is not None else float("inf")),
            -item["elapsed_s"],
        ),
        reverse=True,
    )
    return {
        "schema": "sonic_world_model_contact_sweep_v0",
        "sequence": args.sequence,
        "stage": args.stage,
        "scene": _relative(scene),
        "manifest": _relative(manifest),
        "results": results,
        "summary": {
            "config_count": len(results),
            "terminal_count": sum(len(item["terminal_statuses"]) for item in results),
            "side_grasp_effect_success_count": sum(item["side_grasp_effect_success_count"] for item in results),
            "lift_effect_success_count": sum(item["lift_effect_success_count"] for item in results),
            "best_config": ranked[0]["name"] if ranked else None,
        },
    }


def _float_list(raw: str, name: str) -> list[float]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one number")
    return [float(value) for value in values]


def _bounded(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _finite_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def _min_or_none(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return min(finite) if finite else None


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
