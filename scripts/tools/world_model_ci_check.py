#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_TMP_DIR = "/tmp/sonic_world_ci"
DEFAULT_OUTPUT = "reports/readiness/world_model_ci_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal CI gate for the Sonic world-model benchmark/training framework."
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--offline-limit", type=int, default=500)
    parser.add_argument("--registry-limit", type=int, default=0, help="Registry tasks to validate; 0 means all selected suite tasks.")
    parser.add_argument("--policy-limit", type=int, default=500)
    parser.add_argument("--rollout-smoke-limit", type=int, default=2)
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--headless-probe-limit", type=int, default=1)
    parser.add_argument("--headless-probe-retries", type=int, default=1)
    parser.add_argument("--skip-headless-probe", action="store_true")
    parser.add_argument("--require-headless-probe", action="store_true")
    parser.add_argument("--skip-ros-smoke", action="store_true")
    parser.add_argument("--require-ros-smoke", action="store_true")
    parser.add_argument("--skip-visual-gate", action="store_true")
    parser.add_argument("--visual-vlm-report", default="reports/perception/visual_shadow_04_anchor_eval.json")
    parser.add_argument("--visual-summary", default="reports/policy_data/physical_qwen_shadow_visual_v0.summary.json")
    parser.add_argument("--min-visual-transitions", type=int, default=100)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument(
        "--python",
        default=os.environ.get("WORLD_MODEL_PYTHON", sys.executable),
        help="Python executable used for non-ROS CI subprocesses.",
    )
    parser.add_argument("--ros-timeout", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.python = _resolve_python(args.python)
    tmp_dir = _repo_path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    steps.append(
        _run_step(
            "world_model_replay",
            [args.python, str(SCRIPT_DIR / "world_model_replay.py")],
        )
    )
    steps.append(
        _run_step(
            "offline_benchmark",
            [
                args.python,
                str(SCRIPT_DIR / "benchmark_runner.py"),
                "--suite",
                str(args.suite),
                "--limit",
                str(max(0, int(args.offline_limit))),
                "--no-scene-validate",
                "--output-dir",
                str(tmp_dir / "benchmarks"),
                "--name",
                "sonic_general_v0_ci",
            ],
        )
    )
    steps.append(
        _run_step(
            "sequence_benchmark",
            [
                args.python,
                str(SCRIPT_DIR / "world_model_sequence_benchmark.py"),
                "--suite",
                str(args.suite),
                "--no-scene-validate",
                "--output-dir",
                str(tmp_dir / "sequence_benchmarks"),
                "--name",
                "sonic_sequence_ci",
                "--strict",
            ],
        )
    )
    registry_cmd = [
        args.python,
        str(SCRIPT_DIR / "task_object_registry_check.py"),
        "--suite",
        str(args.suite),
    ]
    if int(args.registry_limit) > 0:
        registry_cmd.extend(["--limit", str(int(args.registry_limit))])
    registry_cmd.extend(["--output", str(tmp_dir / "task_object_registry.json")])
    steps.append(_run_step("task_object_registry_check", registry_cmd))
    steps.append(
        _run_step(
            "world_model_unit_tests",
            [
                args.python,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(SCRIPTS_DIR / "sonic_world" / "tests"),
                "-p",
                "test_*.py",
            ],
            env_override={"PYTHONPATH": str(SCRIPTS_DIR)},
        )
    )

    policy_output = tmp_dir / "policy_samples.jsonl"
    policy_summary = tmp_dir / "policy_samples.csv"
    steps.append(
        _run_step(
            "policy_dataset_builder",
            [
                args.python,
                str(SCRIPT_DIR / "policy_dataset_builder.py"),
                "--suite",
                str(args.suite),
                "--limit",
                str(max(0, int(args.policy_limit))),
                "--output",
                str(policy_output),
                "--summary",
                str(policy_summary),
            ],
        )
    )
    steps.append(_policy_schema_step(policy_output))
    steps.append(
        _run_step(
            "rollout_dry_run_smoke",
            [
                args.python,
                str(SCRIPT_DIR / "task_suite_rollout.py"),
                "--suite",
                str(args.suite),
                "--limit",
                str(max(0, int(args.rollout_smoke_limit))),
                "--executable-tier",
                "current",
                "--runs-per-task",
                "1",
                "--output-dir",
                str(tmp_dir / "task_batches"),
                "--tag",
                "world_model_ci_dry_run",
                "--dry-run",
            ],
        )
    )
    steps.append(_headless_probe_step(args, tmp_dir))
    steps.append(_ros_smoke_step(args))
    steps.append(_visual_gate_step(args, tmp_dir))

    report = {
        "schema": "sonic_world_model_ci_report_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": str(args.suite),
        "summary": _summary(steps),
        "steps": steps,
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"\nWrote CI report: {_rel(output)}")
    return 0 if report["summary"]["failed"] == 0 else 1


def _run_step(
    name: str,
    cmd: list[str],
    *,
    required: bool = True,
    env_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    proc = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    duration = time.monotonic() - started
    passed = proc.returncode == 0
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "required": bool(required),
        "returncode": int(proc.returncode),
        "duration_s": round(duration, 3),
        "command": cmd,
        "output_tail": _tail(proc.stdout),
    }


def _policy_schema_step(path: Path) -> dict[str, Any]:
    started = time.monotonic()
    errors: list[str] = []
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row_count += 1
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    errors.append(f"{line_no}: row is not an object")
                    continue
                if payload.get("schema") != "task_skill_policy_v0":
                    errors.append(f"{line_no}: unexpected schema {payload.get('schema')!r}")
                action = payload.get("action")
                observation = payload.get("observation")
                if not isinstance(action, dict):
                    errors.append(f"{line_no}: missing action")
                    continue
                if not isinstance(observation, dict):
                    errors.append(f"{line_no}: missing observation")
                for key in ("policy_id", "task_id", "status", "skill_selection"):
                    if key not in action:
                        errors.append(f"{line_no}: action missing {key}")
                if not isinstance(action.get("skill_selection"), list):
                    errors.append(f"{line_no}: skill_selection is not a list")
                if len(errors) >= 12:
                    break
    except Exception as exc:
        errors.append(str(exc))
    duration = time.monotonic() - started
    if row_count <= 0:
        errors.append("no policy samples were written")
    return {
        "name": "policy_dataset_schema_check",
        "status": "passed" if not errors else "failed",
        "required": True,
        "returncode": 0 if not errors else 1,
        "duration_s": round(duration, 3),
        "path": str(path),
        "row_count": row_count,
        "errors": errors,
    }


def _headless_probe_step(args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    if args.skip_headless_probe:
        return _skipped("headless_mujoco_probe", "disabled by --skip-headless-probe", required=False)
    if not _python_import_ok(args.python, "mujoco"):
        if args.require_headless_probe:
            return _failed_optional("headless_mujoco_probe", "mujoco import failed", required=True)
        return _skipped("headless_mujoco_probe", "mujoco is not importable in this Python", required=False)
    cmd = [
        args.python,
        str(SCRIPT_DIR / "headless_mujoco_probe.py"),
        "--suite",
        str(args.suite),
        "--all-tasks",
        "--limit",
        str(max(1, int(args.headless_probe_limit))),
        "--steps",
        "0",
        "--table",
    ]
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(1, int(args.headless_probe_retries) + 1)):
        step = _run_step("headless_mujoco_probe", cmd, required=True)
        attempts.append(
            {
                "returncode": step["returncode"],
                "status": step["status"],
                "duration_s": step["duration_s"],
                "output_tail": step.get("output_tail", ""),
            }
        )
        if step["status"] == "passed":
            step["attempts"] = attempts
            return step
    step["attempts"] = attempts
    return step


def _ros_smoke_step(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_ros_smoke:
        return _skipped("world_model_ros_smoke", "disabled by --skip-ros-smoke", required=False)
    ros_python = str(args.ros_python or sys.executable)
    if not Path(ros_python).exists():
        ros_python = sys.executable
    if not _python_import_ok(ros_python, "rclpy"):
        if args.require_ros_smoke:
            return _failed_optional("world_model_ros_smoke", f"rclpy import failed under {ros_python}", required=True)
        return _skipped("world_model_ros_smoke", f"rclpy is not importable under {ros_python}", required=False)
    ros_log_dir = Path("/tmp/sonic_world_ci_ros_log")
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    return _run_step(
        "world_model_ros_smoke",
        [
            ros_python,
            str(SCRIPT_DIR / "world_model_ros_smoke_test.py"),
            "--timeout",
            str(float(args.ros_timeout)),
        ],
        required=True,
        env_override={"ROS_LOG_DIR": str(ros_log_dir)},
    )


def _visual_gate_step(args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    if args.skip_visual_gate:
        return _skipped("visual_training_gate", "disabled by --skip-visual-gate", required=False)
    report, summary = _repo_path(args.visual_vlm_report), _repo_path(args.visual_summary)
    if not report.is_file() or not summary.is_file():
        return _skipped("visual_training_gate", f"missing shadow artifacts: vlm={report.is_file()} summary={summary.is_file()}", required=False)
    output = tmp_dir / "visual_training_gate.json"
    step = _run_step(
        "visual_training_gate",
        [
            args.python, str(SCRIPT_DIR / "world_model_training_gate.py"), "--vlm-report", str(report),
            "--visual-summary", str(summary), "--min-visual-transitions", str(max(1, int(args.min_visual_transitions))),
            "--output", str(output),
        ],
        required=True,
    )
    if step["status"] != "passed":
        return step
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        decision = str(payload.get("decision") or "")
        if decision not in {"eligible_for_ab", "shadow_training_only"}:
            raise ValueError(f"unexpected decision {decision!r}")
        step["decision"] = decision
        step["checks"] = payload.get("checks")
    except Exception as exc:
        step.update({"status": "failed", "returncode": 1, "error": str(exc)})
    return step


def _python_import_ok(python: str, module: str) -> bool:
    proc = subprocess.run(
        [python, "-c", f"import {module}"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode == 0


def _resolve_python(value: str) -> str:
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(str(value))
    if resolved:
        return resolved
    raise ValueError(f"Python executable was not found: {value}")


def _skipped(name: str, reason: str, *, required: bool) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skipped",
        "required": bool(required),
        "returncode": 0,
        "duration_s": 0.0,
        "reason": reason,
    }


def _failed_optional(name: str, reason: str, *, required: bool) -> dict[str, Any]:
    return {
        "name": name,
        "status": "failed",
        "required": bool(required),
        "returncode": 1,
        "duration_s": 0.0,
        "reason": reason,
    }


def _summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    required = [step for step in steps if step.get("required")]
    failed = [step for step in required if step.get("status") == "failed"]
    return {
        "step_count": len(steps),
        "required_count": len(required),
        "passed": sum(1 for step in steps if step.get("status") == "passed"),
        "failed": len(failed),
        "skipped": sum(1 for step in steps if step.get("status") == "skipped"),
        "failed_steps": [step["name"] for step in failed],
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "world_model_ci="
        f"steps={summary['step_count']} required={summary['required_count']} "
        f"passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']}"
    )
    for step in report["steps"]:
        detail = step.get("reason") or ""
        if step.get("status") == "failed" and step.get("output_tail"):
            detail = step["output_tail"].splitlines()[-1]
        print(
            f"{step['status']:7s} {step['name']:30s} "
            f"required={str(bool(step.get('required'))):5s} "
            f"duration={float(step.get('duration_s') or 0.0):7.3f}s {detail}"
        )


def _tail(text: str, *, max_lines: int = 80, max_chars: int = 12000) -> str:
    lines = (text or "").splitlines()
    out = "\n".join(lines[-max_lines:])
    if len(out) > max_chars:
        return out[-max_chars:]
    return out


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
