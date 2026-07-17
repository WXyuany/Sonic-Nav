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
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sonic_world.task_suites import load_robocasa_task_suite, task_executability
from sonic_world.world_model import TaskObjectRegistry


DEFAULT_SUITE = "configs/world_model/task_suites/sonic_general_v0.yaml"
DEFAULT_OUTPUT_DIR = "reports/task_batches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Sonic world-model task suite through real reusable-stack rollouts. "
            "Each selected task gets its own stack because the MuJoCo XML scene is task-specific; "
            "rollouts inside that task reuse the same deployed SONIC model and reset MuJoCo state."
        )
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task", action="append", help="Task id to run. May be repeated.")
    parser.add_argument("--demo", choices=["ball", "box"], help="Only run tasks mapped to this demo kind.")
    parser.add_argument(
        "--executable-tier",
        choices=["all", "current"],
        default="all",
        help="Optionally filter to the tasks supported by the currently wired rollout primitives.",
    )
    parser.add_argument("--tag", default=None, help="Batch id. Defaults to a timestamp.")
    parser.add_argument("--runs-per-task", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default="reports/rollouts")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--camera", action="store_true", default=False)
    parser.add_argument("--reset-each-rollout", action="store_true", default=True)
    parser.add_argument("--continue-state", dest="reset_each_rollout", action="store_false")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--fail-on-rollout-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--policy-action-json")
    parser.add_argument("--policy-action-apply", choices=["off", "safe", "full"], default="off")
    parser.add_argument(
        "--world-policy-backend",
        choices=["heuristic", "memory", "learned"],
        default="heuristic",
        help="Policy backend passed through to tools/rollout_batch.py and world_model_node.py.",
    )
    parser.add_argument("--world-policy-model", help="Policy model JSON passed through to world_model_node.py.")
    parser.add_argument("--vlm-anchor-bridge", action="store_true", help="Pass through to rollout_batch.py.")
    parser.add_argument("--vlm-detections-topic", default="/sonic_world/vlm_detections")
    parser.add_argument("--qwen-vl-shadow", action="store_true", help="Run local Qwen-VL only as a paired-anchor shadow evaluator.")
    parser.add_argument("--qwen-vl-model-path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen-vl-host", default="127.0.0.1")
    parser.add_argument("--qwen-vl-port", type=int, default=8000)
    parser.add_argument("--qwen-vl-period", type=float, default=5.0)
    parser.add_argument("--qwen-vl-max-new-tokens", type=int, default=512)
    parser.add_argument("--qwen-vl-device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--qwen-vl-gpu-memory-gib", type=float, default=0.0)
    parser.add_argument("--qwen-vl-depth-cache-size", type=int, default=480)
    parser.add_argument("--qwen-vl-health-timeout", type=float, default=180.0)
    parser.add_argument("--qwen-vl-shadow-dir", default="reports/perception")
    parser.add_argument("--qwen-vl-audit-output", default="")
    parser.add_argument("--qwen-vl-instruction", default="")
    parser.add_argument("--qwen-vl-gate-report")
    parser.add_argument("--qwen-vl-shadow-max-pairs", type=int, default=0)
    parser.add_argument("--qwen-vl-calibration-max-samples", type=int, default=20)
    parser.add_argument("--qwen-vl-temporal-window", type=int, default=3)
    parser.add_argument("--qwen-vl-temporal-min-observations", type=int, default=3)
    parser.add_argument("--visual-auto-reobserve", action="store_true")
    parser.add_argument("--visual-reobserve-max-attempts", type=int, default=2)
    parser.add_argument("--visual-reobserve-cooldown-s", type=float, default=1.0)
    parser.add_argument("--visual-recovery-escalate-navigation", action="store_true")
    parser.add_argument("--perception-shadow-backend", choices=["qwen", "grounding_dino", "hsv"], default="qwen")
    parser.add_argument("--grounding-dino-model-path", default="models/grounding-dino-tiny")
    parser.add_argument("--grounding-dino-host", default="127.0.0.1")
    parser.add_argument("--grounding-dino-port", type=int, default=8001)
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.20)
    parser.add_argument("--world-primitive-runner", action="store_true", help="Pass through to rollout_batch.py.")
    parser.add_argument(
        "--assisted-grasp",
        action="store_true",
        help="Enable non-physical attach assistance for debugging. Official success metrics keep this disabled.",
    )
    parser.add_argument(
        "--world-primitive-backend",
        choices=["status_only", "zmq_phase"],
        default="status_only",
    )
    parser.add_argument(
        "--extra-demo-arg",
        action="append",
        default=[],
        help="Extra argument passed to the underlying manipulation demo. Use multiple times for pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs_per_task <= 0:
        raise SystemExit("--runs-per-task must be positive")
    if args.qwen_vl_shadow:
        args.camera = True

    suite = load_robocasa_task_suite(_repo_path(args.suite), repo_root=REPO)
    tasks = _select_tasks(suite.tasks, args)
    if not tasks:
        raise SystemExit("No task-suite rollouts selected.")

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    output_dir = _repo_path(args.output_dir) / _safe_name(tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        spec = _task_rollout_spec(task, args=args, tag=tag, order=index)
        records.append(spec)
        _print_task_header(index, len(tasks), spec)
        if args.dry_run:
            continue
        code = subprocess.call(spec["command"], cwd=REPO)
        spec["exit_code"] = int(code)
        spec["status"] = "success" if code == 0 else "failed"
        if code != 0:
            failures.append(spec)
            if args.stop_on_fail:
                break

    _write_manifest(output_dir, records, suite_name=suite.name, suite_version=suite.version, suite_path=args.suite)
    print(f"\nWrote task batch manifest: {_rel(output_dir / 'manifest.json')}")
    if failures:
        print("[TASK_BATCH] Failed tasks:")
        for item in failures:
            print(f"  {item['task_id']}: exit_code={item.get('exit_code')}")
        if args.fail_on_rollout_fail or args.stop_on_fail:
            return 1
    return 0


def _select_tasks(tasks: tuple[Any, ...], args: argparse.Namespace) -> list[Any]:
    selected = list(tasks)
    if args.task:
        wanted = set(args.task)
        selected = [task for task in selected if task.task_id in wanted]
        missing = sorted(wanted - {task.task_id for task in selected})
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
    if args.demo:
        selected = [task for task in selected if _demo_kind(task) == args.demo]
    if args.executable_tier == "current":
        kept = []
        skipped = []
        for task in selected:
            executable = task_executability(task, tier="current")
            if executable.executable:
                kept.append(task)
            else:
                skipped.append((task.task_id, executable.reason))
        if skipped:
            print("[TASK_BATCH] skipped ineligible current-tier tasks:")
            for task_id, reason in skipped[:20]:
                print(f"  {task_id}: {reason}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more")
        selected = kept
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    return selected


def _task_rollout_spec(task: Any, *, args: argparse.Namespace, tag: str, order: int) -> dict[str, Any]:
    demo = _demo_kind(task)
    if demo not in {"ball", "box"}:
        raise ValueError(f"task {task.task_id} has unsupported demo kind {demo!r}")
    object_id = str(task.request.object_id or _first_pickable_object_id(task) or "")
    target_id = str(task.request.target_id or "")
    object_category = _object_category(task, object_id)
    scene_xml = str(task.scene.scene_xml)
    prefix = _safe_name(f"{tag}_{task.task_id}")
    registry = TaskObjectRegistry.from_task_case(task)
    object_anchor = _object_anchor_name(task, object_id, registry=registry)
    place_site = _target_site_name(task, target_id, registry=registry)
    command = [
        str(args.python),
        str(SCRIPT_DIR / "rollout_batch.py"),
        demo,
        "--scene",
        scene_xml,
        "--runs",
        str(args.runs_per_task),
        "--prefix",
        prefix,
        "--report-path",
        str(args.report_path),
        "--python",
        str(args.python),
        "--anchor-object-name",
        object_anchor,
        "--task-object-id",
        object_id,
        "--task-object-category",
        object_category,
        "--world-policy-backend",
        str(args.world_policy_backend),
    ]
    if args.world_policy_model:
        command.extend(["--world-policy-model", str(args.world_policy_model)])
    if args.vlm_anchor_bridge:
        command.extend(["--vlm-anchor-bridge", "--vlm-detections-topic", str(args.vlm_detections_topic)])
    if args.qwen_vl_shadow:
        command.extend(
            [
                "--qwen-vl-shadow",
                "--qwen-vl-model-path",
                str(args.qwen_vl_model_path),
                "--qwen-vl-host",
                str(args.qwen_vl_host),
                "--qwen-vl-port",
                str(int(args.qwen_vl_port)),
                "--qwen-vl-period",
                str(float(args.qwen_vl_period)),
                "--qwen-vl-max-new-tokens",
                str(int(args.qwen_vl_max_new_tokens)),
                "--qwen-vl-device",
                str(args.qwen_vl_device),
                "--qwen-vl-gpu-memory-gib",
                str(max(0.0, float(args.qwen_vl_gpu_memory_gib))),
                "--qwen-vl-depth-cache-size",
                str(max(24, int(args.qwen_vl_depth_cache_size))),
                "--qwen-vl-health-timeout",
                str(float(args.qwen_vl_health_timeout)),
                "--qwen-vl-shadow-dir",
                str(args.qwen_vl_shadow_dir),
                "--qwen-vl-audit-output",
                str(args.qwen_vl_audit_output),
                "--qwen-vl-instruction",
                str(args.qwen_vl_instruction),
                "--qwen-vl-shadow-max-pairs",
                str(int(args.qwen_vl_shadow_max_pairs)),
                "--qwen-vl-calibration-max-samples",
                str(int(args.qwen_vl_calibration_max_samples)),
                "--qwen-vl-temporal-window",
                str(int(args.qwen_vl_temporal_window)),
                "--qwen-vl-temporal-min-observations",
                str(int(args.qwen_vl_temporal_min_observations)),
                *( [
                    "--visual-auto-reobserve",
                    "--visual-reobserve-max-attempts",
                    str(max(0, int(args.visual_reobserve_max_attempts))),
                    "--visual-reobserve-cooldown-s",
                    str(max(0.0, float(args.visual_reobserve_cooldown_s))),
                    *( ["--visual-recovery-escalate-navigation"] if args.visual_recovery_escalate_navigation else [] ),
                ] if args.visual_auto_reobserve else [] ),
                "--perception-shadow-backend",
                str(args.perception_shadow_backend),
                "--grounding-dino-model-path",
                str(args.grounding_dino_model_path),
                "--grounding-dino-host",
                str(args.grounding_dino_host),
                "--grounding-dino-port",
                str(int(args.grounding_dino_port)),
                "--grounding-dino-box-threshold",
                str(float(args.grounding_dino_box_threshold)),
                "--grounding-dino-text-threshold",
                str(float(args.grounding_dino_text_threshold)),
                "--grounding-dino-object-label",
                _grounding_label(object_category),
            ]
        )
        if target_id and demo == "ball":
            command.extend(["--grounding-dino-target-label", "green target|green circle|target"])
        if args.qwen_vl_gate_report:
            command.extend(["--qwen-vl-gate-report", str(args.qwen_vl_gate_report)])
    if args.world_primitive_runner:
        command.extend(["--world-primitive-runner", "--world-primitive-backend", str(args.world_primitive_backend)])
    if args.start_index is not None:
        command.extend(["--start-index", str(int(args.start_index))])
    if args.headless:
        command.append("--headless")
    if not args.camera:
        command.append("--no-camera")
    if args.reset_each_rollout:
        command.append("--reset-each-rollout")
    if args.stop_on_fail:
        command.append("--stop-on-fail")
    if args.fail_on_rollout_fail:
        command.append("--fail-on-rollout-fail")
    if place_site and demo == "ball":
        command.extend(["--anchor-place-site", place_site])
    if target_id:
        command.extend(["--task-target-id", target_id])
    command.extend(_generated_batch_args(task, scene_xml))

    demo_args = ["--task-id", str(task.task_id)]
    if args.policy_action_json and args.policy_action_apply != "off":
        demo_args.extend(
            [
                "--policy-action-json",
                str(args.policy_action_json),
                "--policy-action-task-id",
                str(task.task_id),
                "--policy-action-apply",
                str(args.policy_action_apply),
            ]
        )
    demo_args.extend(_generated_scene_demo_args(task, scene_xml, assisted_grasp=bool(args.assisted_grasp)))
    demo_args.extend(str(item) for item in args.extra_demo_arg)
    if demo_args:
        command.append("--")
        command.extend(demo_args)

    run_ids = [
        f"{prefix}_{(args.start_index or 1) + offset:03d}"
        for offset in range(int(args.runs_per_task))
    ]
    return {
        "schema": "sonic_task_suite_rollout_spec_v0",
        "order": order,
        "task_id": task.task_id,
        "description": task.description,
        "demo_kind": demo,
        "scene": scene_xml,
        "object_id": object_id,
        "object_category": object_category,
        "target_id": target_id,
        "anchor_object_name": object_anchor,
        "anchor_place_site": place_site,
        "object_registry": registry.to_dict(),
        "runs": int(args.runs_per_task),
        "run_id_prefix": prefix,
        "expected_run_ids": run_ids,
        "reset_each_rollout": bool(args.reset_each_rollout),
        "headless": bool(args.headless),
        "camera": bool(args.camera),
        "policy_action_json": str(args.policy_action_json or ""),
        "policy_action_apply": str(args.policy_action_apply),
        "world_policy_backend": str(args.world_policy_backend),
        "world_policy_model": str(args.world_policy_model or ""),
        "vlm_anchor_bridge": bool(args.vlm_anchor_bridge),
        "vlm_detections_topic": str(args.vlm_detections_topic),
        "world_primitive_runner": bool(args.world_primitive_runner),
        "world_primitive_backend": str(args.world_primitive_backend),
        "assisted_grasp": bool(args.assisted_grasp),
        "executability": task_executability(task, tier="current").to_dict(),
        "command": command,
        "status": "dry_run" if args.dry_run else "pending",
        "exit_code": None,
    }


def _demo_kind(task: Any) -> str:
    expect = task.expectation if isinstance(task.expectation, dict) else {}
    demo = str(expect.get("demo_kind") or "").strip()
    if demo:
        return demo
    tags = set(str(tag) for tag in task.tags)
    if "bimanual_clamp" in tags:
        return "box"
    return "ball"


def _generated_scene_demo_args(task: Any, scene_xml: str, *, assisted_grasp: bool = False) -> list[str]:
    generated = "scene_sonic_task_" in str(scene_xml) or str(task.metadata.get("generated_scene_xml") or "")
    if not generated:
        return []
    # Generated benchmark scenes place tabletop objects around z=0.84 in MuJoCo world space.
    # The demo anchor gate reads the same point in the live pelvis/base frame, so the first
    # anchor can be around z=1.0 even though the object is a normal tabletop object.
    # The anchor publishers also clamp walk_duration to at least 0.6s; generated close-start
    # tasks need to be allowed to stand still instead of taking an extra step away.
    if _demo_kind(task) == "ball":
        out = [
            "--post-start-anchor-delay",
            "2.0",
            "--initial-map-anchor-fallback",
            "--initial-anchor-min-x",
            "0.05",
            "--initial-anchor-max-z",
            "0.95",
            "--runtime-anchor-max-z",
            "1.35",
        ]
    else:
        out = [
            "--post-start-anchor-delay",
            "2.0",
            "--initial-map-anchor-fallback",
            "--initial-anchor-min-x",
            "0.0",
            "--initial-anchor-max-abs-y",
            "1.25",
            "--initial-anchor-max-z",
            "1.35",
            "--walk-extra-duration",
            "0.0",
            "--min-approach-duration",
            "0.2",
        ]
    if _demo_kind(task) == "ball":
        out.extend(
            [
                "--walk-extra-duration",
                "0.0",
                "--min-approach-duration",
                "0.2",
                "--max-approach-retries",
                "10",
                "--no-approach-response-adapt",
                "--align-target-x",
                "0.46",
                "--align-target-y",
                "-0.24",
                "--align-max-lateral",
                "0.18",
                "--align-lateral-gain",
                "1.0",
                "--align-forward-response-sign",
                "1.0",
                "--align-lateral-response-sign",
                "-1.0",
                "--no-align-response-adapt",
                "--pregrasp-align-base",
                "--pregrasp-align-duration",
                "1.2",
                "--palm-pocket-table-z-radius",
                "-0.35",
                "--palm-pocket-lift-z-radius",
                "-0.20",
                "--approach-tolerance",
                "0.04",
                "--approach-soft-tolerance",
                "0.10",
            ]
        )
        if assisted_grasp:
            out.append("--ball-attach")
    else:
        out.extend(
            [
                "--max-approach-retries",
                "8",
                "--approach-retry-duration",
                "1.2",
            ]
        )
    return out


def _generated_batch_args(task: Any, scene_xml: str) -> list[str]:
    generated = "scene_sonic_task_" in str(scene_xml) or str(task.metadata.get("generated_scene_xml") or "")
    if not generated:
        return []
    if _demo_kind(task) == "ball":
        return ["--anchor-approach-standoff", "0.46"]
    return ["--anchor-approach-standoff", "0.45"]


def _object_anchor_name(task: Any, object_id: str, *, registry: TaskObjectRegistry | None = None) -> str:
    if not object_id:
        return "demo_ball_visual" if _demo_kind(task) == "ball" else "demo_box_visual"
    if registry is not None:
        resolved = registry.resolve_anchor_name(object_id, demo_kind=_demo_kind(task))
        if resolved:
            return resolved
    scene_xml = str(task.scene.scene_xml)
    generated = "scene_sonic_task_" in scene_xml or str(task.metadata.get("generated_scene_xml") or "")
    return f"{object_id}_geom" if generated else object_id


def _target_site_name(task: Any, target_id: str, *, registry: TaskObjectRegistry | None = None) -> str:
    if not target_id:
        return ""
    if registry is not None:
        record = registry.get(target_id)
        if record is not None and record.site_name:
            return record.site_name
    scene_xml = str(task.scene.scene_xml)
    generated = "scene_sonic_task_" in scene_xml or str(task.metadata.get("generated_scene_xml") or "")
    return f"{target_id}_site" if generated else target_id


def _first_pickable_object_id(task: Any) -> str:
    support_ids = {str(obj.get("support") or "") for obj in task.objects if isinstance(obj, dict)}
    for obj in task.objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id") or obj.get("id") or "")
        if not object_id or object_id in support_ids:
            continue
        if str(obj.get("category") or "") in {"table", "counter", "place_target"}:
            continue
        return object_id
    return ""


def _object_category(task: Any, object_id: str) -> str:
    for item in task.objects:
        if isinstance(item, dict) and str(item.get("object_id") or item.get("id") or "") == object_id:
            return str(item.get("category") or "object")
    return "object"


def _grounding_label(category: str) -> str:
    category = " ".join(part for part in str(category).replace("_", " ").split() if part) or "object"
    # Generated Sonic sphere tasks use the green sonic_task_sphere_mat material.
    return "green ball|ball" if category == "ball" else category


def _write_manifest(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    suite_name: str,
    suite_version: str,
    suite_path: str,
) -> None:
    payload = {
        "schema": "sonic_task_suite_rollout_manifest_v0",
        "suite": suite_name,
        "suite_version": suite_version,
        "suite_path": _rel(_repo_path(suite_path)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": len(records),
        "tasks": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "order",
        "task_id",
        "demo_kind",
        "scene",
        "object_id",
        "object_category",
        "target_id",
        "anchor_object_name",
        "anchor_place_site",
        "runs",
        "run_id_prefix",
        "reset_each_rollout",
        "headless",
        "camera",
        "policy_action_apply",
        "world_policy_backend",
        "world_policy_model",
        "vlm_anchor_bridge",
        "vlm_detections_topic",
        "world_primitive_runner",
        "world_primitive_backend",
        "status",
        "exit_code",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _print_task_header(index: int, total: int, spec: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(
        f"[TASK_BATCH] {index}/{total} {spec['task_id']} "
        f"demo={spec['demo_kind']} runs={spec['runs']} scene={spec['scene']}"
    )
    print(
        f"[TASK_BATCH] anchor object={spec['anchor_object_name']} "
        f"place={spec['anchor_place_site'] or '-'} reset={spec['reset_each_rollout']}"
    )
    print("[TASK_BATCH] " + " ".join(_shell_quote(part) for part in spec["command"]))
    print("=" * 80)


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _shell_quote(value: object) -> str:
    text = str(value)
    if not text or any(ch.isspace() or ch in "'\"$`\\!" for ch in text):
        return "'" + text.replace("'", "'\"'\"'") + "'"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
