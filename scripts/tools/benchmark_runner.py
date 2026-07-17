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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from headless_mujoco_probe import probe_scene
from sonic_world.scenarios import ScenarioSpec, replay_scenario
from sonic_world.task_suites import load_robocasa_task_suite


DEFAULT_SUITE = "configs/world_model/task_suites/molmospaces_robocasa_v0.yaml"
DEFAULT_OUTPUT_DIR = Path("reports/benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline benchmark runner for Sonic world-model task suites.")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task", action="append", help="Task id to evaluate. May be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None, help="Report basename. Defaults to the suite filename stem.")
    parser.add_argument("--no-scene-validate", action="store_true")
    parser.add_argument("--json", action="store_true", default=True, help="Write JSON report.")
    parser.add_argument("--csv", action="store_true", default=True, help="Write CSV leaderboard.")
    parser.add_argument("--markdown", action="store_true", default=True, help="Write Markdown leaderboard.")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--headless-probe", action="store_true", help="Also run a short headless MuJoCo physics probe per task.")
    parser.add_argument("--probe-steps", type=int, default=0, help="Uncontrolled mj_step count for --headless-probe.")
    parser.add_argument("--probe-fall-height", type=float, default=0.35)
    parser.add_argument("--probe-fall-angle-deg", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_path = _repo_path(args.suite)
    suite = load_robocasa_task_suite(suite_path, repo_root=REPO)
    tasks = list(suite.tasks)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.task_id in wanted]
        missing = sorted(wanted - {task.task_id for task in tasks})
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(missing)}")
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]
    if not tasks:
        raise SystemExit("No benchmark tasks selected.")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            _evaluate_task(
                task,
                validate_scene=not args.no_scene_validate,
                headless_probe=args.headless_probe,
                probe_steps=args.probe_steps,
                probe_fall_height=args.probe_fall_height,
                probe_fall_angle_deg=args.probe_fall_angle_deg,
            )
        )

    summary = _summary(rows, suite_name=suite.name, suite_version=suite.version, suite_path=suite_path)
    report = {"summary": summary, "tasks": rows}
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(rows, summary)

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or suite_path.stem
    if args.json:
        (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(output_dir / f"{stem}.csv", rows)
    if args.markdown:
        (output_dir / f"{stem}.md").write_text(_markdown(rows, summary), encoding="utf-8")
    print(f"\nWrote benchmark report: {_rel(output_dir / f'{stem}.json')}")
    return 0 if summary["offline_success_rate"] >= 1.0 else 1


def _evaluate_task(
    task: Any,
    *,
    validate_scene: bool,
    headless_probe: bool,
    probe_steps: int,
    probe_fall_height: float,
    probe_fall_angle_deg: float,
) -> dict[str, Any]:
    scene_valid, scene_error = _validate_scene(task.scene.scene_xml) if validate_scene else (None, "")
    physics_probe: dict[str, Any] | None = None
    physics_healthy: bool | None = None
    physics_fallen: bool | None = None
    physics_base_z: float | None = None
    physics_object_z: float | None = None
    physics_contact_count: int | None = None
    physics_object_contact_count: int | None = None
    physics_error = ""
    if headless_probe:
        physics_probe = probe_scene(
            task.scene.scene_xml,
            task_id=task.task_id,
            object_body=task.request.object_id,
            object_id=task.request.object_id,
            steps=probe_steps,
            fall_height=probe_fall_height,
            fall_angle_deg=probe_fall_angle_deg,
            max_contacts=8,
        )
        physics_healthy = bool(physics_probe.get("healthy"))
        physics_fallen = bool(physics_probe.get("fallen"))
        physics_error = str(physics_probe.get("error") or "")
        base = physics_probe.get("base") or {}
        obj = physics_probe.get("object") or {}
        if base.get("position"):
            physics_base_z = float(base["position"][2])
        if obj.get("position"):
            physics_object_z = float(obj["position"][2])
        physics_contact_count = int(physics_probe.get("contact_count") or 0)
        physics_object_contact_count = int(physics_probe.get("object_contact_count") or 0)

    replay_error = ""
    try:
        replay = replay_scenario(ScenarioSpec.from_dict(task.scenario()))
        result = replay.tasks[0].result
        expectation_passed = replay.passed
        steps = [step.name for step in result.skill_graph.steps]
        handlers = [step.handler for step in result.dispatch_plan.steps]
        dispatch = result.dispatch_plan.metadata
        runtime = result.runtime_plan.metadata
        decision_status = result.decision_plan.status
        grasp = result.skill_graph.metadata.get("grasp_affordance")
        category = result.skill_graph.metadata.get("object_category")
        demo_kind = result.runtime_plan.demo_kind
        warnings = result.skill_graph.metadata.get("warnings") or []
    except Exception as exc:
        replay_error = str(exc)
        expectation_passed = False
        steps = []
        handlers = []
        dispatch = {}
        runtime = {}
        decision_status = "error"
        grasp = None
        category = None
        demo_kind = None
        warnings = []

    scene_ok = True if scene_valid is None else bool(scene_valid)
    contract_errors = int(dispatch.get("contract_error_count") or 0)
    unready = int(dispatch.get("unready_count") or 0)
    missing_skills = list(runtime.get("missing_skills") or [])
    plan_ready = decision_status == "ready_to_execute" and not missing_skills and unready == 0 and contract_errors == 0
    physics_ok = True if physics_healthy is None else physics_healthy
    offline_success = bool(expectation_passed and scene_ok and plan_ready and physics_ok)
    return {
        "task_id": task.task_id,
        "scene": task.scene.scene_xml,
        "request": task.request.verb,
        "object_id": task.request.object_id,
        "target_id": task.request.target_id,
        "object_category": category,
        "grasp_affordance": grasp,
        "demo_kind": demo_kind,
        "skill_count": len(steps),
        "steps": steps,
        "handlers": handlers,
        "scene_valid": scene_valid,
        "scene_error": scene_error,
        "expectation_passed": expectation_passed,
        "plan_ready": plan_ready,
        "decision_status": decision_status,
        "missing_skills": missing_skills,
        "unready_count": unready,
        "contract_error_count": contract_errors,
        "contract_warning_count": int(dispatch.get("contract_warning_count") or 0),
        "warnings": warnings,
        "headless_probe_enabled": headless_probe,
        "physics_healthy": physics_healthy,
        "physics_fallen": physics_fallen,
        "physics_base_z": physics_base_z,
        "physics_object_z": physics_object_z,
        "physics_contact_count": physics_contact_count,
        "physics_object_contact_count": physics_object_contact_count,
        "physics_error": physics_error,
        "physics_probe": physics_probe,
        "offline_success": offline_success,
        "replay_error": replay_error,
    }


def _validate_scene(scene_xml: str) -> tuple[bool, str]:
    try:
        import mujoco
    except Exception as exc:
        return False, f"mujoco import failed: {exc}"
    path = _repo_path(scene_xml)
    if not path.exists():
        return False, f"scene XML not found: {path}"
    try:
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _summary(rows: list[dict[str, Any]], *, suite_name: str, suite_version: str, suite_path: Path) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row["offline_success"])
    scene_valid = sum(1 for row in rows if row["scene_valid"] in {True, None})
    plan_ready = sum(1 for row in rows if row["plan_ready"])
    physics_rows = [row for row in rows if row.get("headless_probe_enabled")]
    by_affordance: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in rows:
        by_affordance[str(row.get("grasp_affordance") or "none")] = by_affordance.get(str(row.get("grasp_affordance") or "none"), 0) + 1
        by_category[str(row.get("object_category") or "unknown")] = by_category.get(str(row.get("object_category") or "unknown"), 0) + 1
    return {
        "suite": suite_name,
        "version": suite_version,
        "suite_path": _rel(suite_path),
        "task_count": total,
        "offline_success_count": successes,
        "offline_success_rate": successes / total if total else 0.0,
        "scene_valid_count": scene_valid,
        "plan_ready_count": plan_ready,
        "physics_probe_count": len(physics_rows),
        "physics_healthy_count": sum(1 for row in physics_rows if row.get("physics_healthy")),
        "physics_fallen_count": sum(1 for row in physics_rows if row.get("physics_fallen")),
        "by_affordance": dict(sorted(by_affordance.items())),
        "by_category": dict(sorted(by_category.items())),
    }


def _print_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(f"benchmark={summary['suite']}:{summary['version']} tasks={summary['task_count']} offline_success={summary['offline_success_count']}/{summary['task_count']}")
    has_physics = any(row.get("headless_probe_enabled") for row in rows)
    if has_physics:
        print(
            f"{'task_id':28s} {'cat':10s} {'grasp':18s} {'scene':5s} {'phys':5s} "
            f"{'base_z':>6s} {'obj_z':>6s} {'ready':5s} {'success':7s} decision"
        )
    else:
        print(
            f"{'task_id':28s} {'cat':10s} {'grasp':18s} {'scene':5s} {'ready':5s} "
            f"{'success':7s} {'skills':6s} decision"
        )
    for row in rows:
        if has_physics:
            print(
                f"{row['task_id'][:28]:28s} "
                f"{str(row.get('object_category') or '-')[:10]:10s} "
                f"{str(row.get('grasp_affordance') or '-')[:18]:18s} "
                f"{_yes(row['scene_valid'] in {True, None}):5s} "
                f"{_yes(bool(row.get('physics_healthy'))):5s} "
                f"{_fmt_float(row.get('physics_base_z')):>6s} "
                f"{_fmt_float(row.get('physics_object_z')):>6s} "
                f"{_yes(row['plan_ready']):5s} "
                f"{_yes(row['offline_success']):7s} "
                f"{row['decision_status']}"
            )
        else:
            print(
                f"{row['task_id'][:28]:28s} "
                f"{str(row.get('object_category') or '-')[:10]:10s} "
                f"{str(row.get('grasp_affordance') or '-')[:18]:18s} "
                f"{_yes(row['scene_valid'] in {True, None}):5s} "
                f"{_yes(row['plan_ready']):5s} "
                f"{_yes(row['offline_success']):7s} "
                f"{row['skill_count']:<6d} "
                f"{row['decision_status']}"
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "request",
        "object_category",
        "grasp_affordance",
        "demo_kind",
        "skill_count",
        "scene_valid",
        "plan_ready",
        "expectation_passed",
        "offline_success",
        "decision_status",
        "unready_count",
        "contract_error_count",
        "contract_warning_count",
        "headless_probe_enabled",
        "physics_healthy",
        "physics_fallen",
        "physics_base_z",
        "physics_object_z",
        "physics_contact_count",
        "physics_object_contact_count",
        "physics_error",
        "scene",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['suite']}:{summary['version']}",
        "",
        f"- tasks: {summary['task_count']}",
        f"- offline success: {summary['offline_success_count']}/{summary['task_count']} ({summary['offline_success_rate']:.1%})",
        f"- scene valid: {summary['scene_valid_count']}/{summary['task_count']}",
        f"- plan ready: {summary['plan_ready_count']}/{summary['task_count']}",
    ]
    if summary.get("physics_probe_count"):
        lines.extend(
            [
                f"- physics healthy: {summary['physics_healthy_count']}/{summary['physics_probe_count']}",
                f"- physics fallen: {summary['physics_fallen_count']}/{summary['physics_probe_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "| task | category | grasp | scene | physics | ready | success | skills |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task_id"]),
                    str(row.get("object_category") or "-"),
                    str(row.get("grasp_affordance") or "-"),
                    _yes(row["scene_valid"] in {True, None}),
                    _physics_cell(row),
                    _yes(row["plan_ready"]),
                    _yes(row["offline_success"]),
                    str(row["skill_count"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _fmt_float(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _physics_cell(row: dict[str, Any]) -> str:
    if not row.get("headless_probe_enabled"):
        return "-"
    return _yes(bool(row.get("physics_healthy")))


if __name__ == "__main__":
    raise SystemExit(main())
