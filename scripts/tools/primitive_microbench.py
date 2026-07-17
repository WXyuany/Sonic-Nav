#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
from sonic_world.world_model import anchor_to_world
from sonic_world.world_model.affordance import select_grasp_affordance
from sonic_world.world_model.entities import WorldObject, finite_float


DEFAULT_SUITE = "configs/world_model/task_suites/molmospaces_robocasa_v0.yaml"
DEFAULT_OUTPUT_DIR = Path("reports/primitive_microbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-GUI primitive feasibility microbenchmark for Sonic world-model tasks. "
            "It scores approach/grasp/lift/place/fall preconditions before expensive rollout."
        )
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task", action="append", help="Task id to evaluate. May be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None, help="Report basename. Defaults to the suite filename stem.")
    parser.add_argument("--no-headless-probe", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    parser.add_argument("--print-json", action="store_true")
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
        raise SystemExit("No primitive microbench tasks selected.")

    rows = [
        evaluate_task(
            task,
            headless_probe=not args.no_headless_probe,
            probe_steps=args.probe_steps,
            threshold=args.score_threshold,
        )
        for task in tasks
    ]
    summary = _summary(rows, suite_name=suite.name, suite_version=suite.version, suite_path=suite_path)
    report = {"summary": summary, "tasks": rows}

    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(rows, summary)

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or suite_path.stem
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / f"{stem}.csv", rows)
    (output_dir / f"{stem}.md").write_text(_markdown(rows, summary), encoding="utf-8")
    print(f"\nWrote primitive microbench report: {_rel(output_dir / f'{stem}.json')}")
    return 0 if summary["success_count"] == summary["task_count"] else 1


def evaluate_task(task: Any, *, headless_probe: bool, probe_steps: int, threshold: float) -> dict[str, Any]:
    world = anchor_to_world(task.anchor())
    obj = world.get_object(task.request.object_id) if task.request.object_id else world.primary_object()
    target = world.get_object(task.request.target_id) if task.request.target_id else None
    grasp = select_grasp_affordance(obj) if obj is not None else None
    physics = (
        probe_scene(
            task.scene.scene_xml,
            task_id=task.task_id,
            object_body=task.request.object_id,
            object_id=task.request.object_id,
            steps=probe_steps,
            max_contacts=8,
        )
        if headless_probe
        else None
    )

    plan_ready = False
    plan_error = ""
    steps: list[str] = []
    try:
        replay = replay_scenario(ScenarioSpec.from_dict(task.scenario()))
        result = replay.tasks[0].result
        dispatch = result.dispatch_plan.metadata
        runtime = result.runtime_plan.metadata
        plan_ready = bool(
            replay.passed
            and result.decision_plan.status == "ready_to_execute"
            and int(dispatch.get("contract_error_count") or 0) == 0
            and int(dispatch.get("unready_count") or 0) == 0
            and not list(runtime.get("missing_skills") or [])
        )
        steps = [step.name for step in result.skill_graph.steps]
    except Exception as exc:
        plan_error = str(exc)

    stages = [
        _eval_approach(obj, grasp, threshold),
        _eval_workspace(obj, world.objects.get(obj.support) if obj and obj.support else None, threshold),
        _eval_grasp(obj, grasp, threshold),
        _eval_lift(obj, grasp, physics, threshold),
        _eval_place(task, target, threshold),
        _eval_fall(physics, threshold),
    ]
    failed = [stage for stage in stages if not stage["passed"]]
    success = bool(plan_ready and not failed)
    return {
        "task_id": task.task_id,
        "scene": task.scene.scene_xml,
        "request": task.request.verb,
        "object_id": task.request.object_id,
        "target_id": task.request.target_id,
        "object_category": obj.category if obj else None,
        "grasp_affordance": grasp.name if grasp else None,
        "plan_ready": plan_ready,
        "plan_error": plan_error,
        "steps": steps,
        "stages": stages,
        "fail_stage": failed[0]["name"] if failed else "",
        "success": success,
        "physics_healthy": physics.get("healthy") if physics else None,
        "physics_fallen": physics.get("fallen") if physics else None,
        "physics_base_z": _state_z(physics, "base") if physics else None,
        "physics_object_z": _state_z(physics, "object") if physics else None,
        "physics_object_mass": _state_mass(physics, "object") if physics else None,
    }


def _eval_approach(obj: WorldObject | None, grasp: Any, threshold: float) -> dict[str, Any]:
    if obj is None or obj.pose_base is None:
        return _stage("approach", 0.0, ["missing_object_base_pose"], {}, threshold)
    x, y, z = obj.pose_base.position
    params = grasp.params if grasp is not None else obj.properties.get("grasp") or {}
    target_x = finite_float(params.get("reach_x") or params.get("approach_target_x"), 0.52)
    x_error = abs(float(x) - target_x)
    lateral_abs = abs(float(y))
    x_score = _score_abs_error(x_error, soft=0.04, hard=0.20)
    lateral_score = _score_upper(lateral_abs, soft=0.28, hard=0.50)
    reasons: list[str] = []
    if x_score < threshold:
        reasons.append("base_standoff_retarget_needed")
    if lateral_score < threshold:
        reasons.append("lateral_workspace_retarget_needed")
    return _stage(
        "approach",
        min(x_score, lateral_score),
        reasons,
        {
            "object_base_xyz": [x, y, z],
            "target_x": target_x,
            "x_error": x_error,
            "lateral_abs": lateral_abs,
        },
        threshold,
    )


def _eval_workspace(obj: WorldObject | None, support: WorldObject | None, threshold: float) -> dict[str, Any]:
    if obj is None or obj.pose_map is None:
        return _stage("workspace", 0.0, ["missing_object_map_pose"], {}, threshold)
    object_z = float(obj.pose_map.position[2])
    height_score = _score_range(object_z, soft_low=0.72, soft_high=1.05, hard_low=0.55, hard_high=1.25)
    reasons: list[str] = []
    metrics: dict[str, Any] = {"object_z": object_z}
    gap_score = 0.85
    if support is not None and support.pose_map is not None and support.shape.size is not None:
        support_top = float(support.pose_map.position[2]) + abs(float(support.shape.size[2])) * 0.5
        gap = object_z - support_top - _half_height(obj)
        gap_score = _score_range(gap, soft_low=-0.015, soft_high=0.060, hard_low=-0.050, hard_high=0.180)
        metrics.update({"support_top_z": support_top, "object_support_gap": gap})
        if gap_score < threshold:
            reasons.append("object_support_gap_suspicious")
    else:
        reasons.append("support_surface_unknown")
    if height_score < threshold:
        reasons.append("table_height_outside_comfort_workspace")
    return _stage("workspace", min(height_score, gap_score), reasons, metrics, threshold)


def _eval_grasp(obj: WorldObject | None, grasp: Any, threshold: float) -> dict[str, Any]:
    if obj is None:
        return _stage("grasp", 0.0, ["missing_task_object"], {}, threshold)
    if grasp is None:
        return _stage("grasp", 0.0, ["missing_grasp_affordance"], {}, threshold)
    radius = _radius(obj)
    diameter = radius * 2.0
    height = _height(obj)
    max_extent = _max_extent(obj)
    reasons: list[str] = []
    metrics = {
        "affordance": grasp.name,
        "radius": radius,
        "diameter": diameter,
        "height": height,
        "max_extent": max_extent,
    }
    if grasp.name == "single_hand_pinch":
        score = min(_score_range(radius, 0.025, 0.055, 0.012, 0.085), _score_upper(max_extent, 0.14, 0.22))
        if score < threshold:
            reasons.append("single_hand_pinch_geometry_mismatch")
    elif grasp.name == "side_grasp":
        score = min(_score_upper(radius, 0.075, 0.110), _score_upper(height, 0.22, 0.34))
        if score < threshold:
            reasons.append("side_grasp_geometry_mismatch")
    elif grasp.name == "top_grasp":
        aperture = finite_float(grasp.params.get("aperture"), diameter)
        score = min(_score_upper(aperture, 0.12, 0.17), _score_upper(height, 0.065, 0.13), _score_upper(max_extent, 0.20, 0.32))
        metrics["aperture"] = aperture
        if score < threshold:
            reasons.append("top_grasp_geometry_mismatch")
    elif grasp.name == "bimanual_clamp":
        score = min(_score_range(max_extent, 0.10, 0.34, 0.05, 0.48), _score_upper(height, 0.25, 0.40))
        if score < threshold:
            reasons.append("bimanual_clamp_geometry_mismatch")
    else:
        score = 0.35
        reasons.append(f"unsupported_grasp_affordance:{grasp.name}")
    return _stage("grasp", score, reasons, metrics, threshold)


def _eval_lift(obj: WorldObject | None, grasp: Any, physics: dict[str, Any] | None, threshold: float) -> dict[str, Any]:
    if obj is None:
        return _stage("lift", 0.0, ["missing_task_object"], {}, threshold)
    mass = _state_mass(physics, "object") if physics else None
    mass_score = 0.85 if mass is None else _score_upper(float(mass), soft=0.18, hard=0.75)
    size_score = _score_upper(_max_extent(obj), soft=0.34, hard=0.55)
    stable_score = 1.0 if not physics or physics.get("healthy") else 0.0
    reasons: list[str] = []
    if mass_score < threshold:
        reasons.append("object_mass_lift_risk")
    if size_score < threshold:
        reasons.append("object_size_lift_risk")
    if stable_score < threshold:
        reasons.append("initial_physics_unstable")
    return _stage(
        "lift",
        min(mass_score, size_score, stable_score),
        reasons,
        {"mass": mass, "max_extent": _max_extent(obj), "grasp": grasp.name if grasp else None},
        threshold,
    )


def _eval_place(task: Any, target: WorldObject | None, threshold: float) -> dict[str, Any]:
    if task.request.verb not in {"pick_place", "move", "place"}:
        return _stage("place", 1.0, [], {"applicable": False}, threshold, applicable=False)
    if target is None:
        return _stage("place", 0.0, ["missing_place_target"], {}, threshold)
    pose = target.pose_base or target.pose_map
    if pose is None:
        return _stage("place", 0.0, ["missing_place_pose"], {}, threshold)
    x, y, z = pose.position
    score = min(_score_upper(abs(float(y)), 0.45, 0.75), _score_range(float(z), 0.0, 1.10, -0.10, 1.40))
    reasons = [] if score >= threshold else ["place_target_reachability_risk"]
    return _stage("place", score, reasons, {"target_xyz": [x, y, z], "frame_id": pose.frame_id}, threshold)


def _eval_fall(physics: dict[str, Any] | None, threshold: float) -> dict[str, Any]:
    if physics is None:
        return _stage("fall", 1.0, [], {"applicable": False}, threshold, applicable=False)
    score = 1.0 if physics.get("healthy") and not physics.get("fallen") else 0.0
    reasons = [] if score >= threshold else ["robot_initial_fall_or_unhealthy_scene"]
    return _stage(
        "fall",
        score,
        reasons,
        {
            "base_z": _state_z(physics, "base"),
            "object_z": _state_z(physics, "object"),
            "contact_count": physics.get("contact_count"),
            "object_contact_count": physics.get("object_contact_count"),
        },
        threshold,
    )


def _stage(
    name: str,
    score: float,
    reasons: list[str],
    metrics: dict[str, Any],
    threshold: float,
    *,
    applicable: bool = True,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    if not applicable:
        status = "not_applicable"
        passed = True
    elif score >= threshold:
        status = "pass"
        passed = True
    elif score >= threshold * 0.75:
        status = "review"
        passed = False
    else:
        status = "fail"
        passed = False
    return {
        "name": name,
        "status": status,
        "score": score,
        "passed": passed,
        "reasons": reasons,
        "metrics": metrics,
    }


def _summary(rows: list[dict[str, Any]], *, suite_name: str, suite_version: str, suite_path: Path) -> dict[str, Any]:
    stage_names = ["approach", "workspace", "grasp", "lift", "place", "fall"]
    return {
        "suite": suite_name,
        "version": suite_version,
        "suite_path": _rel(suite_path),
        "task_count": len(rows),
        "success_count": sum(1 for row in rows if row["success"]),
        "plan_ready_count": sum(1 for row in rows if row["plan_ready"]),
        "stage_pass_count": {
            name: sum(1 for row in rows for stage in row["stages"] if stage["name"] == name and stage["passed"])
            for name in stage_names
        },
        "fail_stage_count": _counts(row["fail_stage"] or "none" for row in rows),
    }


def _print_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(f"primitive_microbench={summary['suite']}:{summary['version']} success={summary['success_count']}/{summary['task_count']}")
    print(f"{'task_id':28s} {'cat':10s} {'grasp':18s} {'app':>4s} {'wrk':>4s} {'grp':>4s} {'lft':>4s} {'fall':>4s} success fail_stage")
    for row in rows:
        scores = {stage["name"]: stage for stage in row["stages"]}
        print(
            f"{row['task_id'][:28]:28s} "
            f"{str(row.get('object_category') or '-')[:10]:10s} "
            f"{str(row.get('grasp_affordance') or '-')[:18]:18s} "
            f"{_score(scores['approach']):>4s} "
            f"{_score(scores['workspace']):>4s} "
            f"{_score(scores['grasp']):>4s} "
            f"{_score(scores['lift']):>4s} "
            f"{_score(scores['fall']):>4s} "
            f"{_yes(row['success']):7s} "
            f"{row['fail_stage'] or '-'}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "request",
        "object_category",
        "grasp_affordance",
        "plan_ready",
        "success",
        "fail_stage",
        "physics_healthy",
        "physics_fallen",
        "physics_base_z",
        "physics_object_z",
        "physics_object_mass",
        "approach_score",
        "workspace_score",
        "grasp_score",
        "lift_score",
        "place_score",
        "fall_score",
        "scene",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stages = {stage["name"]: stage for stage in row["stages"]}
            flat = dict(row)
            for name in ["approach", "workspace", "grasp", "lift", "place", "fall"]:
                flat[f"{name}_score"] = stages[name]["score"]
            writer.writerow({field: flat.get(field) for field in fields})


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['suite']}:{summary['version']} Primitive Microbench",
        "",
        f"- tasks: {summary['task_count']}",
        f"- success: {summary['success_count']}/{summary['task_count']}",
        f"- plan ready: {summary['plan_ready_count']}/{summary['task_count']}",
        f"- fail stages: {summary['fail_stage_count']}",
        "",
        "| task | category | grasp | approach | workspace | grasp score | lift | fall | success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stages = {stage["name"]: stage for stage in row["stages"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_id"],
                    str(row.get("object_category") or "-"),
                    str(row.get("grasp_affordance") or "-"),
                    _score(stages["approach"]),
                    _score(stages["workspace"]),
                    _score(stages["grasp"]),
                    _score(stages["lift"]),
                    _score(stages["fall"]),
                    _yes(row["success"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _radius(obj: WorldObject) -> float:
    if obj.shape.radius is not None:
        return float(obj.shape.radius)
    if obj.shape.size is None:
        return 0.045
    sx, sy, _ = obj.shape.size
    return max(0.005, min(abs(float(sx)), abs(float(sy))) * 0.5)


def _height(obj: WorldObject) -> float:
    if obj.shape.size is not None:
        return abs(float(obj.shape.size[2]))
    if obj.shape.radius is not None:
        return float(obj.shape.radius) * 2.0
    return 0.10


def _half_height(obj: WorldObject) -> float:
    return _height(obj) * 0.5


def _max_extent(obj: WorldObject) -> float:
    if obj.shape.size is not None:
        return max(abs(float(value)) for value in obj.shape.size)
    if obj.shape.radius is not None:
        return float(obj.shape.radius) * 2.0
    return 0.10


def _score_abs_error(error: float, *, soft: float, hard: float) -> float:
    error = abs(float(error))
    if error <= soft:
        return 1.0
    if error >= hard:
        return 0.0
    return 1.0 - (error - soft) / (hard - soft)


def _score_upper(value: float, soft: float, hard: float) -> float:
    value = float(value)
    if value <= soft:
        return 1.0
    if value >= hard:
        return 0.0
    return 1.0 - (value - soft) / (hard - soft)


def _score_range(value: float, soft_low: float, soft_high: float, hard_low: float, hard_high: float) -> float:
    value = float(value)
    if soft_low <= value <= soft_high:
        return 1.0
    if value < soft_low:
        if value <= hard_low:
            return 0.0
        return (value - hard_low) / max(1e-9, soft_low - hard_low)
    if value >= hard_high:
        return 0.0
    return (hard_high - value) / max(1e-9, hard_high - soft_high)


def _state_z(physics: dict[str, Any] | None, key: str) -> float | None:
    if not physics:
        return None
    state = physics.get(key) or {}
    position = state.get("position")
    return float(position[2]) if position else None


def _state_mass(physics: dict[str, Any] | None, key: str) -> float | None:
    if not physics:
        return None
    state = physics.get(key) or {}
    value = state.get("mass")
    return None if value is None else float(value)


def _score(stage: dict[str, Any]) -> str:
    if stage["status"] == "not_applicable":
        return "n/a"
    return f"{float(stage['score']):.2f}"


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


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


if __name__ == "__main__":
    raise SystemExit(main())
