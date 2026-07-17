#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sonic_world.task_suites import load_robocasa_task_suite


DEFAULT_SUITE = "configs/world_model/task_suites/molmospaces_robocasa_v0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headless MuJoCo health probe for Sonic/RoboCasa task scenes. "
            "This validates initial physics state without launching the GUI or ROS stack."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scene", help="MuJoCo XML scene path to probe directly.")
    source.add_argument("--suite", default=None, help="Task-suite YAML. Defaults to the MolmoSpaces/RoboCasa suite.")
    parser.add_argument("--task", help="Task id inside --suite. Defaults to the first task.")
    parser.add_argument("--all-tasks", action="store_true", help="Probe every selected task in the suite.")
    parser.add_argument("--limit", type=int, help="Limit --all-tasks to the first N tasks.")
    parser.add_argument("--object-body", help="Expected task object body name. Inferred from the suite or free joints by default.")
    parser.add_argument("--base-body", default="pelvis", help="Robot base body to monitor.")
    parser.add_argument("--steps", type=int, default=0, help="Raw uncontrolled mj_step count after the initial forward pass.")
    parser.add_argument("--fall-height", type=float, default=0.35, help="Base z below this is marked fallen.")
    parser.add_argument("--fall-angle-deg", type=float, default=45.0, help="Abs roll/pitch above this is marked fallen.")
    parser.add_argument("--max-contacts", type=int, default=12, help="Maximum contact pairs to include in JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument("--table", action="store_true", help="Print a compact table instead of JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scene:
        results = [
            probe_scene(
                args.scene,
                object_body=args.object_body,
                base_body=args.base_body,
                steps=args.steps,
                fall_height=args.fall_height,
                fall_angle_deg=args.fall_angle_deg,
                max_contacts=args.max_contacts,
            )
        ]
        report: dict[str, Any] = {"summary": _summary(results), "probes": results}
    else:
        suite_path = args.suite or DEFAULT_SUITE
        suite = load_robocasa_task_suite(suite_path, repo_root=REPO)
        tasks = list(suite.tasks)
        if args.task:
            wanted = set([args.task])
            tasks = [task for task in tasks if task.task_id in wanted]
            if not tasks:
                raise SystemExit(f"Unknown task id: {args.task}")
        elif not args.all_tasks:
            tasks = tasks[:1]
        if args.limit is not None:
            tasks = tasks[: max(0, args.limit)]
        if not tasks:
            raise SystemExit("No tasks selected.")
        results = [
            probe_scene(
                task.scene.scene_xml,
                task_id=task.task_id,
                object_body=args.object_body or task.request.object_id,
                object_id=task.request.object_id,
                base_body=args.base_body,
                steps=args.steps,
                fall_height=args.fall_height,
                fall_angle_deg=args.fall_angle_deg,
                max_contacts=args.max_contacts,
            )
            for task in tasks
        ]
        report = {
            "summary": {
                **_summary(results),
                "suite": suite.name,
                "suite_version": suite.version,
                "suite_path": _rel(_repo_path(suite_path)),
            },
            "probes": results,
        }

    if args.table:
        _print_table(results, report["summary"])
    else:
        print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + ("\n" if args.pretty else ""))
    return 0 if report["summary"]["healthy_count"] == report["summary"]["probe_count"] else 1


def probe_scene(
    scene_xml: str | Path,
    *,
    task_id: str | None = None,
    object_body: str | None = None,
    object_id: str | None = None,
    base_body: str = "pelvis",
    steps: int = 0,
    fall_height: float = 0.35,
    fall_angle_deg: float = 45.0,
    max_contacts: int = 12,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": task_id,
        "scene": str(scene_xml),
        "object_id": object_id or object_body,
        "steps": max(0, int(steps)),
        "scene_loaded": False,
        "healthy": False,
        "warnings": [],
        "error": "",
    }
    try:
        import mujoco
    except Exception as exc:
        result["error"] = f"mujoco import failed: {exc}"
        return result

    path = _repo_path(scene_xml)
    if not path.exists():
        result["error"] = f"scene XML not found: {path}"
        return result

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial_object_z: float | None = None
        base_id = _body_id(model, base_body)
        object_id_int = _body_id(model, object_body) if object_body else -1
        if object_id_int < 0:
            object_id_int = _infer_task_object_body(model, base_body=base_body, mujoco=mujoco)
        if object_id_int >= 0:
            initial_object_z = float(data.xpos[object_id_int][2])
        for _ in range(max(0, int(steps))):
            mujoco.mj_step(model, data)
        result.update(
            {
                "scene_loaded": True,
                "scene_path": _rel(path),
                "model": {
                    "nq": int(model.nq),
                    "nv": int(model.nv),
                    "nu": int(model.nu),
                    "nbody": int(model.nbody),
                    "ngeom": int(model.ngeom),
                    "njnt": int(model.njnt),
                    "timestep": float(model.opt.timestep),
                },
                "time": float(data.time),
                "free_joints": _free_joints(model, mujoco),
            }
        )
        base_state = _body_state(model, data, base_id, mujoco=mujoco)
        object_state = _body_state(model, data, object_id_int, mujoco=mujoco)
        contacts = _contacts(model, data, mujoco=mujoco, max_contacts=max_contacts)
        fall_angle_rad = math.radians(float(fall_angle_deg))
        fallen = _fallen(base_state, fall_height=fall_height, fall_angle_rad=fall_angle_rad)
        object_contact_count = _object_contact_count(model, data, object_id_int) if object_id_int >= 0 else 0
        if base_id < 0:
            result["warnings"].append(f"base_body_not_found:{base_body}")
        if object_id_int < 0:
            result["warnings"].append(f"object_body_not_found:{object_body or object_id or 'inferred'}")
        if object_state is not None and initial_object_z is not None:
            object_state["delta_z"] = float(object_state["position"][2] - initial_object_z)
        result.update(
            {
                "base": base_state,
                "object": object_state,
                "contact_count": int(data.ncon),
                "object_contact_count": int(object_contact_count),
                "contacts": contacts,
                "fallen": bool(fallen),
                "fall_threshold": {
                    "height": float(fall_height),
                    "angle_deg": float(fall_angle_deg),
                },
            }
        )
        result["healthy"] = bool(result["scene_loaded"] and base_state is not None and not fallen and object_state is not None)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _body_id(model: Any, name: str | None) -> int:
    if not name:
        return -1
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _infer_task_object_body(model: Any, *, base_body: str, mujoco: Any) -> int:
    candidates: list[tuple[int, str]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, mujoco=mujoco)
        if not name or name == base_body:
            continue
        candidates.append((body_id, name))
    for body_id, name in candidates:
        if "object" in name or name.endswith("_task"):
            return body_id
    return candidates[0][0] if candidates else -1


def _free_joints(model: Any, mujoco: Any) -> list[dict[str, Any]]:
    joints: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        joints.append(
            {
                "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, mujoco=mujoco),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, mujoco=mujoco),
                "qposadr": int(model.jnt_qposadr[joint_id]),
            }
        )
    return joints


def _body_state(model: Any, data: Any, body_id: int, *, mujoco: Any) -> dict[str, Any] | None:
    if body_id < 0:
        return None
    quat = [float(v) for v in data.xquat[body_id]]
    roll, pitch, yaw = _quat_to_euler(quat)
    return {
        "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, mujoco=mujoco),
        "mass": float(model.body_mass[body_id]),
        "position": [float(v) for v in data.xpos[body_id]],
        "quaternion_wxyz": quat,
        "rpy": [roll, pitch, yaw],
        "rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
    }


def _contacts(model: Any, data: Any, *, mujoco: Any, max_contacts: int) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for contact_index in range(min(int(data.ncon), max(0, int(max_contacts)))):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        contacts.append(
            {
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1, mujoco=mujoco) or f"geom#{geom1}",
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2, mujoco=mujoco) or f"geom#{geom2}",
                "body1": _name(model, mujoco.mjtObj.mjOBJ_BODY, body1, mujoco=mujoco) or f"body#{body1}",
                "body2": _name(model, mujoco.mjtObj.mjOBJ_BODY, body2, mujoco=mujoco) or f"body#{body2}",
                "distance": float(contact.dist),
                "position": [float(v) for v in contact.pos],
            }
        )
    return contacts


def _object_contact_count(model: Any, data: Any, body_id: int) -> int:
    geoms = {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id}
    return sum(1 for i in range(int(data.ncon)) if int(data.contact[i].geom1) in geoms or int(data.contact[i].geom2) in geoms)


def _fallen(base_state: dict[str, Any] | None, *, fall_height: float, fall_angle_rad: float) -> bool:
    if base_state is None:
        return True
    position = base_state["position"]
    roll, pitch, _yaw = base_state["rpy"]
    return bool(float(position[2]) < fall_height or abs(float(roll)) > fall_angle_rad or abs(float(pitch)) > fall_angle_rad)


def _quat_to_euler(quat_wxyz: list[float]) -> tuple[float, float, float]:
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = sum(1 for result in results if result.get("scene_loaded"))
    healthy = sum(1 for result in results if result.get("healthy"))
    fallen = sum(1 for result in results if result.get("fallen"))
    object_contact = sum(1 for result in results if int(result.get("object_contact_count") or 0) > 0)
    return {
        "probe_count": len(results),
        "scene_loaded_count": loaded,
        "healthy_count": healthy,
        "fallen_count": fallen,
        "object_contact_task_count": object_contact,
    }


def _print_table(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    suite = summary.get("suite")
    suffix = f" suite={suite}:{summary.get('suite_version')}" if suite else ""
    print(f"probes={summary['probe_count']} healthy={summary['healthy_count']}/{summary['probe_count']}{suffix}")
    print(f"{'task/scene':32s} {'load':5s} {'base_z':>7s} {'obj_z':>7s} {'con':>4s} {'objcon':>6s} {'fall':5s} healthy")
    for result in results:
        label = str(result.get("task_id") or Path(str(result.get("scene"))).stem)[:32]
        base = result.get("base") or {}
        obj = result.get("object") or {}
        base_z = _fmt_z(base)
        obj_z = _fmt_z(obj)
        print(
            f"{label:32s} "
            f"{_yes(bool(result.get('scene_loaded'))):5s} "
            f"{base_z:>7s} "
            f"{obj_z:>7s} "
            f"{int(result.get('contact_count') or 0):>4d} "
            f"{int(result.get('object_contact_count') or 0):>6d} "
            f"{_yes(bool(result.get('fallen'))):5s} "
            f"{_yes(bool(result.get('healthy')))}"
        )


def _fmt_z(state: dict[str, Any]) -> str:
    if not state:
        return "-"
    return f"{float(state['position'][2]):.3f}"


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _name(model: Any, obj_type: Any, obj_id: int, *, mujoco: Any) -> str | None:
    if obj_id < 0:
        return None
    return mujoco.mj_id2name(model, obj_type, int(obj_id))


def _yes(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
