#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sonic_world.task_suites import load_robocasa_task_suite
from sonic_world.rollout_logging import add_rollout_log_args, logger_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a RoboCasa scene and print/persist the selected world-model task anchor."
    )
    parser.add_argument("task", nargs="?", default="tabletop_ball_to_tray")
    parser.add_argument("--suite", default="configs/world_model/task_suites/robocasa_v0.yaml")
    parser.add_argument("--launcher", choices=["start", "dwa", "mppi"], default="dwa")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument(
        "--anchor-out",
        default="/tmp/sonic_robocasa_task_anchor.json",
        help="Path where the selected generic anchor is written for publishing/replay.",
    )
    parser.add_argument(
        "--request-out",
        default="/tmp/sonic_robocasa_task_request.json",
        help="Path where the selected task request is written for publishing/replay.",
    )
    add_rollout_log_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_robocasa_task_suite(args.suite, repo_root=REPO)
    task = suite.get_task(args.task)
    anchor_path = Path(args.anchor_out)
    request_path = Path(args.request_out)
    anchor_path.write_text(json.dumps(task.anchor(), indent=2, sort_keys=True), encoding="utf-8")
    request_path.write_text(json.dumps(task.request.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    rollout = logger_from_args(
        args,
        demo_kind="robocasa",
        task_id=task.task_id,
        scene=task.scene.scene_xml or task.scene.scene_name,
        metadata={
            "suite": args.suite,
            "scene_name": task.scene.scene_name,
            "scene_xml": task.scene.scene_xml,
            "request": task.request.to_dict(),
            "anchor_path": str(anchor_path),
            "request_path": str(request_path),
        },
    )
    rollout.log_event(
        "task_context",
        status="ready",
        metadata={
            "description": task.description,
            "tags": list(task.tags),
            "objects": [obj.get("object_id") or obj.get("id") for obj in task.objects],
        },
    )
    print(f"[ROBOCASA_TASK] task={task.task_id} scene={task.scene.scene_name}")
    print(f"[ROBOCASA_TASK] anchor={anchor_path}")
    print(f"[ROBOCASA_TASK] request={request_path}")
    print(f"[ROBOCASA_TASK] rollout_log={rollout.path}")
    print("[ROBOCASA_TASK] publish later with:")
    print(
        "  "
        + " ".join(
            shlex.quote(str(part))
            for part in [
                "/usr/bin/python3",
                "scripts/tools/world_model_object_anchor.py",
                "--file",
                str(anchor_path),
            ]
        )
    )
    print(
        "  "
        + " ".join(
            shlex.quote(str(part))
            for part in [
                "/usr/bin/python3",
                "scripts/tools/world_model_task_request.py",
                "--file",
                str(request_path),
            ]
        )
    )
    launch_scene = task.scene.scene_xml or task.scene.scene_name
    if args.no_launch:
        rollout.log_event(
            "task_end",
            status="skipped",
            reason="no_launch",
            metadata={"launch_scene": launch_scene, "launcher": args.launcher},
        )
        rollout.close()
        print(f"[ROBOCASA_TASK] launch skipped; scene command: python scripts/start_{args.launcher}.py {launch_scene}")
        return 0
    launcher = {
        "start": SCRIPT_DIR / "start.py",
        "dwa": SCRIPT_DIR / "start_dwa.py",
        "mppi": SCRIPT_DIR / "start_mppi.py",
    }[args.launcher]
    rollout.log_event(
        "launch_exec",
        status="running",
        metadata={
            "launcher": str(launcher),
            "launch_scene": launch_scene,
            "argv": [sys.executable, str(launcher), launch_scene],
        },
    )
    rollout.close(status="exec")
    os.execv(sys.executable, [sys.executable, str(launcher), launch_scene])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
