#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = Path(SCRIPTS_DIR).parent
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sonic_world.planners import WorldModelPipeline
from sonic_world.scenarios import ScenarioSpec, replay_scenario
from sonic_world.task_suites import load_robocasa_task_suite
from sonic_world.world_model import WorldMemory, anchor_to_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview RoboCasa task-suite anchors and world-model plans.")
    parser.add_argument("--suite", default="configs/world_model/task_suites/robocasa_v0.yaml")
    parser.add_argument("--task", help="Task id. Defaults to the first task in the suite.")
    parser.add_argument("--list", action="store_true", help="List task ids and scenes.")
    parser.add_argument("--dump-anchor", action="store_true")
    parser.add_argument("--dump-request", action="store_true")
    parser.add_argument("--dump-scenario", action="store_true")
    parser.add_argument("--dump-plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_robocasa_task_suite(args.suite, repo_root=REPO)
    if args.list:
        for task in suite.tasks:
            tags = ",".join(task.tags)
            print(f"{task.task_id:24s} scene={task.scene.scene_name:18s} request={task.request.verb:10s} tags={tags}")
        return 0

    task = suite.get_task(args.task) if args.task else suite.tasks[0]
    if args.dump_anchor:
        print(json.dumps(task.anchor(), indent=2, sort_keys=True))
        return 0
    if args.dump_request:
        print(json.dumps(task.request.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.dump_scenario:
        print(json.dumps(task.scenario(), indent=2, sort_keys=True))
        return 0

    scenario = ScenarioSpec.from_dict(task.scenario())
    replay = replay_scenario(scenario)
    result = replay.tasks[0].result
    if args.dump_plan:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    print(f"suite={suite.name}:{suite.version} task={task.task_id}")
    print(f"scene={task.scene.scene_name} xml={task.scene.scene_xml}")
    print(f"request={task.request.verb} object={task.request.object_id or task.request.object_category} target={task.request.target_id}")
    print(f"objects={list(result.world.objects)}")
    print(f"steps={[step.name for step in result.skill_graph.steps]}")
    print(
        "dispatch="
        f"unready:{result.dispatch_plan.metadata['unready_count']} "
        f"errors:{result.dispatch_plan.metadata['contract_error_count']} "
        f"decision:{result.decision_plan.status}"
    )
    print(f"passed_expectations={replay.passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
