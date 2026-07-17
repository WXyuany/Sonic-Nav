#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.scenarios import load_scenarios, replay_scenario


DEFAULT_SCENARIO_DIR = Path(REPO_ROOT) / "configs" / "world_model" / "scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Sonic world-model scenario JSON files through the task planner."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Scenario JSON files or directories. Defaults to configs/world_model/scenarios.",
    )
    parser.add_argument("--dump", action="store_true", help="Print full replay JSON instead of a compact summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or [str(DEFAULT_SCENARIO_DIR)]
    scenarios = load_scenarios(paths)
    if not scenarios:
        raise SystemExit(f"no scenarios found in: {paths}")

    replays = [replay_scenario(scenario) for scenario in scenarios]
    if args.dump:
        print(json.dumps([replay.to_dict() for replay in replays], indent=2, sort_keys=True))
    else:
        for replay in replays:
            status = "PASS" if replay.passed else "FAIL"
            task_names = ", ".join(task.task.name for task in replay.tasks)
            print(f"{status} {replay.scenario.name}: objects={list(replay.world_objects)} tasks=[{task_names}]")

    failed = [replay.scenario.name for replay in replays if not replay.passed]
    if failed:
        raise SystemExit(f"world_model_replay failed: {failed}")
    print(f"world_model_replay: ok ({len(replays)} scenario(s))")


if __name__ == "__main__":
    main()
