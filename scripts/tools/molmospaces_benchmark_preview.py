#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.datasets import DEFAULT_MOLMOSPACES_BENCHMARK, MolmoSpacesBenchmark
from sonic_world.planners import WorldModelPipeline
from sonic_world.world_model import WorldMemory, anchor_to_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview MolmoSpaces benchmark episodes as Sonic world-model tasks."
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=str(DEFAULT_MOLMOSPACES_BENCHMARK),
        help="Path to a MolmoSpaces benchmark.json file or benchmark directory.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-kind", help="Filter --list-episodes by pick, pick_place, navigate, open_close.")
    parser.add_argument("--scene-dataset", help="Filter --list-episodes by scene dataset, e.g. holodeck-objaverse.")
    parser.add_argument("--house-index", type=int, help="Filter --list-episodes by house index.")
    parser.add_argument("--limit", type=int, default=10, help="Rows shown by --list-episodes.")
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--no-context", action="store_true", help="Only include task-relevant objects in the anchor.")
    parser.add_argument("--max-context-objects", type=int, default=40)
    parser.add_argument("--dump-anchor", action="store_true")
    parser.add_argument("--dump-request", action="store_true")
    parser.add_argument("--dump-plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = MolmoSpacesBenchmark(Path(args.benchmark))
    if args.list_episodes:
        episodes = benchmark.episodes(
            limit=args.limit,
            task_kind=args.task_kind,
            scene_dataset=args.scene_dataset,
            house_index=args.house_index,
        )
        for episode in episodes:
            pickup = episode.pickup_object_id or "-"
            place = episode.place_receptacle_id or "-"
            print(
                f"{episode.index:04d} kind={episode.task_kind:10s} scene={episode.scene_key:34s} "
                f"robot={episode.robot_name:12s} pickup={_clip(pickup, 34):34s} "
                f"target={_clip(place, 28):28s} text={_clip(episode.language_description, 64)}"
            )
        return 0

    episode = benchmark.episode(args.episode_index)
    anchor = benchmark.episode_anchor(
        episode,
        include_context=not args.no_context,
        max_context_objects=args.max_context_objects,
    )
    request = benchmark.episode_task_request(episode)
    if args.dump_anchor:
        print(json.dumps(anchor, indent=2, sort_keys=True))
        return 0
    if args.dump_request:
        print(json.dumps(request.to_dict(), indent=2, sort_keys=True))
        return 0

    pipeline = WorldModelPipeline(memory=WorldMemory(stale_after_s=0.0))
    world = pipeline.memory.update(anchor_to_world(anchor))
    result = pipeline.plan(world, request, kind="task_request", source="molmospaces_preview")
    if args.dump_plan:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    print(f"benchmark={args.benchmark}")
    print(f"episode={episode.index} id={episode.episode_id}")
    print(f"scene={episode.scene_key} split={episode.data_split} robot={episode.robot_name}")
    print(f"task_kind={episode.task_kind} task_cls={episode.task_cls}")
    print(f"language={episode.language_description}")
    print(f"request={request.verb} object={request.object_id} target={request.target_id}")
    print(f"world_objects={len(world.objects)} relations={len(world.relations)}")
    print(f"steps={[step.name for step in result.skill_graph.steps]}")
    next_handler = result.decision_plan.next_action.handler if result.decision_plan.next_action else None
    print(f"decision={result.decision_plan.status} next={next_handler}")
    if result.skill_graph.metadata.get("warnings"):
        print(f"warnings={result.skill_graph.metadata['warnings']}")
    return 0


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
