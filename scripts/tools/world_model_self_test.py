#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sonic_world.planners import TaskPlanner, TaskRequest, WorldModelPipeline, task_request_from_json
from sonic_world.scenarios import ScenarioSpec, load_scenarios, replay_scenario
from sonic_world.task_suites import load_robocasa_task_suite
from sonic_world.skills import (
    SkillExecutionMonitor,
    dispatch_plan_for_graph,
    phase_to_skill_index,
    runtime_plan_for_graph,
)
from sonic_world.world_model import WorldMemory, anchor_to_world
from world_model_preview import SAMPLE_BALL_ANCHOR, SAMPLE_BOX_ANCHOR, SAMPLE_NAV_ANCHOR


def _steps(graph) -> list[str]:
    return [step.name for step in graph.steps]


def main() -> None:
    planner = TaskPlanner()

    ball_world = anchor_to_world(SAMPLE_BALL_ANCHOR)
    ball_graph = planner.plan(ball_world, TaskRequest(verb="pick_place"))
    assert ball_graph.metadata["grasp_affordance"] == "single_hand_pinch"
    assert _steps(ball_graph) == [
        "navigate.approach_object",
        "manip.align_workspace",
        "manip.single_hand_pinch",
        "manip.lift_object",
        "manip.transport_object",
        "manip.place_object",
        "manip.release",
    ]
    ball_runtime = runtime_plan_for_graph(ball_graph)
    ball_dispatch = dispatch_plan_for_graph(ball_graph, ball_runtime, ball_world)
    ball_phase_index = phase_to_skill_index(ball_runtime)
    assert ball_runtime.metadata["missing_skills"] == []
    assert ball_dispatch.metadata["unready_count"] == 0
    assert ball_dispatch.metadata["contract_error_count"] == 0
    assert ball_dispatch.steps[0].handler == "demo_locomotion_phase_runtime"
    assert ball_dispatch.steps[0].contract.ready
    assert ball_dispatch.steps[2].handler == "contact_grasp_primitive"
    assert ball_dispatch.steps[2].capability == "contact_grasp"
    assert ball_dispatch.steps[2].contract.ready
    assert ball_phase_index["walk_to_table"] == "navigate.approach_object"
    assert ball_phase_index["capture_ball_contact"] == "manip.single_hand_pinch"
    assert ball_phase_index["release_ball"] == "manip.release"
    ball_monitor = SkillExecutionMonitor()
    planned = ball_monitor.set_runtime(ball_runtime)
    assert planned.status == "planned"
    running = ball_monitor.update_phase("capture_ball_contact")
    assert running.status == "running"
    assert running.current_skill == "manip.single_hand_pinch"
    assert "retry_capture" in running.recovery_options
    assert "navigate.approach_object" in running.completed_skills
    done = ball_monitor.update_phase("demo done")
    assert done.status == "succeeded"
    assert done.completed_skills[-1] == "manip.release"

    box_world = anchor_to_world(SAMPLE_BOX_ANCHOR)
    box_graph = planner.plan(box_world, TaskRequest(verb="pick"))
    assert box_graph.metadata["grasp_affordance"] == "bimanual_clamp"
    assert _steps(box_graph) == [
        "navigate.approach_object",
        "manip.align_workspace",
        "manip.bimanual_clamp",
        "manip.lift_object",
    ]
    box_runtime = runtime_plan_for_graph(box_graph)
    box_dispatch = dispatch_plan_for_graph(box_graph, box_runtime, box_world)
    box_phase_index = phase_to_skill_index(box_runtime)
    assert box_runtime.metadata["missing_skills"] == []
    assert box_dispatch.metadata["unready_count"] == 0
    assert box_dispatch.metadata["contract_error_count"] == 0
    assert box_dispatch.steps[2].handler == "contact_grasp_primitive"
    assert box_dispatch.steps[2].contract.ready
    assert box_phase_index["walk_two_steps"] == "navigate.approach_object"
    assert box_phase_index["forearm_clamp_box"] == "manip.bimanual_clamp"
    assert box_phase_index["lift_box_from_table"] == "manip.lift_object"

    nav_world = anchor_to_world(SAMPLE_NAV_ANCHOR)
    nav_graph = planner.plan(nav_world, TaskRequest(verb="navigate"))
    assert nav_graph.metadata["task_template"] == "navigation"
    assert nav_graph.metadata["object_category"] == "navigation_goal"
    assert _steps(nav_graph) == ["navigate.goto"]
    nav_runtime = runtime_plan_for_graph(nav_graph)
    nav_dispatch = dispatch_plan_for_graph(nav_graph, nav_runtime, nav_world)
    nav_phase_index = phase_to_skill_index(nav_runtime)
    assert nav_runtime.demo_kind == "navigation"
    assert nav_runtime.metadata["missing_skills"] == []
    assert nav_dispatch.steps[0].handler == "ros2_goal_pose"
    assert nav_dispatch.metadata["contract_error_count"] == 0
    assert nav_dispatch.steps[0].contract.ready
    assert nav_dispatch.steps[0].command["topic"] == "/goal_pose"
    assert nav_phase_index["navigate_to_goal"] == "navigate.goto"
    nav_monitor = SkillExecutionMonitor()
    nav_monitor.set_runtime(nav_runtime)
    reached = nav_monitor.update_status_text("state=reached cmd_v=0.000 goal_dist=0.00")
    assert reached.status == "succeeded"
    assert reached.current_skill == "navigate.goto"

    memory = WorldMemory(stale_after_s=0.0)
    memory.update(anchor_to_world(SAMPLE_BOX_ANCHOR))
    memory.update(anchor_to_world(SAMPLE_BALL_ANCHOR))
    vlm_request = task_request_from_json(
        '{"task":"move","object":"demo_ball","target":"place_target","id":"sample-vla-1"}'
    )
    assert vlm_request.metadata["request_id"] == "sample-vla-1"
    memory_graph = planner.plan(memory.current(), vlm_request)
    memory_runtime = runtime_plan_for_graph(memory_graph)
    memory_dispatch = dispatch_plan_for_graph(memory_graph, memory_runtime, memory.current())
    assert memory_graph.metadata["task_template"] == "pick_place"
    assert memory_graph.metadata["object_category"] == "ball"
    assert _steps(memory_graph)[2] == "manip.single_hand_pinch"
    assert memory_dispatch.metadata["unready_count"] == 0

    pipeline = WorldModelPipeline(memory=WorldMemory(stale_after_s=0.0))
    default_ball = pipeline.observe_anchor(SAMPLE_BALL_ANCHOR)
    assert default_ball.kind == "ball"
    assert default_ball.skill_graph.metadata["task_template"] == "pick_place"
    assert default_ball.dispatch_plan.metadata["unready_count"] == 0
    assert default_ball.recovery_plan.status == "not_needed"
    assert default_ball.decision_plan.status == "ready_to_execute"
    assert default_ball.decision_plan.next_action.kind == "dispatch"
    request_result = pipeline.plan_current(vlm_request, kind="task_request", source="self_test")
    assert request_result.source == "self_test"
    assert request_result.request.metadata["request_id"] == "sample-vla-1"
    assert request_result.dispatch_plan.steps[2].handler == "contact_grasp_primitive"
    assert request_result.decision_plan.status == "ready_to_execute"
    assert request_result.decision_plan.next_action.handler == "demo_locomotion_phase_runtime"

    scenario_dir = Path(REPO_ROOT) / "configs" / "world_model" / "scenarios"
    scenario_replays = [replay_scenario(scenario) for scenario in load_scenarios([scenario_dir])]
    assert scenario_replays
    for replay in scenario_replays:
        assert replay.passed, replay.to_dict()
    missing_base_replay = next(replay for replay in scenario_replays if replay.scenario.name == "generic_contract_missing_base")
    missing_base_task = missing_base_replay.tasks[0]
    missing_base_dispatch = missing_base_task.result.dispatch_plan
    missing_base_recovery = missing_base_task.result.recovery_plan
    assert "publish_object_anchor_with_pose_base" in missing_base_dispatch.metadata["recovery_suggestions"]
    assert "reobserve_from_current_view" in missing_base_dispatch.steps[1].contract.recovery_suggestions
    assert missing_base_recovery.status == "needs_recovery"
    assert len(missing_base_recovery.actions) == 3
    assert missing_base_recovery.actions[0].handler == "object_anchor_update"
    assert missing_base_task.result.decision_plan.status == "needs_recovery"
    assert missing_base_task.result.decision_plan.next_action.kind == "recovery"
    assert missing_base_task.result.decision_plan.next_action.handler == "object_anchor_update"
    assert missing_base_task.recovery_result is not None
    assert missing_base_task.recovery_result.dispatch_plan.metadata["contract_error_count"] == 0
    assert missing_base_task.recovery_result.recovery_plan.status == "not_needed"
    assert missing_base_task.recovery_result.decision_plan.status == "ready_to_execute"
    assert missing_base_task.recovery_result.decision_plan.next_action.kind == "dispatch"

    robocasa_suite = load_robocasa_task_suite(repo_root=Path(REPO_ROOT))
    assert len(robocasa_suite.tasks) >= 5
    for task in robocasa_suite.tasks:
        replay = replay_scenario(ScenarioSpec.from_dict(task.scenario()))
        assert replay.passed, replay.to_dict()
        result = replay.tasks[0].result
        assert result.dispatch_plan.metadata["contract_error_count"] == 0
        assert result.decision_plan.status == "ready_to_execute"

    print("world_model_self_test: ok")


if __name__ == "__main__":
    main()
