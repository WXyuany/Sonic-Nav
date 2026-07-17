from __future__ import annotations

from types import SimpleNamespace
import unittest

from sonic_world.task_suites import evaluate_task_success


def _world(object_position, *, stable=True, on_support=True):
    relations = []
    if on_support:
        relations.append({"subject": "obj", "relation": "on", "object": "table", "confidence": 1.0})
    return {
        "frame_id": "map",
        "robot_pose_map": [0.0, 0.0, 0.8],
        "robot": {"stable": stable},
        "objects": [
            {"object_id": "obj", "category": "ball", "shape": {"kind": "sphere", "radius": 0.04}, "pose_map": object_position},
            {"object_id": "goal", "category": "place_target", "shape": "target", "pose_map": [1.0, 0.0, 0.8], "support": "table"},
            {"object_id": "table", "category": "table", "shape": {"kind": "box", "size": [1, 1, 0.7]}, "pose_map": [1.0, 0.0, 0.4]},
        ],
        "relations": relations,
    }


class TaskOracleTest(unittest.TestCase):
    def test_move_requires_lift_and_stable_target(self) -> None:
        task = SimpleNamespace(
            task_id="move_obj",
            request=SimpleNamespace(verb="move", object_id="obj", target_id="goal"),
            anchor=lambda: _world([0.5, 0.0, 0.8]),
        )
        history = [
            _world([0.5, 0.0, 0.8]),
            _world([0.7, 0.0, 0.9]),
            _world([1.0, 0.0, 0.8]),
            _world([1.0, 0.0, 0.8]),
            _world([1.0, 0.0, 0.8]),
        ]
        result = evaluate_task_success(task, final_world=history[-1], world_history=history)
        self.assertTrue(result.success, result.to_dict())

    def test_move_rejects_phase_only_success(self) -> None:
        task = SimpleNamespace(
            task_id="move_obj",
            request=SimpleNamespace(verb="move", object_id="obj", target_id="goal"),
            anchor=lambda: _world([0.5, 0.0, 0.8]),
        )
        final = _world([1.0, 0.0, 0.8])
        result = evaluate_task_success(
            task,
            final_world=final,
            world_history=[final, final, final],
            rollout_summary={"final_status": "success"},
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
