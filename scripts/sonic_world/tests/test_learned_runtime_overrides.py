from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sonic_world.policies.hybrid_backend import _apply_residual, _manipulation_only, _skill_runtime_overrides


class LearnedRuntimeOverrideTests(unittest.TestCase):
    def test_hybrid_overrides_are_bounded(self) -> None:
        overrides = _skill_runtime_overrides([0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 10.0, 0.0])
        self.assertEqual(0.025, overrides["manip.side_grasp"]["contact_x_delta_m"])
        self.assertEqual(-0.015, overrides["manip.side_grasp"]["contact_z_delta_m"])
        self.assertEqual(0.08, overrides["manip.lift_object"]["squeeze_close_ratio"])
        self.assertEqual(0.0, overrides["manip.lift_object"]["lift_z"])
        high_lift = _skill_runtime_overrides([0.0] * 7 + [10.0])
        self.assertEqual(0.08, high_lift["manip.lift_object"]["lift_z"])

    def test_lift_recovery_does_not_overwrite_executor_owned_recovery_profile(self) -> None:
        overrides = _skill_runtime_overrides([0.0] * 8, recovery_context={"failed_skill": "manip.lift_object", "reason": "failed effects: object_in_hand", "attempt": 1})
        self.assertNotIn("close_ratio", overrides["manip.side_grasp"])
        self.assertEqual(0.0, overrides["manip.lift_object"]["squeeze_close_ratio"])

    def test_manipulation_only_checkpoint_does_not_change_base_goal(self) -> None:
        payload = {
            "base_goal": {"position": [1.0, 2.0, 0.0], "yaw": 0.1},
            "grasp_offsets": {"contact_offset": [0.1, 0.0, 0.2]},
        }
        self.assertTrue(_manipulation_only({"manip.lift_object"}))
        _apply_residual(payload, [1.0] * 8, "continue", allow_base_residual=False, allow_grasp_residual=False, allow_recovery_override=False)
        self.assertEqual([1.0, 2.0, 0.0], payload["base_goal"]["position"])
        self.assertEqual(0.1, payload["base_goal"]["yaw"])
        self.assertEqual([0.1, 0.0, 0.2], payload["grasp_offsets"]["contact_offset"])

    def test_lift_only_checkpoint_cannot_emit_grasp_overrides(self) -> None:
        overrides = _skill_runtime_overrides([1.0] * 8, training_skills={"manip.lift_object"})
        self.assertNotIn("manip.side_grasp", overrides)
        self.assertIn("manip.lift_object", overrides)


if __name__ == "__main__":
    unittest.main()
