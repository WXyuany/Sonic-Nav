from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sonic_world.skills import verify_backend_effects
from sonic_world.world_model import (
    apply_translation_offset,
    evaluate_anchor_pairs,
    gate_anchor_metrics,
    project_detection_with_depth,
    transform_point_pose,
    robust_translation_offset,
    translation_residual,
)
from sonic_world.world_model.vlm_gate import load_passing_vlm_gate
from sonic_world.world_model.temporal_anchor import TemporalAnchorFilter
from sonic_world.skills.mujoco_effect_observer import MujocoQposEffectObserver
from sonic_world.skills.primitive_family import primitive_profile


class EffectAndPerceptionTest(unittest.TestCase):
    def test_temporal_anchor_filter_uses_median_and_waits_for_track_history(self) -> None:
        filter_ = TemporalAnchorFilter(window_size=3, min_observations=3)
        positions = [[1.0, 0.0, 0.8], [1.6, 0.0, 0.8], [2.0, 0.0, 0.8]]
        for index, position in enumerate(positions):
            stable = filter_.update(
                {
                    "objects": [
                        {
                            "object_id": "ball_1",
                            "tracking_id": "ball_1",
                            "category": "ball",
                            "pose_base": {"frame_id": "base_link", "position": position},
                        }
                    ]
                }
            )
            self.assertEqual(len(stable["objects"]), 0 if index < 2 else 1)
        self.assertEqual(stable["objects"][0]["pose_base"]["position"], [1.6, 0.0, 0.8])
    def test_vlm_gate_requires_passing_evaluation_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "gate.json"
            report.write_text(
                json.dumps({"schema": "sonic_vlm_anchor_eval_report_v0", "gate": {"passed": True}}),
                encoding="utf-8",
            )
            self.assertTrue(load_passing_vlm_gate(report)["gate"]["passed"])
            report.write_text(
                json.dumps(
                    {"schema": "sonic_vlm_anchor_eval_report_v0", "gate": {"passed": False, "failed_checks": ["recall"]}},
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "did not pass"):
                load_passing_vlm_gate(report)

    def test_declared_effects_are_required(self) -> None:
        action = {"metadata": {"effects": ["object_contact_ready", "object_in_hand"]}}
        passed, reason, _metrics = verify_backend_effects(
            action,
            {
                "effect_evidence": {
                    "passed": True,
                    "effects": {"object_contact_ready": {"passed": True}},
                }
            },
        )
        self.assertFalse(passed)
        self.assertIn("object_in_hand", reason)

    def test_rgbd_projection_and_transform(self) -> None:
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        detection = project_detection_with_depth(
            {"category": "cup", "bbox": [8, 8, 12, 12]},
            depth,
            [100.0, 0.0, 10.0, 0.0, 100.0, 10.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(detection["pose_camera"]["position"], [0.0, 0.0, 2.0])
        base = transform_point_pose(
            detection["pose_camera"],
            {"translation": [1.0, 2.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            frame_id="base_link",
        )
        self.assertEqual(base["position"], [1.0, 2.0, 2.0])
        self.assertTrue(detection["tracking_id"].startswith("cup:"))

    def test_rgbd_projection_accepts_qwen_bbox_2d_and_label(self) -> None:
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        detection = project_detection_with_depth(
            {"label": "robot", "bbox_2d": [8, 8, 12, 12]},
            depth,
            [100.0, 0.0, 10.0, 0.0, 100.0, 10.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(detection["pose_camera"]["position"], [0.0, 0.0, 2.0])
        self.assertTrue(detection["tracking_id"].startswith("robot:"))

    def test_vlm_anchor_metrics_gate_pose_support_and_tracking(self) -> None:
        references = [
            {
                "objects": [
                    {
                        "object_id": "cup_1",
                        "category": "cup",
                        "support": "table",
                        "pose_base": {"position": [0.5, 0.0, 0.1]},
                    }
                ]
            }
        ]
        predictions = [
            {
                "objects": [
                    {
                        "object_id": "cup_1",
                        "category": "cup",
                        "support": "table",
                        "properties": {"tracking_id": "track-cup-1", "uncertainty": {"depth_mad_m": 0.003}},
                        "pose_base": {"position": [0.52, 0.0, 0.1]},
                    }
                ]
            }
        ]
        metrics = evaluate_anchor_pairs(references, predictions)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["base_pose_error_median_m"], 0.02)
        gate = gate_anchor_metrics(
            metrics,
            min_precision=0.9,
            min_recall=0.9,
            max_base_pose_error_m=0.03,
            min_support_accuracy=0.9,
            min_tracking_consistency=0.9,
            min_target_region_recall=0.0,
        )
        self.assertTrue(gate["passed"])
        rejected = gate_anchor_metrics(
            metrics,
            min_precision=0.9,
            min_recall=0.9,
            max_base_pose_error_m=0.01,
            min_support_accuracy=0.9,
            min_tracking_consistency=0.9,
            min_target_region_recall=0.0,
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("base_pose_error_median_m", rejected["failed_checks"])

    def test_visual_translation_calibration_is_robust_and_explicit(self) -> None:
        residual = translation_residual(
            {"position": [1.2, -0.2, 0.8]}, {"position": [1.0, -0.1, 0.7]}
        )
        self.assertIsNotNone(residual)
        self.assertAlmostEqual(0.2, residual[0])
        self.assertAlmostEqual(-0.1, residual[1])
        self.assertAlmostEqual(0.1, residual[2])
        offset = robust_translation_offset([residual, [0.2, -0.1, 0.1], [2.0, 2.0, 2.0]])
        self.assertIsNotNone(offset)
        self.assertAlmostEqual(0.2, offset[0])
        self.assertAlmostEqual(-0.1, offset[1])
        self.assertAlmostEqual(0.1, offset[2])
        corrected = apply_translation_offset({"frame_id": "base_link", "position": [1.0, -0.1, 0.7]}, offset)
        self.assertAlmostEqual(1.2, corrected["position"][0])
        self.assertAlmostEqual(-0.2, corrected["position"][1])
        self.assertAlmostEqual(0.8, corrected["position"][2])

    def test_navigation_and_workspace_effects_share_reachability_threshold(self) -> None:
        observer = object.__new__(MujocoQposEffectObserver)
        command = {"effects": ["robot_near_object", "object_in_hand_workspace"]}
        before = {"target_position": [1.0, 0.0, 0.0], "base_position": [0.0, 0.0, 0.0]}
        after = {"target_position": [1.0, 0.0, 0.0], "base_position": [0.21, 0.0, 0.0], "target_contact_count": 0}
        evidence = observer.evaluate(command, before, after)
        effects = evidence["effects"]
        self.assertEqual(0.78, effects["robot_near_object"]["threshold_m"])
        self.assertEqual(0.78, effects["object_in_hand_workspace"]["threshold_m"])

    def test_lift_profile_biases_contact_retention(self) -> None:
        profile = primitive_profile("manip.lift_object")
        self.assertIsNotNone(profile)
        self.assertGreater(profile.runtime_overrides["squeeze_close_ratio"], 0.65)
        self.assertGreater(profile.runtime_overrides["hold_close_ratio"], 0.76)
        self.assertLess(profile.runtime_overrides["servo_lift_z_lead"], 0.05)


if __name__ == "__main__":
    unittest.main()
