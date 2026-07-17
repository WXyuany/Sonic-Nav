from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_visual_transition_join import _attach_visual


class VisualTransitionJoinTests(unittest.TestCase):
    def test_attaches_nearest_qwen_pose_and_quality_features(self) -> None:
        transition = {
            "policy_stamp": 10.0,
            "policy": {"object_id": "mug_1"},
            "observation": {"entity": [[0.4, 0.0, 0.1] + [0.0] * 9, [0.0] * 12], "context": [0.0] * 24},
        }
        anchors = [
            {
                "recorded_at": 10.4,
                "sample_id": "shadow_1",
                "objects": [
                    {
                        "object_id": "mug_1",
                        "source": "vlm_anchor_backend",
                        "pose_base": {"position": [0.7, -0.2, 0.3]},
                        "properties": {
                            "confidence": 0.9,
                            "tracking_id": "track-mug-1",
                            "uncertainty": {"depth_mad_m": 0.01, "depth_sample_count": 49},
                        },
                    }
                ],
            }
        ]
        row, reason = _attach_visual(transition, anchors, max_skew_s=1.0)
        self.assertIsNone(reason)
        self.assertIsNotNone(row)
        self.assertEqual([0.7, -0.2, 0.3], row["observation"]["entity"][0][:3])
        self.assertEqual(0.9, row["observation"]["context"][12])
        self.assertEqual(1.0, row["observation"]["context"][13])
        self.assertEqual("shadow_1", row["visual_alignment"]["sample_id"])

    def test_rejects_stale_visual_anchor(self) -> None:
        transition = {"policy_stamp": 10.0, "policy": {"object_id": "mug_1"}, "observation": {"entity": [[0.0] * 12, [0.0] * 12], "context": [0.0] * 24}}
        anchors = [{"recorded_at": 15.0, "objects": [{"object_id": "mug_1", "pose_base": {"position": [0.0, 0.0, 0.0]}}]}]
        row, reason = _attach_visual(transition, anchors, max_skew_s=1.0)
        self.assertIsNone(row)
        self.assertEqual("stale_anchor", reason)


if __name__ == "__main__":
    unittest.main()
