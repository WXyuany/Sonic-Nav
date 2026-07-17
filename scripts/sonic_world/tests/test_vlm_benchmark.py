from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_vlm_benchmark import _aggregate_metrics, _pair_records


class VlmBenchmarkTests(unittest.TestCase):
    def test_pairing_and_aggregate_include_unmatched_records(self) -> None:
        references = [
            {"sample_id": "frame-1", "objects": [{"object_id": "ball"}]},
            {"sample_id": "frame-2", "objects": [{"object_id": "ball"}]},
        ]
        predictions = [{"sample_id": "frame-1", "objects": [{"object_id": "ball"}]}]
        paired, missing_reference, missing_prediction = _pair_records(references, predictions)
        self.assertEqual(1, len(paired))
        self.assertEqual(1, len(missing_reference))
        self.assertEqual([], missing_prediction)
        metrics = _aggregate_metrics(
            [
                {
                    "reference_object_count": 1,
                    "prediction_object_count": 1,
                    "matched_object_count": 1,
                    "base_pose_error_median_m": 0.04,
                }
            ],
            missing_reference,
            missing_prediction,
        )
        self.assertEqual(2, metrics["reference_object_count"])
        self.assertEqual(1, metrics["prediction_object_count"])
        self.assertEqual(0.5, metrics["recall"])
        self.assertEqual(0.04, metrics["base_pose_error_median_m"])


if __name__ == "__main__":
    unittest.main()
