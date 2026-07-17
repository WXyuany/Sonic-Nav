from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_curriculum_ab import compare_curricula


class CurriculumAbTests(unittest.TestCase):
    def test_requires_floor_and_non_regression(self) -> None:
        baseline = {"summary": {"trial_count": 5, "stage_success_rate": 0.6}}
        candidate = {"summary": {"trial_count": 5, "stage_success_rate": 0.8}}
        report = compare_curricula(baseline, candidate, min_stage_success_rate=0.6, max_baseline_regression=0.05)
        self.assertEqual("advance_to_sequence_eval", report["decision"])
        held = compare_curricula(baseline, {"summary": {"trial_count": 4, "stage_success_rate": 1.0}}, min_stage_success_rate=0.6, max_baseline_regression=0.05)
        self.assertEqual("hold", held["decision"])


if __name__ == "__main__":
    unittest.main()
