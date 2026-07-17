from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_training_gate import evaluate_gate


class TrainingGateTests(unittest.TestCase):
    def test_requires_both_anchor_quality_and_data_volume(self) -> None:
        shadow = evaluate_gate({"gate": {"passed": False, "failed_checks": ["recall"]}}, {"visual_transition_count": 200}, min_visual_transitions=100)
        self.assertEqual("shadow_training_only", shadow["decision"])
        ready = evaluate_gate({"gate": {"passed": True}}, {"visual_transition_count": 100}, min_visual_transitions=100)
        self.assertEqual("eligible_for_ab", ready["decision"])


if __name__ == "__main__":
    unittest.main()
