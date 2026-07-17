from __future__ import annotations

import unittest

from sonic_world.world_model.visual_recovery import VisualRecoveryBudget


class VisualRecoveryBudgetTests(unittest.TestCase):
    def test_missing_object_is_rate_limited_then_exhausted(self) -> None:
        budget = VisualRecoveryBudget(["ball", "target"], max_attempts=2, cooldown_s=1.0)
        self.assertEqual({"ball"}, budget.observe({"target"}))
        self.assertEqual(1, budget.request("ball", now=10.0))
        self.assertIsNone(budget.request("ball", now=10.5))
        self.assertEqual(2, budget.request("ball", now=11.0))
        self.assertIsNone(budget.request("ball", now=12.0))

    def test_observation_resets_budget(self) -> None:
        budget = VisualRecoveryBudget(["ball"], max_attempts=1, cooldown_s=0.0)
        self.assertEqual(1, budget.request("ball", now=1.0))
        self.assertEqual(set(), budget.observe({"ball"}))
        self.assertEqual(1, budget.request("ball", now=2.0))


if __name__ == "__main__":
    unittest.main()
