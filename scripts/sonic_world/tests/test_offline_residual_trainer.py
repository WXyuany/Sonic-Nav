from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[2] / "tools"
SCRIPTS = TOOLS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("offline_residual_trainer", TOOLS / "train_world_model_residual_offline.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfflineResidualTrainerTests(unittest.TestCase):
    def test_positive_effect_requires_success_and_effect_evidence(self) -> None:
        self.assertTrue(MODULE._positive_effect({"primitive": {"status": "success"}, "outcome": {"effect_passed": True}}))
        self.assertFalse(MODULE._positive_effect({"primitive": {"status": "failed"}, "outcome": {"effect_passed": True}}))
        self.assertFalse(MODULE._positive_effect({"primitive": {"status": "success"}, "outcome": {"effect_passed": False}}))


if __name__ == "__main__":
    unittest.main()
