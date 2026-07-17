from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_contact_diagnostic import summarize


class ContactDiagnosticTests(unittest.TestCase):
    def test_summarizes_success_and_failure_groups(self) -> None:
        report = summarize([
            {"passed": True, "contact_count": 4.0, "servo_ik_error": 0.1, "close_ratio": 0.7},
            {"passed": False, "contact_count": 0.0, "servo_ik_error": 0.3, "close_ratio": 0.4},
        ], skill="manip.side_grasp")
        self.assertEqual(2, report["summary"]["attempt_count"])
        self.assertEqual(0.5, report["summary"]["success_rate"])
        self.assertEqual(1, report["by_outcome"]["passed"]["count"])


if __name__ == "__main__":
    unittest.main()
