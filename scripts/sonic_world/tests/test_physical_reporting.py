from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from world_model_physical_leaderboard import _paths, _row
from world_model_policy_promotion import _episodes


class PhysicalReportingTests(unittest.TestCase):
    def test_discovers_curriculum_logs_by_event_contract_not_prefix(self) -> None:
        events = [
            {"schema": "sonic_world_model_episode_event_v0", "event": "stage_start", "stamp": 1.0, "sequence_id": "stage_1"},
            {"schema": "sonic_world_model_episode_event_v0", "event": "stage_terminal", "stamp": 2.0, "status": "succeeded"},
            {"schema": "sonic_world_model_episode_event_v0", "event": "episode_terminal", "stamp": 3.0, "status": "succeeded", "episode_scope": "curriculum_stage"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "curriculum_stage1_trial01.jsonl"
            path.parent.mkdir()
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            discovered = _paths([directory])
            self.assertEqual([path], discovered)
            self.assertEqual("succeeded", _row(path)["final_status"])
            self.assertFalse(_row(path)["physical_sequence_success"])
            self.assertEqual(1, len(_episodes([directory])))


if __name__ == "__main__":
    unittest.main()
