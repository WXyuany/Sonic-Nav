from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from world_model_contact_sweep import _grid


class ContactSweepTest(unittest.TestCase):
    def test_grid_clamps_and_preserves_cartesian_product(self) -> None:
        grid = _grid("0.1,1.2", "-0.1,0.0", "-0.1,0.1", "-1.0,1.0")
        self.assertEqual(len(grid), 16)
        self.assertEqual(grid[0], {"close_ratio": 0.2, "contact_x_delta_m": -0.025, "contact_z_delta_m": -0.015, "grasp_wrist_pitch": -0.45})
        self.assertEqual(grid[-1], {"close_ratio": 0.95, "contact_x_delta_m": 0.0, "contact_z_delta_m": 0.015, "grasp_wrist_pitch": 0.2})


if __name__ == "__main__":
    unittest.main()
