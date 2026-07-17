from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = SCRIPTS_DIR / "tools"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sonic_world.task_suites import load_robocasa_task_suite
from world_model_episode_materializer import REPO, _sequence_stages, materialize_episode


class EpisodeMaterializerTests(unittest.TestCase):
    def test_materializes_single_scene_carry_state_manifest(self) -> None:
        suite = load_robocasa_task_suite(
            REPO / "configs/world_model/task_suites/sonic_general_v0.yaml",
            repo_root=REPO,
        )
        stages = _sequence_stages(suite.tasks, "set_table_sequence")
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "episode.xml"
            manifest = materialize_episode(
                stages,
                sequence_id="set_table_sequence",
                scene_output=scene,
                suite_path="configs/world_model/task_suites/sonic_general_v0.yaml",
            )
            self.assertTrue(scene.exists())
            self.assertEqual("single_scene_carry_state", manifest["execution_mode"])
            self.assertEqual(3, manifest["stage_count"])
            self.assertEqual([1, 2, 3], [item["stage_index"] for item in manifest["stages"]])
            xml = scene.read_text(encoding="utf-8")
            for stage in manifest["stages"]:
                self.assertIn(stage["request"]["object_id"], xml)


if __name__ == "__main__":
    unittest.main()
