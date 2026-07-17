# Sonic-Nav Data Repository

The companion repository is `WXyuany/Sonic-Nav-Data`. It is deliberately
separate from code so generated MuJoCo XML, raw episode logs, checkpoints, and
visual artifacts do not bloat the framework history.

Create it once, then attach it as a submodule:

```bash
gh auth login -h github.com
gh repo create WXyuany/Sonic-Nav-Data --private --clone
git -C Sonic-Nav-Data lfs install
git -C Sonic-Nav-Data lfs track "*.pt" "*.jsonl" "*.mp4" "*.png"
git -C Sonic-Nav-Data add .gitattributes README.md
git -C Sonic-Nav-Data commit -m "Initialize Sonic-Nav data store"
git -C Sonic-Nav-Data push -u origin main
git submodule add git@github.com:WXyuany/Sonic-Nav-Data.git external_data
```

Expected layout:

```text
checkpoints/       promoted and candidate .pt files
datasets/          training JSONL and task-suite exports
episodes/          compact episode logs and manifests
benchmarks/        leaderboard inputs and reports
perception/        Qwen/RGB-D shadow outputs and gate reports
manifests/         SHA256 inventory of every published artifact
```

Only publish reproducible inputs, compact terminal logs, approved checkpoints,
and benchmark summaries. Keep transient ROS logs, simulator caches, generated
scene XML, and teacher-assisted episodes out of leaderboard inputs. Every
checkpoint manifest must record its source dataset SHA256, training command,
policy scope, and visual deployment gate.

## Next Training Loop

1. Collect late-attach teacher episodes until `manip.side_grasp` has at least
   eight `effect_passed=true` rows whose `effect_source` is `mujoco_qpos`.
2. Build transitions with `world_model_episode_dataset.py`, then train a
   side-grasp-only AWR checkpoint with `--positive-effect-only --effect-source mujoco_qpos`.
3. Run a non-assisted, 20-trial physical curriculum evaluation. Teacher-assisted
   episodes must never enter the physical leaderboard.
4. Promote only if the task oracle, recovery rate, and baseline comparison pass
   `world_model_policy_promotion.py`; otherwise retain the candidate as shadow-only.
5. Export the selected dataset, checkpoint, evaluation summary, and SHA256
   manifest to `external_data/`, then tag both repositories with the same run id.
