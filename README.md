# Sonic-Nav

ROS2 Navigation Stack for GR00T Whole-Body Control (Unitree G1 humanoid robot).

Based on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

## Reproducible World-Model Setup

This repository contains the executable framework: MuJoCo/SONIC scenes,
world-model planning and dispatch, RGB-D/VLM anchor interfaces, recovery,
skill-level PPO, physical episode logging, and CI. Generated rollouts,
training JSONL, and model checkpoints belong in the companion data repository
`WXyuany/Sonic-Nav-Data`, mounted at `external_data/`.

```bash
git clone --recurse-submodules git@github.com:WXyuany/Sonic-Nav.git
cd Sonic-Nav
git lfs install
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements-rl.txt
source /opt/ros/humble/setup.bash
make world-model-tests
make world-model-ci
```

The full ROS/MuJoCo rollout requires the SONIC deployment assets and the
Unitree G1 policy files used by the upstream project. The CPU-only checks above
validate schemas, planner/executor contracts, policy bounds, task oracle, and
the headless MuJoCo probe without starting a robot stack.

To reproduce a physical stage-isolated rollout after installing SONIC:

```bash
python3 scripts/tools/world_model_curriculum_batch.py \
  --sequence set_table_sequence --demo ball --stages 1 --trials-per-stage 3 \
  --output-dir reports/curriculum/repro_stage1 \
  --policy-backend learned \
  --policy-model external_data/checkpoints/world_model_hybrid_ppo_lift_positive_aw_v2.pt \
  --rollout-arg=--world-runtime-override-file \
  --rollout-arg=configs/world_model/contact_profiles/stage1_center_contact_v1.json
```

`reports/` is intentionally ignored by the code repository. See
[`DATA_REPOSITORY.md`](DATA_REPOSITORY.md) for the data layout, integrity rules,
and the next training/evaluation sequence.

---

## Quick Start

```bash
# Terminal 1 — One-click launch
python scripts/start.py          # basic go-to-point
python scripts/start_mppi.py     # MPPI collision avoidance
python scripts/start_dwa.py      # DWA local planner
python scripts/start_box_demo.py box_demo  # table-top box grasp demo
python scripts/start_ball_demo.py ball_demo # single-hand ball pick-place demo
python scripts/start_molmospaces_demo.py --episode-index 0 # MolmoSpaces task-source scene

# Terminal 2 — RViz
bash scripts/rviz.sh
```

Click **2D Goal Pose** in RViz to navigate. Ctrl+C to stop.

## Features

| Mode | Script | Description |
|------|--------|-------------|
| Go-to-Point | `start.py` | Smooth turning, proportional control |
| MPPI Nav | `start_mppi.py` | GPU trajectory sampling + collision avoidance |
| DWA Nav | `start_dwa.py` | Dynamic window local planning |
| Box Demo | `start_box_demo.py` | Vision-anchor table approach + upper-body contact grasp |
| Ball Demo | `start_ball_demo.py` | Right-hand tabletop ball pick-and-place |
| MolmoSpaces Demo | `start_molmospaces_demo.py` | MolmoSpaces episode adapter for world-model tasks |
| Keyboard | `scripts/tools/keyboard_control.py` | WASD manual control |

## Scenes

```bash
bash scripts/switch_scene.sh <name>
bash scripts/switch_scene.sh --list
python scripts/start.py indoor               # launch directly in the indoor scene
python scripts/start.py robocasa_kitchen     # launch the RoboCasa kitchen scene
python scripts/start_mppi.py robocasa_galley # launch MPPI in a tight kitchen scene
python scripts/start_dwa.py robocasa_cafe    # launch DWA in the cafe scene
```

| Scene | Description |
|-------|-------------|
| `default` | 8m×8m room, cylinder obstacles |
| `box_demo` | Table-top lightweight box grasp demo scene |
| `ball_demo` | Table-top light ball pick-and-place scene |
| `dynamic` | Moving obstacles (sliding + rotating) |
| `stairs` | 10-step staircase + ramp |
| `uneven` | Bumpy terrain + rocks |
| `table` | Table and small-object interaction area |
| `indoor` | Detailed office/lab indoor navigation floor |
| `robocasa_kitchen` | Doorless 14m x 9m RoboCasa-style kitchen/living scene with real fixture meshes |
| `robocasa_galley` | Narrow dual-counter RoboCasa kitchen for tight-lane navigation |
| `robocasa_apartment` | Multi-zone apartment with partial walls, doorways, kitchen, dining, and living areas |
| `robocasa_cafe` | Small cafe with counter, stools, tables, queue posts, and display fridge |

Restart sim after switching.

## Box Grasp Demo

The box demo is a lightweight VLM-ready interaction pipeline: the simulator publishes a known box anchor in map/base/camera frames, the robot walks to the table using the anchor distance, then upper-body WBC/IK tracks the box and performs a contact grasp.

```bash
python scripts/start_box_demo.py box_demo
```

By default, the demo does **not** use the old visible box suction helper. The terminal should print:

```text
box attach assist disabled; using contact/friction grasp only
```

Useful options:

```bash
python scripts/start_box_demo.py box_demo --box-attach          # explicit debug fallback
python scripts/start_box_demo.py box_demo --no-hold             # release final pose; stack stays running
python scripts/start_box_demo.py box_demo --no-box-anchor       # run fixed-pose fallback
```

Key topics:

| Topic | Type | Description |
|-------|------|-------------|
| `/sonic_demo/box_anchor` | `std_msgs/String` | JSON anchor with box pose, size, camera point, and grasp plan |
| `/sonic_demo/box_pose` | `geometry_msgs/PoseStamped` | Box center in map frame |
| `/sonic_demo/box_grasp_base_pose` | `geometry_msgs/PoseStamped` | Suggested base target near the table |
| `/sonic_demo/phase` | `std_msgs/String` | Current demo phase for RViz/debugging |

During grasp, `box_grasp_demo.py` retries approach until the box is close enough, filters implausible startup anchors, checks whether the box has lifted, and then tightens the hands with a small `squeeze_box_secure` phase.

## Ball Pick-Place Demo

The ball demo follows the same anchor-driven path as the box demo, but uses a single right-hand IK sequence: walk to the table, approach from above, close on the small ball, lift it, move to the target marker on the other side of the table, release, and retreat.

The manipulation logic is factored through `scripts/manipulation/wam_primitives.py`: object anchors are treated as a lightweight world model, `WorkspaceAligner` keeps the object inside the reachable hand workspace with base micro-adjustments, and `ContactServoPolicy` uses fingertip contact error to adjust hand targets, close ratio, and lift lead online. This is the intended bridge toward VLA/VLM policies: the model can provide object/goal anchors while the classical primitive layer handles stability, contact, and retry logic.

```bash
python scripts/start_ball_demo.py ball_demo
```

Default behavior uses contact/friction only:

```text
ball contact-lock assist disabled; using contact/friction grasp only
```

Useful options:

```bash
python scripts/start_ball_demo.py ball_demo --ball-attach       # explicit debug fallback
python scripts/start_ball_demo.py ball_demo --no-hold           # release final pose; stack stays running
python scripts/start_ball_demo.py ball_demo --no-ball-anchor    # run fixed-pose fallback
```

Key topics:

| Topic | Type | Description |
|-------|------|-------------|
| `/sonic_demo/ball_anchor` | `std_msgs/String` | JSON anchor with ball pose, camera point, place target, and pick plan |
| `/sonic_demo/ball_pose` | `geometry_msgs/PoseStamped` | Ball center in map frame |
| `/sonic_demo/ball_place_pose` | `geometry_msgs/PoseStamped` | Fixed place target on the table |
| `/sonic_demo/phase` | `std_msgs/String` | Current demo phase for RViz/debugging |

During the pick, the terminal reports workspace alignment residuals, contact servo errors, grasp geometry, and lift detection. Those logs are useful for telling whether the failure is perception/anchor drift, workspace reachability, IK, or physical contact.

## World Model Framework

The box and ball demos now feed a shared lightweight world-model layer. Anchors from perception or known simulation objects are normalized into `WorldObject` records with shape, map/base/camera poses, support relations, and affordances. A task planner then turns the current world state into a reusable skill graph.

All entry points use `scripts/sonic_world/planners/pipeline.py`, which owns the shared flow from anchor/task request to world memory, skill graph, runtime plan, dispatch plan, recovery plan, and final decision plan.

```bash
python scripts/tools/world_model_preview.py --sample nav
python scripts/tools/world_model_preview.py --sample ball --verb pick_place
python scripts/tools/world_model_preview.py --sample box --verb pick
python scripts/tools/world_model_preview.py --sample ball --request-json '{"task":"move","object":"demo_ball","target":"place_target"}'
```

RoboCasa is the preferred physical-interaction backend for the VLA line. MolmoSpaces is kept as a task and world-model data source: it packages indoor assets and object metadata as benchmark episodes with explicit scene dataset, robot base pose, object poses, camera specs, task fields, and language. The intended split is:

```text
MolmoSpaces benchmark -> scene/task/object/camera/language state
World Model           -> normalized objects, relations, affordances, tasks
RoboCasa/MuJoCo+SONIC -> authoritative G1 physics/control/contact
```

Clone MolmoSpaces locally, then preview its benchmark episodes through the Sonic world-model pipeline:

```bash
git clone --depth 1 https://github.com/allenai/molmospaces.git external_dependencies/molmospaces-src
python3 scripts/tools/molmospaces_benchmark_preview.py --list-episodes --limit 5
python3 scripts/tools/molmospaces_benchmark_preview.py --episode-index 0
python3 scripts/tools/molmospaces_benchmark_preview.py --episode-index 0 --dump-anchor
python3 scripts/tools/molmospaces_benchmark_preview.py --episode-index 0 --dump-plan
```

The adapter reads either a single `benchmark.json` file or a benchmark directory, then emits the same generic `objects[]` anchor contract used by the box/ball/RoboCasa demos. Pick tasks become `pick`, pick-and-place tasks become `pick_place`, and navigation-to-object tasks become `navigate`. Object shape, category, grasp affordance, base-relative pose, support relation, robot start pose, and language referral expressions are preserved in the anchor metadata.

To inspect a MolmoSpaces episode in the Sonic MuJoCo GUI, generate and launch an episode-derived scene:

```bash
python3 scripts/start_molmospaces_demo.py --episode-index 0
python3 scripts/start_molmospaces_demo.py --episode-index 0 --no-launch
python3 scripts/tools/molmospaces_scene_builder.py --episode-index 0 --scene-mode real --validate
```

The first real-scene run downloads the selected MolmoSpaces scene archive and the object/texture archives referenced by that scene, so it can take a while. These files are cached under `MLSPACES_CACHE_DIR` and symlinked under `MLSPACES_ASSETS_DIR`; later runs are much faster. The builder writes a generated XML such as `gear_sonic/data/robot_model/model_data/g1/scene_molmospaces_real_*.xml`, plus `/tmp/sonic_molmospaces_anchor.json` and `/tmp/sonic_molmospaces_task_request.json`. After the GUI stack is up, publish the exact world-model inputs with:

Real scenes are transformed into G1's local robot frame for XY/yaw while preserving the MolmoSpaces floor height by default. Sonic adds a transparent collision floor at `z=0` so G1 does not fall through visual-only imported geometry. If a scene still looks slightly high or low, tune only the visual room height with `--real-z-shift`; use `--real-z-align` only when you explicitly want to subtract MolmoSpaces `robot_base_pose.z`.

```bash
python3 scripts/start_molmospaces_demo.py --episode-index 0 --real-z-shift -0.04
/usr/bin/python3 scripts/tools/world_model_object_anchor.py --file /tmp/sonic_molmospaces_anchor.json
/usr/bin/python3 scripts/tools/world_model_task_request.py --file /tmp/sonic_molmospaces_task_request.json
```

For quick control debugging without downloading real assets, explicitly request the simplified local proxy:

```bash
python3 scripts/start_molmospaces_demo.py --episode-index 0 --scene-mode proxy
```

The proxy is intentionally fake and should only be used for fast control/anchor testing. The default route is still the real MolmoSpaces scene, but RoboCasa remains the main backend for polished manipulation tasks.

When you want to inspect or instantiate the actual MolmoSpaces assets, install the upstream package and point it at a local asset cache:

```bash
export MLSPACES_CACHE_DIR=~/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=$PWD/MolmoSpacesAssets
export MLSPACES_FORCE_INSTALL=True
cd external_dependencies/molmospaces-src
python3 -m pip install -e ".[mujoco]"
python3 -m molmo_spaces.molmo_spaces_constants
python3 - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
PY
python3 -m mujoco.viewer --mjcf "$MLSPACES_ASSETS_DIR/scenes/ithor/FloorPlan1_physics.xml"
```

The current Sonic integration uses the benchmark JSON first, because that is the stable world-model contract. The next step is task instantiation: convert selected MolmoSpaces episodes into RoboCasa/MuJoCo interaction tasks, preserving language, object anchors, support relations, grasp hints, and success conditions.

Replay task scenarios from JSON:

```bash
python scripts/tools/world_model_replay.py
python scripts/tools/world_model_replay.py configs/world_model/scenarios/mixed_tabletop_tasks.json --dump
```

Run a ROS-level smoke test without starting MuJoCo/SONIC:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/tools/world_model_ros_smoke_test.py
```

Publish a live task request after a demo anchor or RViz goal has populated world memory:

```bash
python scripts/tools/world_model_task_request.py --task move --object demo_ball --target place_target
```

Optionally inspect decision/executor events without taking control:

```bash
python scripts/tools/world_model_executor.py
python scripts/tools/world_model_executor.py --dispatch-topic /sonic_world/dispatch_plan  # also observe legacy dispatch steps
```

`world_model_executor.py --execute-navigation` can publish `ros2_goal_pose` decision actions to `/goal_pose`, but only for plans sourced from `/sonic_world/task_request`; this avoids echoing RViz goals back into the planner.

When `start.py`, `start_dwa.py`, `start_mppi.py`, `start_box_demo.py`, or `start_ball_demo.py` runs, `scripts/tools/world_model_node.py`, `scripts/tools/world_model_recovery_coordinator.py`, and `scripts/tools/world_model_executor.py` are launched automatically and publish:

| Topic | Type | Description |
|-------|------|-------------|
| `/sonic_world/model` | `std_msgs/String` | Normalized world state JSON from the latest object anchor |
| `/sonic_world/skill_graph` | `std_msgs/String` | Planned skill graph such as approach, align, grasp, lift, place |
| `/sonic_world/runtime_plan` | `std_msgs/String` | Mapping from skill graph nodes to the current demo's low-level phases |
| `/sonic_world/dispatch_plan` | `std_msgs/String` | Execution-handler contract for each skill, including command type and target topic |
| `/sonic_world/recovery_plan` | `std_msgs/String` | Structured recovery actions when dispatch contracts are not executable yet |
| `/sonic_world/decision_plan` | `std_msgs/String` | The current next action: execute a ready dispatch step or run a recovery action first |
| `/sonic_world/execution_state` | `std_msgs/String` | Current skill/phase status, completed skills, and recovery options |
| `/sonic_world/executor_event` | `std_msgs/String` | Optional executor observations/actions from `world_model_executor.py` |
| `/sonic_world/recovery_request` | `std_msgs/String` | Machine-readable recovery command emitted by `world_model_executor.py` |
| `/sonic_world/recovery_status` | `std_msgs/String` | Route acknowledgement from `world_model_recovery_coordinator.py` |
| `/sonic_world/perception_recovery_request` | `std_msgs/String` | Recovery requests routed to perception/world-memory backends |
| `/sonic_world/navigation_recovery_request` | `std_msgs/String` | Recovery requests routed to navigation/base-adjust backends |
| `/sonic_world/runtime_recovery_request` | `std_msgs/String` | Recovery requests routed to runtime/phase binding backends |
| `/sonic_world/manual_recovery_request` | `std_msgs/String` | Recovery requests that need manual or VLA-level review |
| `/sonic_world/active_task` | `std_msgs/String` | Latest task request used to build the skill graph |
| `/sonic_world/object_anchor` | `std_msgs/String` | Input generic object anchor JSON from perception/VLM |
| `/sonic_world/task_request` | `std_msgs/String` | Input JSON request from a VLM/VLA or scripted task source |
| `/sonic_demo/skill_graph` | `std_msgs/String` | Demo-local skill graph published by the active manipulation script |
| `/sonic_demo/runtime_plan` | `std_msgs/String` | Demo-local skill-to-phase binding used for monitoring and recovery |

The node listens to both object anchors and RViz navigation goals:

| Input | World Object | Default Task Template |
|-------|--------------|-----------------------|
| `/goal_pose` | `navigation_goal` | `navigation`: `navigate.goto` |
| `/sonic_demo/box_anchor` | `box` | `pick`: approach, align, bimanual clamp, lift |
| `/sonic_demo/ball_anchor` | `ball` + `place_target` | `pick_place`: approach, align, pinch, lift, transport, place, release |
| `/sonic_world/object_anchor` | generic `objects[]` | `pick` or `pick_place`, inferred from objects and place targets |
| `/sonic_world/task_request` | request over current world memory | request-selected template |

Execution monitoring is intentionally separate from low-level control. `world_model_node.py` watches `/sonic_demo/phase`, `/sonic_nav/dwa/status`, `/sonic_nav/mppi/status`, and `/sonic_nav/metrics_summary`, then publishes `/sonic_world/execution_state` with the active skill, completed skills, remaining skills, and recovery candidates from the runtime plan.

Task request JSON accepts strict or VLM-friendly field names:

```json
{"verb":"pick_place","object_id":"demo_ball","target_id":"place_target"}
{"task":"move","object":"demo_ball","target":"place_target"}
{"action":"navigate","goal":"rviz_goal"}
```

Explicit task requests are sticky. After `/sonic_world/task_request` or an RViz goal sets the active task, later relevant object anchors update world memory and produce an `anchor_replan` candidate for inspection. Candidate plans never preempt an active executor session. A failed contract routes to its recovery backend, which publishes the repaired anchor and then explicitly requests `runtime_replan`; that is the only automatic path that starts a fresh dispatch plan for the active task.

Scenario JSON in `configs/world_model/scenarios/` uses the same contract: a list of perception-style anchors, one or more task requests, and optional expectations for skill order, handlers, runtime bindings, dispatch contract errors, recovery plans, and readiness. A task may also include a `recovery` block with follow-up anchors and a second expectation; `world_model_replay.py` applies those anchors to world memory and replans the same task, so the regression set can prove closed-loop recovery instead of only proving the initial failure. This gives the framework a small regression set that is closer to future VLM/VLA input than one-off hard-coded demos.

Anchors can still use the old demo-specific fields (`box_center_map`, `ball_center_map`, `goal_center_map`), but the world model also accepts generic VLM-style objects:

```json
{
  "scene": "generic_tabletop",
  "frame_id": "map",
  "objects": [
    {
      "object_id": "red_fruit",
      "category": "fruit",
      "shape": {"kind": "sphere", "radius": 0.04},
      "pose_map": {"frame_id": "map", "position": [1.5, -0.22, 0.84]},
      "pose_base": {"frame_id": "base_link", "position": [0.5, -0.22, 0.03]},
      "support": "table",
      "grasp": {"base_target_map": [1.0, -0.16, 0.0], "approach_target_x": 0.54}
    },
    {
      "object_id": "right_tray",
      "category": "place_target",
      "shape": "target",
      "pose_map": {"frame_id": "map", "position": [1.5, 0.1, 0.84]}
    }
  ]
}
```

For generic anchors, the affordance library now covers the first useful object families: sphere-like and very small objects get `single_hand_pinch`, box-like objects get `bimanual_clamp`, cylindrical objects such as bottles/cans get `side_grasp`, flat/thin objects get `top_grasp`, support surfaces get `support_surface`, and explicit affordances may still be supplied by the perception/VLM layer when a shape rule is not enough. `configs/world_model/scenarios/generic_cylinder_affordance.json` is the regression case for a bottle-style object selecting `side_grasp`.

RoboCasa is the main physical-interaction scene source for the VLA line. `configs/world_model/task_suites/robocasa_v0.yaml` defines the first standard task suite: ball-to-tray, box clamp-and-lift, bottle-to-tray, walk-to-table fruit pick, and mug counter-to-table. These tasks generate generic object anchors, support relations, task requests, expected skill graphs, and dispatch contracts without hard-coding a new Python demo for each object.

```bash
python scripts/tools/robocasa_task_preview.py --list
python scripts/tools/robocasa_task_preview.py --task bottle_to_tray
python scripts/tools/robocasa_task_preview.py --task mug_counter_to_table --dump-plan
python scripts/start_robocasa_task.py bottle_to_tray --no-launch
```

`start_robocasa_task.py` writes the selected anchor and task request to `/tmp/sonic_robocasa_task_anchor.json` and `/tmp/sonic_robocasa_task_request.json`, then launches the selected RoboCasa scene unless `--no-launch` is set. With a running world-model node, publish the generated task inputs through:

```bash
/usr/bin/python3 scripts/tools/world_model_object_anchor.py --file /tmp/sonic_robocasa_task_anchor.json
/usr/bin/python3 scripts/tools/world_model_task_request.py --file /tmp/sonic_robocasa_task_request.json
```

MolmoSpaces episodes can be converted into a benchmark suite with matching RoboCasa/Sonic MuJoCo scenes. The generator reads MolmoSpaces language, task kind, object category, shape, and grasp hints, then writes a task-suite YAML plus one `scene_ms_*.xml` per task. Each generated scene starts from an existing RoboCasa-style scene and injects a reachable support surface, dynamic task object, and optional place target.

```bash
python3 scripts/tools/molmospaces_robocasa_benchmark.py --limit 8 --overwrite --validate --validate-xml
python3 scripts/tools/generate_sonic_task_suite.py --overwrite
python3 scripts/tools/robocasa_task_preview.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml --list
python3 scripts/tools/robocasa_task_preview.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml --task ms_ep0000_cup_pick --dump-plan
python3 scripts/tools/robocasa_task_preview.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --list
python3 scripts/tools/benchmark_runner.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml
python3 scripts/tools/benchmark_runner.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --name sonic_general_v0
python3 scripts/tools/headless_mujoco_probe.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml --all-tasks --table
python3 scripts/tools/headless_mujoco_probe.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --all-tasks --table
python3 scripts/tools/benchmark_runner.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml --headless-probe
python3 scripts/tools/primitive_microbench.py --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml
python3 scripts/tools/rollout_report.py reports/rollouts
python3 scripts/start_robocasa_task.py ms_ep0000_cup_pick --suite configs/world_model/task_suites/molmospaces_robocasa_v0.yaml --no-launch
```

This is the benchmark path: MolmoSpaces supplies one task distribution, RoboCasa/MuJoCo supplies executable physics scenes, and the world-model planner supplies the common anchor/task/skill interface. `sonic_general_v0.yaml` is the current unified in-repo suite for fast iteration. It now contains 500 generated tasks and 500 matching MuJoCo scenes. The suite covers short tabletop manipulation, dense tabletop pick/place, cluttered tabletop tasks with distractor objects, navigation-conditioned manipulation, cross-scene generalization, and long-sequence stages grouped by `sequence_id`. The current coverage includes 7 short seed tasks, 12 dense seed tasks, 180 tabletop-grid tasks, 128 clutter/distractor tasks, 101 navigation-conditioned tasks, 98 cross-scene tasks, 72 long-sequence tasks, and 52 generated sequence stages; these tags intentionally overlap. Generated tabletop scenes keep the support surface center stable but place the task object and target in the front reachable band, so the benchmark geometry remains physically table-like while staying inside G1's reachable workspace. The grasp-affordance distribution is 125 `single_hand_pinch`, 117 `side_grasp`, 138 `top_grasp`, and 120 `bimanual_clamp`, spanning balls/fruit, bottles/cups/cans/mugs, plates/bowls/books/cloth/sponge/tools, and boxes/packages. `generate_sonic_task_suite.py` writes both the suite YAML and one MuJoCo XML per task, so the same task list can be checked by the planner and by headless physics. `benchmark_runner.py` is the fast offline leaderboard: it checks scene XML validity, skill graph generation, runtime bindings, dispatch contracts, and expectation matches, then writes JSON/CSV/Markdown reports under `reports/benchmarks/`. Add `--headless-probe` when you also want a no-GUI MuJoCo health pass for base height, object pose, contacts, and fall checks. `primitive_microbench.py` is the next filter: it scores approach, workspace, grasp, lift, place, and fall preconditions without launching the GUI, so bad task geometry can be rejected before expensive controller rollout. The same generated suite can be evaluated in privileged-anchor mode first, then later in visual-perception mode when a VLM/VLA is producing anchors from camera/depth.

Real rollout entry points write JSONL logs under `reports/rollouts/` by default. Box and ball demos record phase starts/ends, retries, anchor updates, workspace observations, grasp/lift checks, and final task status with a shared `primitive_stage` label such as `approach`, `workspace`, `grasp`, `lift`, `transport`, or `place`. `start_robocasa_task.py` records the selected task context before launch, so RoboCasa task runs can be joined with later controller logs by `run_id`. Use `--rollout-log`, `--rollout-id`, or `--no-rollout-log` on these entry points when you need a fixed path, a shared id, or no local logging.

For repeated real-controller smoke tests, use the batch wrapper instead of launching each demo manually. By default it reuses one simulator/deploy stack, keeps the world state rolling across runs, assigns the next run ids automatically, and writes the summary at the end. If the reused scene reaches a terminal anchor state, such as the ball falling below the tabletop workspace, the batch runner requests an in-process MuJoCo scene reset instead of restarting the whole stack. Add `--headless --no-camera` for faster no-GUI controller rollouts, or `--restart-each` only when you need a clean initial scene per rollout:

```bash
python3 scripts/tools/rollout_batch.py ball --runs 5
python3 scripts/tools/rollout_batch.py ball --runs 20 --headless --no-camera --reset-each-rollout
python3 scripts/tools/rollout_batch.py ball --runs 20 --headless --no-camera --no-reset-on-anchor-loss
python3 scripts/tools/rollout_batch.py ball --runs 5 --restart-each
python3 scripts/tools/rollout_batch.py box --runs 3
python3 scripts/tools/rollout_batch.py ball --runs 2 -- --servo-contact-ready-error 0.058
```

`--headless` still runs MuJoCo physics; it only disables the GUI. `--reset-each-rollout` resets MuJoCo robot/object state between runs while keeping the simulator, deploy model, sensors, and world-model nodes alive. Use this for benchmark data collection when you want clean per-episode samples without paying the deploy startup cost. Use `--fail-on-rollout-fail` when a CI job should return non-zero for any failed task. Without that flag, failed rollouts remain in the JSONL/CSV report while the batch keeps collecting the remaining runs.

For the general benchmark suite, prefer the suite-level runner. It reads `configs/world_model/task_suites/sonic_general_v0.yaml`, selects each task's generated RoboCasa/MuJoCo scene, passes the correct generated object geom/site names to the anchor publisher, and then calls `rollout_batch.py` so repeated runs of the same task reuse one SONIC deploy stack:

```bash
python3 scripts/tools/task_suite_rollout.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --limit 2 --runs-per-task 1 --dry-run
python3 scripts/tools/task_suite_rollout.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --demo ball --limit 3 --runs-per-task 5 --headless --reset-each-rollout
python3 scripts/tools/task_suite_rollout.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --task ball_left_to_tray --runs-per-task 20 --headless --reset-each-rollout
python3 scripts/tools/task_batch_report.py reports/task_batches --rollouts reports/rollouts
```

Generated suite rollouts use the world model as an initial global prior, then switch back to live anchors for closed-loop skill execution. The runner passes a generated-scene profile into the box/ball demos: stale transient-local anchor messages are filtered by timestamp, old demo walk padding is disabled, and the first approach step can fall back from unstable start-time base coordinates to `*_center_map` plus the robot start pose. This keeps benchmark startup deterministic while preserving live base/object anchors for approach retry, workspace alignment, grasp, lift, and place diagnostics.

Use `--continue-state` only when intentionally collecting long-horizon, non-reset trajectories. For most benchmark data, keep the default reset behavior: the deploy model stays loaded, but robot/object state resets between rollout episodes.

Sequence benchmark and ranking are explicit. The current `sequence_id` groups are ordered multi-stage benchmark episodes; the sequence runner evaluates every stage's world-model plan, dispatch contract, and optional physics health, then ranks whole-sequence completion separately from stage completion:

```bash
python3 scripts/tools/world_model_sequence_benchmark.py \
  --suite configs/world_model/task_suites/sonic_general_v0.yaml \
  --headless-probe --name sonic_sequence_v0 --strict
```

The report writes JSON, CSV, and Markdown under `reports/benchmarks/`. Its score is an ordered planning/contract benchmark: stage scenes are independent, so it must not be presented as continuous physical manipulation.

For a physical carry-state episode, materialize a `sequence_id` into one MuJoCo XML and manifest. The generator keeps all stage objects and targets in one scene, validates it with MuJoCo, and rewrites resource paths so the artifact can also live outside the source scene directory:

```bash
python3 scripts/tools/world_model_episode_materializer.py \
  --sequence set_table_sequence \
  --scene-output-dir /tmp/sonic_episode_scenes \
  --manifest-output-dir /tmp/sonic_episode_manifests
```

Start the simulator, world-model node, recovery backends, primitive runner, and executor once against that generated scene. Then use the persistent episode client rather than restarting a demo for each stage:

```bash
python3 scripts/tools/world_model_episode_anchor.py \
  --scene /tmp/sonic_episode_scenes/scene_sonic_episode_set_table_sequence.xml \
  --manifest /tmp/sonic_episode_manifests/set_table_sequence.json

python3 scripts/tools/world_model_autonomous_episode.py \
  --manifest /tmp/sonic_episode_manifests/set_table_sequence.json \
  --output-jsonl reports/episodes/set_table_sequence.jsonl
```

Run the primitive runner with `--prefer-object-anchor` in this mode so each dispatch consumes the active qpos-backed anchor instead of a fixed demo anchor. The episode client advances only on an executor `plan_terminal=succeeded` event. A failure is recorded as a failed stage and stops the episode by default; `--continue-on-failure` is for recovery diagnostics, not official success scoring.

`rollout_batch.py` can launch the complete headless carry-state stack in one command. It starts no fixed demo anchor in episode mode, starts the qpos-backed episode anchor, uses effect evidence, and leaves MuJoCo state intact between stages:

```bash
python3 scripts/tools/rollout_batch.py ball \
  --scene /tmp/sonic_episode_scenes/scene_sonic_episode_set_table_sequence.xml \
  --episode-manifest /tmp/sonic_episode_manifests/set_table_sequence.json \
  --episode-output-jsonl reports/episodes/set_table_sequence.jsonl \
  --headless --no-camera --fail-on-rollout-fail

python3 scripts/tools/world_model_physical_leaderboard.py \
  --input reports/episodes --name sonic_physical_episode_latest
```

Training happens above the frozen SONIC control layer. SONIC is treated as a stable low-level whole-body controller: do not train locomotion, WBC, raw joint-space actions, or the SONIC deployment policy in this benchmark line. The trainable policy can still make the robot move by choosing skills and task-space primitive commands such as base goals, hand poses, wrist targets, close ratios, grasp offsets, lift/place targets, and recovery adjustments. First collect privileged-anchor rollouts and train/evaluate a VLA-style policy that predicts task intent, object/target anchors, skill selection, primitive parameters, and recovery decisions. Then move from privileged anchors to RGB-D/VLM-produced anchors. PPO/ERL-style methods, if used, should optimize these task/skill-level decisions and primitive parameters, not replace SONIC locomotion/control.

The residual RL route uses RSL-RL PPO from `requirements-rl.txt`. Its policy is a `MLP(256,256) -> GRU(128)` actor-critic over 48 world-model features, with eight bounded task-space residual actions and a five-way recovery action. Train it only against the vectorized pure-MuJoCo residual environment; use the ROS/SONIC stack for held-out physical evaluation and videos.

Every physical carry-state episode can also become a residual-learning dataset. The episode client records the PPO entity/context observation, residual action, effect oracle result, reward, and recovery handler; `world_model_episode_dataset.py` emits only terminal primitive transitions, so phase updates are never mislabeled as success. The promotion gate remains conservative: it requires multiple physical episodes, sequence and stage-effect success, a bounded recovery rate, and no material regression against the baseline before a checkpoint may become the default backend.

```bash
python3 scripts/tools/world_model_episode_dataset.py \
  --input reports/episodes/set_table_sequence_hybrid_ppo_v5.jsonl \
  --output reports/policy_data/physical_episode_residual_features_v0.jsonl

python3 scripts/tools/world_model_policy_promotion.py \
  --candidate reports/episodes/set_table_sequence_hybrid_ppo_v4.jsonl \
  --candidate reports/episodes/set_table_sequence_hybrid_ppo_v5.jsonl \
  --baseline reports/episodes/set_table_sequence_physical_v3.jsonl
```

The policy dataset boundary is explicit. `policy_dataset_builder.py` converts any task suite into JSONL samples whose observation contains the world-model objects, relations, sensor contract, available skills, and dispatch contract; the action label contains the trainable high-level outputs: task intent, object/target anchors, skill selection, base goal, hand pose target, wrist target, grasp close ratio, grasp offsets, lift/place targets, recovery decision, and ordered skill commands. The default teacher is a geometry/affordance heuristic, used as a baseline and sanity check before replacing it with a learned VLA/RL policy:

```bash
python3 scripts/tools/policy_dataset_builder.py --suite configs/world_model/task_suites/sonic_general_v0.yaml
python3 scripts/tools/policy_dataset_builder.py --suite configs/world_model/task_suites/sonic_general_v0.yaml --include-planning --print-sample
```

This JSONL is the first training artifact for the "brain": it does not contain low-level joint targets, but it does contain body-moving task-space commands that SONIC can execute through skill handlers. Real rollout logs then provide outcome labels and corrections for the same skill stages, so later training can learn when to adjust standoff, hand height, close ratio, target offsets, or recovery decisions.

At runtime, `world_model_node.py` publishes the same high-level action contract on `/sonic_world/policy_action`. The current publisher uses `heuristic_task_skill_policy_v0` as the teacher/baseline; a learned policy can replace that module while keeping the downstream SONIC executors unchanged.

Use the readiness check as the main closed-loop sanity gate. It runs the offline benchmark, regenerates the 500-task teacher policy, joins real rollout outcomes, builds the rollout episode dataset, aggregates feedback, writes a feedback-adjusted policy, trains the lightweight policy-memory baseline, exports runnable policy actions, and writes a compact report under `reports/readiness/`:

```bash
python3 scripts/tools/training_readiness_check.py --offline-limit 30
python3 scripts/tools/training_readiness_check.py --offline-limit 0
```

`--offline-limit 30` is the fast smoke path. `--offline-limit 0` checks the full 500-task suite. A healthy report should keep `offline` at 100%, keep teacher policy samples equal to the suite size, and increase `exact rollout-covered tasks` as more real task-suite rollouts are collected. If `top_issues` is dominated by `missing_or_implausible_anchor`, fix perception/state reset first. If it is dominated by `approach_failed`, `workspace_alignment_residual`, `capture_contact_not_ready`, or `lift_delta_below_threshold`, the data is already in the right stage-level form for task/skill policy training.

Use the selector to choose the next real rollout batch from coverage gaps instead of repeatedly running the first tasks in the YAML. It reads the latest policy outcomes, prefers uncovered task ids, balances ball/box demos by default, stratifies by grasp affordance and category, and prints a ready-to-run `task_suite_rollout.py` command:

```bash
python3 scripts/tools/task_rollout_selector.py --count 20 --tag wm_next20
python3 scripts/tools/task_rollout_selector.py --count 12 --tag wm_next12_real --runs-per-task 1
```

The selector writes JSON/CSV plans under `reports/task_selection/`. The reported `coverage` is real suite-task coverage, not generic demo fallback coverage. Generic `ball_demo` or `box_demo` rollouts remain useful for fallback policy memory, but they should not be counted as proof that a specific suite task has been evaluated.

Join policy samples with real rollout outcomes before training. This turns each run into `observation + teacher_action + outcome`, including dense score, quality label, terminal stage, retry/failure counts, and correction targets for the trainable high-level outputs:

```bash
python3 scripts/tools/policy_outcome_joiner.py \
  --policy-jsonl reports/policy_data/sonic_general_v0_heuristic.jsonl \
  --rollouts reports/rollouts
python3 scripts/tools/policy_feedback_report.py \
  --input reports/policy_outcomes/sonic_policy_outcomes.jsonl
python3 scripts/tools/policy_apply_feedback.py \
  --policy-jsonl reports/policy_data/sonic_general_v0_heuristic.jsonl \
  --feedback reports/policy_outcomes/sonic_feedback_profile.json
```

`policy_feedback_report.py` aggregates failure modes into a feedback profile such as "approach still far -> adjust base_goal/standoff", "capture contact not ready -> adjust hand_pose/wrist/close_ratio", or "missing anchor -> recovery/perception first". This is the bridge from benchmark logs to policy improvement: clean successes become strong positive samples, rough successes become low-weight positives with correction targets, and failures become negative/recovery samples.

`policy_apply_feedback.py` writes a separate feedback-adjusted policy JSONL for A/B testing. It applies only task/skill-level deltas, such as standoff, hand contact point, wrist pitch, close ratio, and lift target. It does not write joint targets and does not alter SONIC. In rollout application, `safe` mode is intentionally conservative and keeps approach standoff fixed unless explicitly enabled; use `full` mode or a dedicated batch when evaluating standoff/contact/wrist changes.

Train the first task-level baseline after joining outcomes. `task_policy_train.py` currently writes a lightweight policy-memory model and train/val JSONL splits. This is the first "brain" training artifact: it stores the best high-level action candidates and failure feedback by task/affordance while keeping SONIC frozen. A neural VLA/IL/RL trainer can later consume the same split format.

```bash
python3 scripts/tools/rollout_dataset_builder.py \
  --rollouts reports/rollouts \
  --policy-outcomes reports/policy_outcomes/sonic_policy_outcomes.jsonl \
  --suite configs/world_model/task_suites/sonic_general_v0.yaml
python3 scripts/tools/task_policy_train.py \
  --input reports/policy_outcomes/sonic_policy_outcomes.jsonl \
  --name sonic_general_v0_task_policy_memory
python3 scripts/tools/task_policy_export.py \
  --model reports/policy_models/sonic_general_v0_task_policy_memory.json \
  --output reports/policy_data/sonic_general_v0_task_policy_memory_actions.jsonl
```

`rollout_dataset_builder.py` writes one compact episode row per rollout: task spec, stage timeline, summary outcome, policy observation/action, correction labels, sensor contract, and artifact references. `task_policy_export.py` converts the trained policy-memory model back into policy-action JSONL. The ball demo can read either the exported JSONL or the policy-memory JSON directly through `--policy-action-json`, so the next A/B cycle can use the trained task-level policy without changing SONIC:

```bash
python3 scripts/tools/task_suite_rollout.py \
  --suite configs/world_model/task_suites/sonic_general_v0.yaml \
  --task ball_left_to_tray \
  --runs-per-task 10 \
  --headless \
  --reset-each-rollout \
  --policy-action-json reports/policy_data/sonic_general_v0_task_policy_memory_actions.jsonl \
  --policy-action-apply safe
```

Run an A/B controller rollout when you want to compare the old demo behavior against the feedback-adjusted high-level policy. The adjusted variant passes `--policy-action-json` into the ball demo, which maps policy feedback into bounded primitive parameters; the baseline leaves the demo unchanged. Start with `--dry-run`, then remove it for a real reused-stack rollout:

```bash
python3 scripts/tools/policy_ab_runner.py ball --runs 1 --dry-run --adjusted-apply safe
python3 scripts/tools/policy_ab_runner.py ball --runs 6 --headless --no-reset-each-rollout --adjusted-apply safe
python3 scripts/tools/policy_ab_runner.py ball --runs 6 --headless --adjusted-apply full
```

`/sonic_world/dispatch_plan` is the boundary between high-level reasoning and executors. It maps each skill to a handler such as `ros2_goal_pose`, `demo_locomotion_phase_runtime`, `contact_grasp_primitive`, or `lift_stability_primitive`, with structured commands, phase names, monitor events, and recovery events.

Each dispatch step also carries a capability contract:

```json
{
  "readiness": "ready",
  "contract": {
    "capability": "contact_grasp",
    "handler": "contact_grasp_primitive",
    "ready": true,
    "failed_errors": [],
    "recovery_suggestions": [],
    "checks": [
      {"name": "target.exists", "passed": true, "severity": "error"},
      {"name": "grasp.target_pose_base", "passed": true, "severity": "error"}
    ]
  }
}
```

This is the explicit safety boundary for VLM/VLA outputs. For example, `configs/world_model/scenarios/generic_contract_missing_base.json` is expected to plan a graspable object but mark manipulation steps as `needs_attention` because the object has no base-frame pose. The dispatch metadata then suggests recovery actions such as `reobserve_from_current_view`, `publish_object_anchor_with_pose_base`, and `micro_adjust_base_for_observation`. The planner can therefore say “this task is conceptually valid, but not executable yet; here is what information/action is missing” instead of silently handing a bad command to SONIC.

`/sonic_world/recovery_plan` turns those suggestions into routable actions:

```json
{
  "status": "needs_recovery",
  "actions": [
    {
      "suggestion": "publish_object_anchor_with_pose_base",
      "handler": "object_anchor_update",
      "target_id": "far_fruit",
      "command": {
        "type": "publish_object_anchor_with_pose_base",
        "required_fields": ["pose_base"],
        "target_topic": "/sonic_world/object_anchor"
      }
    }
  ]
}
```

This gives a future VLA loop a concrete next step: observe again, request a richer anchor, micro-adjust base pose, repair affordances, replan runtime, or send the task to manual review.

`/sonic_world/decision_plan` is the single routing output for the next layer. It chooses recovery before execution when contracts are not ready, and otherwise exposes the first ready dispatch action:

```json
{
  "status": "needs_recovery",
  "next_action": {
    "kind": "recovery",
    "handler": "object_anchor_update",
    "target_id": "far_fruit",
    "reason": "publish_object_anchor_with_pose_base"
  }
}
```

`world_model_executor.py` owns the ordered dispatch lifecycle. It accepts one plan at a time, validates action readiness, publishes only the current skill, waits for a matching terminal status, verifies declared effects, and then advances. Duplicate plans/statuses are idempotent; timeout, cancellation, skipped actuation, or failed effect evidence stops the plan and emits `/sonic_world/recovery_request`. `queued` is not treated as success.

Run the dispatch-driven manipulation path without the legacy demo main loop:

```bash
python3 scripts/tools/world_model_primitive_runner.py \
  --backend zmq_phase \
  --scene gear_sonic/data/robot_model/model_data/g1/scene_sonic_task_ball_left_to_tray.xml \
  --effect-observer mujoco_qpos
python3 scripts/tools/world_model_executor.py --require-effect-evidence
python3 scripts/tools/world_model_autonomous_task.py \
  --task move --object-id sg_ball --target-id ball_left_to_tray_target
```

The primitive runner executes one skill family at a time (`navigation`, `workspace`, `single_hand_grasp`, `bimanual_grasp`, `lift`, `transport`, `place`, or `release`). Pinch, side, and top grasp are profiles of the same single-hand primitive. The MuJoCo effect observer checks the live qpos scene identity, target/base geometry, contact count, lift delta, target distance, and base stability before reporting structured effect evidence. `status_only` validates a command but now reports `skipped`; `contract_test` is reserved for deterministic ROS integration tests.

`world_model_recovery_coordinator.py` subscribes to `/sonic_world/recovery_request`, routes each request to perception, navigation, runtime, or manual/VLA review topics, and publishes `/sonic_world/recovery_status`. The coordinator remains transport-only; executable behavior lives in the recovery backend nodes.

`world_model_recovery_backends.py` provides the executable backends: perception re-observation triggers the RGB-D/VLM pipeline, base micro-adjust publishes a bounded `/cmd_vel_nav` command and stop, anchor/affordance/support repair republishes normalized state, and runtime replan makes `world_model_node.py` rebuild the remaining plan from the latest world revision.

For real perception, run the request-driven Qwen-VL detector and RGB-D fusion backend. The local model is free to use once downloaded; it needs a CUDA GPU and does not call a paid API:

```bash
# One-time local setup. With the local SOCKS proxy used in this workspace:
uv venv --python /usr/bin/python3 .venv_qwen_vl
uv pip install --python .venv_qwen_vl/bin/python -r requirements-qwen-vl.txt torchvision
export all_proxy=socks5://127.0.0.1:7890 HF_HUB_DISABLE_XET=1
.venv_qwen_vl/bin/hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct

# Start the local OpenAI-compatible endpoint after the download completes.
bash scripts/tools/start_local_qwen_vl.sh

# Separate terminals, shadow mode only.
export QWEN_VL_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export QWEN_VL_MODEL=models/Qwen2.5-VL-3B-Instruct
/usr/bin/python3 scripts/tools/world_model_qwen_vl_detector.py
/usr/bin/python3 scripts/tools/world_model_rgbd_anchor_backend.py \
  --anchor-topic /sonic_world/qwen_rgbd_anchor
```

Qwen-VL produces semantic 2D detections. The RGB-D backend uses a robust depth patch, camera intrinsics, and TF to add camera/base/map poses, tracking ids, support, confidence, and uncertainty before publishing the generic `objects[]` anchor contract. Introduce Qwen-VL first in shadow mode: record its anchors next to privileged simulation anchors, then run the anchor gate before enabling it for recovery or planning:

```bash
python3 scripts/tools/world_model_rgbd_anchor_backend.py \
  --anchor-topic /sonic_world/qwen_rgbd_anchor
python3 scripts/tools/world_model_shadow_anchor_recorder.py \
  --reference-topic /sonic_demo/ball_anchor \
  --prediction-topic /sonic_world/qwen_rgbd_anchor \
  --max-pairs 200
python3 scripts/tools/world_model_vlm_anchor_eval.py \
  --reference reports/perception/privileged_anchors.jsonl \
  --prediction reports/perception/qwen_rgbd_anchors.jsonl \
  --strict
```

The default gate requires 0.90 precision/recall, median base-pose error at most 8 cm, 0.90 support and tracking accuracy, and 0.90 target-region recall. A failed gate keeps Qwen-VL out of execution and limits it to recorded diagnostics; a passing gate can enable request-driven perception recovery before considering continuous visual planning.

`rollout_batch.py` packages that shadow path for a controller batch. It waits for the local server's health endpoint, starts Qwen detection at a conservative cadence, fuses RGB-D anchors, and writes per-batch paired JSONL under `reports/perception/`. It deliberately never publishes the Qwen output to `/sonic_world/object_anchor`:

```bash
/usr/bin/python3 scripts/tools/rollout_batch.py ball --runs 3 --headless \
  --qwen-vl-shadow --qwen-vl-period 5 --qwen-vl-shadow-max-pairs 200
python3 scripts/tools/world_model_vlm_anchor_eval.py \
  --reference reports/perception/ball_test_privileged_anchors.jsonl \
  --prediction reports/perception/ball_test_qwen_rgbd_anchors.jsonl --strict
```

The batch gives Qwen the active task object/category and target ID by default. Override that task-conditioned query with `--qwen-vl-instruction` only when evaluating a different language grounding prompt.

Each Qwen shadow batch also writes `<prefix>_rgbd_calibration.jsonl`. This reconstructs a privileged anchor's known image pixel through the same depth and TF path, separating RGB-D calibration error from Qwen 2D grounding error before changing any gate threshold.

Before recording Qwen predictions, the batch applies a bounded median tracker (`--qwen-vl-temporal-window`, default 3) and waits for the same number of observations. This is still shadow-only; it prevents a one-frame visual outlier from appearing as a candidate execution anchor.

For deterministic simulation diagnostics, use `--perception-shadow-backend hsv`. It replaces only the 2D detector with HSV regions for the benchmark's colored object/target materials; RGB-D fusion, calibration, temporal tracking, JSONL pairing, and the gate stay exactly the same. It is a detector baseline, not a substitute for the Qwen real-perception path.

`--perception-shadow-backend grounding_dino` starts a local Grounding DINO Tiny service and uses task-conditioned labels for 2D boxes before the same RGB-D/gate pipeline. This is the recommended next backend for real visual grounding: Qwen supplies task semantics while Grounding DINO supplies spatial boxes.

Grounding DINO is also local and has no per-request API cost after its one-time download. It remains shadow-only until the same strict anchor gate passes:

```bash
export all_proxy=socks5://127.0.0.1:7890 HF_HUB_DISABLE_XET=1
.venv_qwen_vl/bin/hf download IDEA-Research/grounding-dino-tiny \
  --local-dir models/grounding-dino-tiny

# Runs the local CUDA service on 127.0.0.1:8001.
bash scripts/tools/start_local_grounding_dino.sh

# Starts the service, task-conditioned detector, RGB-D fusion, calibration,
# temporal filter, and paired shadow recorder as one benchmark batch.
/usr/bin/python3 scripts/tools/rollout_batch.py ball --runs 3 --headless \
  --qwen-vl-shadow --perception-shadow-backend grounding_dino
```

All visual detectors now attach the original RGB sensor timestamp to their 2D result. The RGB-D backend selects the matching cached depth frame and queries TF at that timestamp, preventing delayed model inference from using a newer robot pose. A visual anchor with no sufficiently close depth frame is rejected with an explicit backend status rather than being projected with stale geometry.

To exercise the visual recovery loop during a shadow rollout, add `--visual-auto-reobserve`. The RGB-D backend checks the required task object (and ball-task target), emits a bounded `perception_reobserve` request on absence, and the recovery coordinator invokes the detector again through `/sonic_world/perception_reobserve_cmd`. Retries are opt-in, rate-limited, and capped, so a persistent miss becomes an explicit blocked recovery state instead of an unbounded loop:

```bash
/usr/bin/python3 scripts/tools/rollout_batch.py ball --runs 1 --headless \
  --qwen-vl-shadow --perception-shadow-backend grounding_dino \
  --visual-auto-reobserve --visual-reobserve-max-attempts 2
```

For a guarded active-recovery experiment, add `--visual-recovery-escalate-navigation`. Only after the re-observation budget is exhausted does it issue one 8 cm bounded base micro-adjust; the recovery backend stops the command and requests a runtime replan. Keep this flag off for normal shadow evaluation.

Qwen-VL can take tens of seconds when its weights are CPU-offloaded. Use `--qwen-vl-device auto --qwen-vl-gpu-memory-gib 6` on a contended GPU; the corresponding RGB-D path retains 480 depth frames by default so the detection is still fused with its original capture frame. Adjust `--qwen-vl-depth-cache-size` only when the camera rate or measured model latency changes.

Promotion requires a multi-task report, not one rollout report. Aggregate only the selected task prefixes and retain the default gate floor of 20 tasks and 100 paired frames (lower values are useful only for local smoke tests):

```bash
/usr/bin/python3 scripts/tools/world_model_vlm_benchmark.py \
  --perception-dir reports/perception \
  --prefix perception_v1_ball_left_to_tray \
  --prefix perception_v1_fruit_right_to_plate \
  --output reports/perception/grounding_dino_multitask_eval.json --strict
```

After a report passes, use it as the explicit promotion token. The gate relay validates every generic anchor again and then forwards it to `/sonic_world/object_anchor`; an absent, malformed, or failed report prevents that relay from starting:

```bash
/usr/bin/python3 scripts/tools/rollout_batch.py ball --runs 3 --headless \
  --qwen-vl-shadow --qwen-vl-gate-report reports/perception/vlm_anchor_eval.json
```

Train and load the learned task-space policy:

```bash
python3 scripts/tools/task_policy_linear_train.py \
  --held-out-category bottle
python3 scripts/tools/world_model_node.py \
  --policy-backend learned \
  --policy-model reports/policy_models/sonic_linear_task_policy_v0.json
```

The linear checkpoint predicts task-space base, hand, wrist, close-ratio, and lift/place targets online. Its features do not include task id; train/validation splits are grouped by task id and may hold out complete object categories. The model manifest records source/checkpoint hashes and split membership.

Passing `world_model_executor.py --dispatch-topic /sonic_world/dispatch_plan` also enables the old all-step dispatch observation path.

The intended control split is:

```text
Object anchors / map / robot state
        ↓
World model: objects, relations, affordances
        ↓
Task planner: task template + skill graph
        ↓
Dispatch plan: handler + command contract
        ↓
Decision plan: execute or recover
        ↓
Skill primitives: navigation, workspace alignment, grasp, lift, place
        ↓
SONIC locomotion + upper-body IK/WBC + contact servo
```

This keeps VLM/VLA integration clean: a visual model can choose the task verb, object, and goal region, while the template planner turns that into typed skills and the primitive layer handles reachability, collision/contact checks, retries, and stable execution.

## Installation

```bash
# 1. System dependencies
sudo apt install ros-humble-desktop ros-humble-navigation2 ros-humble-slam-toolbox

# 2. Python packages
/usr/bin/python3 -m pip install mujoco numpy msgpack torch

# 3. MuJoCo simulator
cd GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
python download_from_hf.py

# 4. Build C++ deploy
cd gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build

# 5. Build ROS2 package
cd ../g1_ros2_nav
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws/src && ln -sf $(pwd) ~/ros2_ws/src/g1_ros2_nav
cd ~/ros2_ws && colcon build --symlink-install

# 6. MPPI (optional)
git clone git@github.com:MarineRock10/CARMA-MPPI.git ~/CARMA-MPPI-main
```

## Architecture

```
MuJoCo Sim ──DDS──► C++ Deploy ◄──ROS2── Goal Follower ◄── RViz
    │                    │
    └── qpos.npy ──► Sensor Bridge ──► /odom /tf
                     Mid360 Pub ────► /mid360_points
                     Camera Pub ────► /camera/*
                     Box Anchor ────► /sonic_demo/box_anchor
                     Ball Anchor ───► /sonic_demo/ball_anchor
                     World Model ───► /sonic_world/model /sonic_world/skill_graph /sonic_world/runtime_plan /sonic_world/dispatch_plan /sonic_world/recovery_plan /sonic_world/decision_plan /sonic_world/execution_state
                     Executor ──────► /sonic_world/executor_event /sonic_world/recovery_request
                     Recovery Coord ─► /sonic_world/recovery_status /sonic_world/*_recovery_request
```

Navigation parameters live in `configs/nav/*.yaml`. The scripts load these YAML defaults and still allow selected `SONIC_*` environment overrides for quick experiments.

## ROS2 Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/odom` | `nav_msgs/Odometry` | `perception/sensor_pub.py` |
| `/tf` | `tf2_msgs/TFMessage` | `perception/sensor_pub.py` |
| `/mid360_points` | `sensor_msgs/PointCloud2` | `perception/mid360_pub.py` |
| `/camera/color/image_raw` | `sensor_msgs/Image` | `perception/camera_pub.py` |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | `perception/camera_pub.py` |
| `/sonic_demo/box_anchor` | `std_msgs/String` | `perception/box_anchor_pub.py` |
| `/sonic_demo/ball_anchor` | `std_msgs/String` | `perception/ball_anchor_pub.py` |
| `/sonic_demo/phase` | `std_msgs/String` | `manipulation/box_grasp_demo.py` / `manipulation/ball_pick_place_demo.py` |
| `/sonic_world/model` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/skill_graph` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/runtime_plan` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/dispatch_plan` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/recovery_plan` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/decision_plan` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/execution_state` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/executor_event` | `std_msgs/String` | `tools/world_model_executor.py` |
| `/sonic_world/recovery_request` | `std_msgs/String` | `tools/world_model_executor.py` |
| `/sonic_world/recovery_status` | `std_msgs/String` | `tools/world_model_recovery_coordinator.py` |
| `/sonic_world/perception_recovery_request` | `std_msgs/String` | `tools/world_model_recovery_coordinator.py` |
| `/sonic_world/navigation_recovery_request` | `std_msgs/String` | `tools/world_model_recovery_coordinator.py` |
| `/sonic_world/runtime_recovery_request` | `std_msgs/String` | `tools/world_model_recovery_coordinator.py` |
| `/sonic_world/manual_recovery_request` | `std_msgs/String` | `tools/world_model_recovery_coordinator.py` |
| `/sonic_world/active_task` | `std_msgs/String` | `tools/world_model_node.py` |
| `/sonic_world/object_anchor` | `std_msgs/String` | VLM/perception generic object anchor input |
| `/sonic_world/task_request` | `std_msgs/String` | VLM/VLA or scripted task requester |
| `/sonic_demo/skill_graph` | `std_msgs/String` | `manipulation/box_grasp_demo.py` / `manipulation/ball_pick_place_demo.py` |
| `/sonic_demo/runtime_plan` | `std_msgs/String` | `manipulation/box_grasp_demo.py` / `manipulation/ball_pick_place_demo.py` |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz |
| `ControlPolicy/upper_body_pose` | `std_msgs/ByteMultiArray` | `navigation/goal_follower.py` |

## Environment Variables

| Variable | Value |
|----------|-------|
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `1` |
| `ROS_DOMAIN_ID` | `42` |

## Scripts

```
scripts/
├── start.py              # One-click: sim + deploy + sensors + nav
├── start_mppi.py         # One-click: MPPI navigation variant
├── start_dwa.py          # One-click: DWA navigation variant
├── start_box_demo.py     # One-click: table-top box grasp demo
├── start_ball_demo.py    # One-click: single-hand ball pick-place demo
├── start_molmospaces_demo.py # One-click: MolmoSpaces episode adapter
├── start_robocasa_task.py # One-click: generated/static RoboCasa task launcher
├── switch_scene.sh       # Scene selection helper
├── navigation/           # planners, costmaps, metrics, go-to-point control
├── perception/           # odom/tf, lidar, camera, and object-anchor publishers
├── manipulation/         # box/ball demos and WAM-style contact primitives
├── sonic_world/          # world objects, affordances, task planner, skill graph
├── tools/                # preview/self-test, scene tools, keyboard/teleop utilities
└── rviz.sh               # RViz with correct environment
```

---

## Credits

Built on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

MPPI sampler from [CARMA-MPPI](https://github.com/MarineRock10/CARMA-MPPI).
