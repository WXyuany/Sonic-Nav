# Sonic-Nav

ROS2 Navigation Stack for GR00T Whole-Body Control (Unitree G1 humanoid robot).

Based on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

---

## Quick Start

```bash
# Terminal 1 — One-click launch
python scripts/start.py          # basic go-to-point
python scripts/start_mppi.py     # MPPI collision avoidance
python scripts/start_dwa.py      # DWA local planner
python scripts/start_box_demo.py box_demo  # table-top box grasp demo
python scripts/start_ball_demo.py ball_demo # single-hand ball pick-place demo

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
| Keyboard | `keyboard_control.py` | WASD manual control |

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

The manipulation logic is factored through `scripts/wam_primitives.py`: object anchors are treated as a lightweight world model, `WorkspaceAligner` keeps the object inside the reachable hand workspace with base micro-adjustments, and `ContactServoPolicy` uses fingertip contact error to adjust hand targets, close ratio, and lift lead online. This is the intended bridge toward VLA/VLM policies: the model can provide object/goal anchors while the classical primitive layer handles stability, contact, and retry logic.

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
```

Navigation parameters live in `configs/nav/*.yaml`. The scripts load these YAML defaults and still allow selected `SONIC_*` environment overrides for quick experiments.

## ROS2 Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/odom` | `nav_msgs/Odometry` | sensor_pub.py |
| `/tf` | `tf2_msgs/TFMessage` | sensor_pub.py |
| `/mid360_points` | `sensor_msgs/PointCloud2` | mid360_pub.py |
| `/camera/color/image_raw` | `sensor_msgs/Image` | camera_pub.py |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | camera_pub.py |
| `/sonic_demo/box_anchor` | `std_msgs/String` | box_anchor_pub.py |
| `/sonic_demo/ball_anchor` | `std_msgs/String` | ball_anchor_pub.py |
| `/sonic_demo/phase` | `std_msgs/String` | box_grasp_demo.py |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz |
| `ControlPolicy/upper_body_pose` | `std_msgs/ByteMultiArray` | goal_follower.py |

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
├── goal_follower.py      # Go-to-point navigation (odom feedback)
├── mppi_nav.py           # MPPI navigation (GPU trajectory sampling)
├── dwa_nav.py            # DWA navigation (local dynamic window search)
├── box_anchor_pub.py     # Publishes known box pose/grasp anchors
├── box_grasp_demo.py     # ZMQ planner sequence for box approach and grasp
├── ball_anchor_pub.py    # Publishes known ball pose/place anchors
├── ball_pick_place_demo.py # ZMQ planner sequence for right-hand ball pick-place
├── sensor_pub.py         # /odom /tf publisher
├── mid360_pub.py         # Livox Mid-360 point cloud simulator
├── camera_pub.py         # RealSense RGB-D camera simulator
├── keyboard_control.py   # WASD keyboard manual control
└── rviz.sh               # RViz with correct environment
```

---

## Credits

Built on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

MPPI sampler from [CARMA-MPPI](https://github.com/MarineRock10/CARMA-MPPI).
