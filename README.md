# Sonic-Nav

ROS2 Navigation Stack for GR00T Whole-Body Control (Unitree G1 humanoid robot).

Based on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

---

## Quick Start

```bash
# Terminal 1 — One-click launch
python scripts/start.py          # basic go-to-point
python scripts/start_mppi.py     # MPPI collision avoidance

# Terminal 2 — RViz
bash scripts/rviz.sh
```

Click **2D Goal Pose** in RViz to navigate. Ctrl+C to stop.

## Features

| Mode | Script | Description |
|------|--------|-------------|
| Go-to-Point | `start.py` | Smooth turning, proportional control |
| MPPI Nav | `start_mppi.py` | GPU trajectory sampling + collision avoidance |
| Keyboard | `keyboard_control.py` | WASD manual control |

## Scenes

```bash
bash scripts/switch_scene.sh <name>
```

| Scene | Description |
|-------|-------------|
| `default` | 8m×8m room, cylinder obstacles |
| `dynamic` | Moving obstacles (sliding + rotating) |
| `stairs` | 10-step staircase + ramp |
| `uneven` | Bumpy terrain + rocks |

Restart sim after switching.

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
```

## ROS2 Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/odom` | `nav_msgs/Odometry` | sensor_pub.py |
| `/tf` | `tf2_msgs/TFMessage` | sensor_pub.py |
| `/mid360_points` | `sensor_msgs/PointCloud2` | mid360_pub.py |
| `/camera/color/image_raw` | `sensor_msgs/Image` | camera_pub.py |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | camera_pub.py |
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
├── goal_follower.py      # Go-to-point navigation (odom feedback)
├── mppi_nav.py           # MPPI navigation (GPU trajectory sampling)
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
