# Sonic-Nav

ROS2 Navigation Stack for GR00T Whole-Body Control (Unitree G1 humanoid robot).

Based on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

---

## Quick Start

### One-Click (2 terminals)

**T1 — Launch all**
```bash
python scripts/start.py
```

**T2 — RViz**
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
rviz2
```
Click **2D Goal Pose** to navigate. Ctrl+C to stop.

### Manual (more control)

**T1 — Sim**
```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate
export PYTHONPATH="$PWD:$PWD/g1_ros2_nav" DISPLAY=:1
python gear_sonic/scripts/run_sim_loop.py
```

**T2 — Deploy** (after MuJoCo window appears)
```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
./target/release/g1_deploy_onnx_ref lo \
    policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type ros2 --output-type all --zmq-host localhost --disable-crc-check
```

**T3 — Sensors** (after Init Done)
```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/sensor_pub.py
```

**T4 — Navigation** (after sensor starts)
```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/goal_follower.py
```

**T5 — RViz**
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
rviz2
```

---

## Keyboard WASD Control

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/keyboard_control.py
```

| Key | Action |
|-----|--------|
| W/S | Forward / Back |
| A/D | Strafe left / right |
| Q/E | Turn left / right |
| 1/2 | Speed down / up |
| Space | Stop |
| Esc | Quit |

---

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU (RTX 5090 tested) + CUDA 13.1
- TensorRT 10.13
- ROS2 Humble
- Python 3.10 (system) + MuJoCo

Install dependencies:
```bash
# ROS2
sudo apt install ros-humble-desktop ros-humble-navigation2 ros-humble-slam-toolbox

# Python packages
/usr/bin/python3 -m pip install mujoco numpy msgpack

# MuJoCo simulator
cd GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
python download_from_hf.py

# Build deploy
cd gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build

# Build ROS2 package
cd ../g1_ros2_nav
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws/src && ln -sf $(pwd) ~/ros2_ws/src/g1_ros2_nav
cd ~/ros2_ws && colcon build --symlink-install
```

---

## Architecture

```
MuJoCo Sim ──DDS──► C++ Deploy ◄──ROS2── Python Goal Follower ◄── RViz
    │                    │                        │
    └── qpos file ──► Sensor Bridge ──► /odom /tf
```

## ROS2 Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/odom` | `nav_msgs/Odometry` | sensor_pub.py |
| `/tf` | `tf2_msgs/TFMessage` | sensor_pub.py |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz |
| `ControlPolicy/upper_body_pose` | `std_msgs/ByteMultiArray` | goal_follower.py |

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | DDS middleware |
| `ROS_LOCALHOST_ONLY` | `1` | Local communication only |
| `ROS_DOMAIN_ID` | `42` | DDS domain isolation |
