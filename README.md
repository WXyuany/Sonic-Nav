# Sonic-Nav

ROS2 Navigation Stack for GR00T Whole-Body Control (Unitree G1 humanoid robot).

## Quick Start

```bash
# Terminal 1 — One-click launch (sim + deploy + sensors + navigation)
python scripts/start.py

# Terminal 2 — RViz
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
rviz2
```

Click **2D Goal Pose** in RViz to navigate. Ctrl+C to stop.

## Features

### Keyboard WASD Control
```bash
python scripts/keyboard_control.py
```
| Key | Action |
|-----|--------|
| W/S | Forward / Back |
| A/D | Strafe |
| Q/E | Turn |
| 1/2 | Speed -/+ |
| Space | Stop |
| Esc | Quit |

### Navigation (2D Goal Pose)
Launches automatically with `scripts/start.py`. Sends goals from RViz.

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU + CUDA + TensorRT 10.13
- ROS2 Humble
- MuJoCo

## Architecture

```
MuJoCo Sim ──DDS──► C++ Deploy ◄──ROS2── Python Goal Follower ◄── RViz
    │                    │                        │
    └── qpos file ──► Sensor Bridge ──► /odom /tf
```

## Credits

Built on [NVIDIA GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl).
