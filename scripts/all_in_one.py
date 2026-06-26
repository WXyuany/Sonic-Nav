#!/usr/bin/env python3
"""Simple navigation: click 2D goal in RViz → robot walks there."""
import os, sys, math, time, subprocess, select, termios, tty, threading, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import ByteMultiArray
import msgpack

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
os.environ.setdefault("RMW_IMPLEMENTATION","rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY","1")  
os.environ.setdefault("ROS_DOMAIN_ID","42")


class GoalFollower(Node):
    def __init__(self, key_writer):
        super().__init__("goal_follower")
        self._kw = key_writer
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self._timer = self.create_timer(0.1, self._tick)
        self._goal = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self.get_logger().info("Goal follower ready. Click 2D Goal Pose in RViz.")

    def update_pose(self, x, y, yaw):
        self._robot_x = x
        self._robot_y = y
        self._robot_yaw = yaw

    def _on_goal(self, msg):
        self._goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"New goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})")

    def _tick(self):
        if self._goal is None:
            return
        gx, gy = self._goal
        dx = gx - self._robot_x
        dy = gx - self._robot_y
        ds = 0
        dist = math.hypot(dx, dy)
        if dist < 0.3:
            self._kw(" ")
            self._goal = None
            self.get_logger().info("Goal reached!")
            return
        target_yaw = math.atan2(dy, dx)
        yaw_diff = target_yaw - self._robot_yaw
        yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
        if abs(yaw_diff) > 0.2:
            self._kw("q" if yaw_diff > 0 else "e")
        elif abs(yaw_diff) > 0.1:
            self._kw("w")
            self._kw("q" if yaw_diff > 0 else "e")
        else:
            self._kw("w")


def main():
    # Use existing tmux session if available
    result = subprocess.run(["tmux","has-session","-t","sonic-nav"],capture_output=True)
    if result.returncode != 0:
        # Start fresh
        subprocess.run(["tmux","new-session","-d","-s","sonic-nav",
            f"export DISPLAY=:1 PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' && source {REPO}/.venv_sim/bin/activate && python {REPO}/gear_sonic/scripts/run_sim_loop.py"],check=True)
        print("[SIM] Started")
        time.sleep(8)
        subprocess.run(["tmux","split-window","-h","-t","sonic-nav"])
        subprocess.run(["tmux","send-keys","-t","sonic-nav:0.1",
            f"cd {REPO}/gear_sonic_deploy && source scripts/setup_env.sh >/dev/null 2>&1 && ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type keyboard --output-type all --zmq-host localhost --disable-crc-check","Enter"],check=True)
        print("[DEPLOY] Starting...")
        time.sleep(25)
        for i in range(60):
            out = subprocess.run(["tmux","capture-pane","-t","sonic-nav:0.1","-p"],capture_output=True,text=True).stdout
            if "Init Done" in out:
                print("[DEPLOY] Init Done!")
                break
            time.sleep(2)
        subprocess.run(["tmux","send-keys","-t","sonic-nav:0.1","]"],timeout=5)
        time.sleep(2)
        subprocess.run(["tmux","send-keys","-t","sonic-nav:0.1","Enter"],timeout=5)
        time.sleep(2)
        print("[DEPLOY] Control started, planner enabled")
    else:
        print("[DEPLOY] Using existing tmux session")

    # Start ROS2 goal follower
    rclpy.init()
    follower = GoalFollower(lambda k: subprocess.run(
        ["tmux","send-keys","-t","sonic-nav:0.1",k],capture_output=True,timeout=1))
    
    print("\n========================================")
    print("  Ready! Click '2D Goal Pose' in RViz.")
    print("  Ctrl+C to stop")
    print("========================================\n")

    while True:
        try:
            rclpy.spin_once(follower, timeout_sec=0.1)
        except KeyboardInterrupt:
            break
        except rclpy.executors.ExternalShutdownException:
            follower.destroy_node()
            try: rclpy.shutdown()
            except: pass
            rclpy.init()
            follower = GoalFollower(lambda k: subprocess.run(
                ["tmux","send-keys","-t","sonic-nav:0.1",k],capture_output=True,timeout=1))
            time.sleep(1)
        except Exception:
            time.sleep(0.5)
    follower.destroy_node()
    rclpy.shutdown()
    subprocess.run(["tmux","kill-session","-t","sonic-nav"])


if __name__ == "__main__":
    main()
