#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import threading
import re

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
os.chdir(REPO)

ENV = os.environ.copy()
ENV["DISPLAY"] = ENV.get("DISPLAY", ":1")
ENV["ROS_DOMAIN_ID"] = "42"
ENV["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
ENV["ROS_LOCALHOST_ONLY"] = "1"
ENV["PYTHONPATH"] = f"{REPO}:{REPO}/g1_ros2_nav:{ENV.get('PYTHONPATH', '')}"
ENV["TensorRT_ROOT"] = os.path.expanduser("~/TensorRT")

processes = []


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def start_sim():
    log("SIM", "Starting MuJoCo simulator...")
    venv = f"source {REPO}/.venv_sim/bin/activate"
    cmd = f"{venv} && exec python {REPO}/gear_sonic/scripts/run_sim_loop.py"
    
    env = ENV.copy()
    env["DISPLAY"] = ":1"
    if "XAUTHORITY" not in env:
        env["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")
    
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(("sim", proc))
    time.sleep(8)
    
    if proc.poll() is not None:
        out = proc.stdout.read().decode()[-500:]
        log("SIM", f"CRASHED (exit={proc.returncode}):\n{out}")
        sys.exit(1)
    log("SIM", "Simulator started")


def start_deploy():
    log("DEPLOY", "Starting C++ deployment (ROS2 input)...")
    setup = (
        f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh > /dev/null 2>&1 && "
        f"cd {REPO}/gear_sonic_deploy"
    )
    binary = "./target/release/g1_deploy_onnx_ref"
    cmd = (
        f"{setup} && exec {binary} lo "
        f"policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml "
        f"--encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
        f"--input-type ros2 --output-type all --zmq-host localhost "
        f"--disable-crc-check"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    processes.append(("deploy", proc))

    def drain():
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if "Loop timing" not in line:
                print(f"  [DEPLOY] {line.rstrip()}", flush=True)
    threading.Thread(target=drain, daemon=True).start()
    return proc


def start_keepalive():
    log("CTRL", "Starting keepalive thread (prevents 1s timeout reset)...")
    ros_setup = "source /opt/ros/humble/setup.bash"
    script = f'''
import os, rclpy, msgpack, time
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["ROS_DOMAIN_ID"] = "42"

from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

rclpy.init()
node = Node("keepalive")
pub = node.create_publisher(ByteMultiArray, "ControlPolicy/upper_body_pose", 10)
time.sleep(2)

payload = {{"navigate_cmd": [0,0,0], "locomotion_mode": 0,
           "base_height_command": 0.78, "toggle_policy_action": True}}
msg = ByteMultiArray()
msg.data = [bytes([b]) for b in msgpack.packb(payload, use_bin_type=True)]
pub.publish(msg)
time.sleep(1)

payload["toggle_policy_action"] = False
print("KEEPALIVE_STARTED")
while True:
    msg.data = [bytes([b]) for b in msgpack.packb(payload, use_bin_type=True)]
    pub.publish(msg)
    time.sleep(0.08)
'''
    cmd = f"{ros_setup} && /usr/bin/python3 -c '{script}'"
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    processes.append(("keepalive", proc))
    start = time.time()
    while time.time() - start < 15:
        line = proc.stdout.readline()
        if "KEEPALIVE_STARTED" in line:
            log("CTRL", "Keepalive running, robot will stay standing")
            return
        if not line and proc.poll() is not None:
            break
    log("CTRL", "Keepalive failed to start")


def start_cmd_vel_bridge():
    log("BRIDGE", "Starting cmd_vel bridge...")
    ros_setup = f"source /opt/ros/humble/setup.bash && source {REPO}/../ros2_ws/install/setup.bash"
    cmd = f"{ros_setup} && exec ros2 run g1_ros2_nav cmd_vel_bridge"
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        env=ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    processes.append(("bridge", proc))
    log("BRIDGE", "cmd_vel bridge started")


def cleanup():
    log("STOP", "Shutting down...")
    subprocess.run(["tmux", "kill-session", "-t", "sonic-sim"], capture_output=True)
    for name, proc in reversed(processes):
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log("STOP", "All stopped")


def main():
    signal.signal(signal.SIGINT, lambda *_: cleanup())
    signal.signal(signal.SIGTERM, lambda *_: cleanup())

    print("=" * 50)
    print("  Sonic-Nav One-Click Launch")
    print(f"  ROS_DOMAIN_ID=42  RMW=rmw_fastrtps_cpp")
    print("=" * 50)

    start_sim()
    deploy_proc = start_deploy()

    log("WAIT", "Waiting for deployment initialization (TRT loading ~30s)...")
    init_done = False
    start = time.time()
    while time.time() - start < 120:
        line = deploy_proc.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
        if "Init Done" in line:
            init_done = True
            log("DEPLOY", "Init Done!")
            break
        if "ERROR" in line or "Failed" in line:
            print(f"  {line.strip()}")
        if "LowState is not available" in line:
            log("DEPLOY", "Waiting for sim connection...")

    if not init_done:
        log("ERROR", "Deploy failed to initialize within 120s")
        cleanup()
        sys.exit(1)

    start_cmd_vel_bridge()
    time.sleep(2)
    start_keepalive()
    time.sleep(3)

    print()
    print("=" * 50)
    print("  Sonic-Nav Ready!")
    print("  Robot standing, waiting for /cmd_vel")
    print()
    print("  Keyboard control:")
    print("    /usr/bin/python3 scripts/keyboard_control.py")
    print("  Or cmd_vel:")
    print('    ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}}"')
    print("=" * 50)

    try:
        signal.pause()
    except AttributeError:
        while True:
            time.sleep(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
