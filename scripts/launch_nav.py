#!/usr/bin/env python3
"""Sonic-Nav navigation launch. Starts sim+bridge, deploy, ready for nav2."""
import os, sys, time, signal, subprocess

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
os.chdir(REPO)

ENV = os.environ.copy()
ENV.update({
    "ROS_DOMAIN_ID": "42",
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "TensorRT_ROOT": os.path.expanduser("~/TensorRT"),
})
DEPLOY_LOG = "/tmp/sonic_deploy.log"
processes = []


def log(tag, msg):
    print(f"[{tag:>6}] {msg}", flush=True)


def start_sim():
    log("SIM", "MuJoCo in offscreen mode...")
    cmd = (
        f"source {REPO}/.venv_sim/bin/activate && "
        f"export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && "
        f"exec python {REPO}/gear_sonic/scripts/run_sim_loop.py --no-enable_onscreen"
    )
    proc = subprocess.Popen(["bash", "-c", cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("sim", proc))
    time.sleep(6)
    if proc.poll() is not None:
        log("SIM", "CRASHED")
        sys.exit(1)
    log("SIM", "Running (offscreen)")


def start_deploy():
    log("DEPLOY", "Starting...")
    open(DEPLOY_LOG, "w").close()
    cmd = (
        f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh > /dev/null 2>&1 && "
        f"cd {REPO}/gear_sonic_deploy && "
        f"exec ./target/release/g1_deploy_onnx_ref lo "
        f"policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml "
        f"--encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
        f"--input-type ros2 --output-type all --zmq-host localhost "
        f"--disable-crc-check"
    )
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=open(DEPLOY_LOG, "w"), stderr=subprocess.STDOUT)
    processes.append(("deploy", proc))


def robot_start():
    log("CTRL", "Sending start command...")
    script = '''
import os, rclpy, msgpack, time
os.environ.update({"RMW_IMPLEMENTATION":"rmw_fastrtps_cpp","ROS_LOCALHOST_ONLY":"1","ROS_DOMAIN_ID":"42"})
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
rclpy.init()
n = Node("starter")
p = n.create_publisher(ByteMultiArray, "ControlPolicy/upper_body_pose", 10)
time.sleep(3)
pl = {"navigate_cmd":[0,0,0],"locomotion_mode":0,"base_height_command":0.78,"toggle_policy_action":True}
m = ByteMultiArray()
m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
p.publish(m)
time.sleep(2)
n.destroy_node()
rclpy.shutdown()
print("OK")
'''
    r = subprocess.run(
        ["bash", "-c", f"source /opt/ros/humble/setup.bash && /usr/bin/python3 -c '{script}'"],
        env=ENV, capture_output=True, text=True, timeout=30)
    if "OK" in r.stdout:
        log("CTRL", "Robot standing")


def start_sensor_bridge():
    log("SENSOR", "/odom /scan /tf bridge...")
    cmd = (f"source /opt/ros/humble/setup.bash && "
           f"exec /usr/bin/python3 {REPO}/g1_ros2_nav/g1_ros2_nav/standalone_bridge.py")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("sensor", proc))
    log("SENSOR", "Running")


def start_cmdvel_bridge():
    log("CMDVEL", "cmd_vel → ControlPolicy bridge...")
    cmd = (f"source /opt/ros/humble/setup.bash && "
           f"source ~/ros2_ws/install/setup.bash && "
           f"exec ros2 run g1_ros2_nav cmd_vel_bridge")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("cmdvel", proc))
    log("CMDVEL", "Running")


def wait_for(pattern, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(DEPLOY_LOG):
            with open(DEPLOY_LOG) as f:
                if pattern in f.read():
                    return True
        time.sleep(0.5)
    return False


def cleanup():
    log("STOP", "Shutting down...")
    subprocess.run(["tmux", "kill-session", "-t", "sonic-sim"], capture_output=True)
    for _, proc in reversed(processes):
        if proc is None: continue
        try: proc.terminate(); proc.wait(timeout=5)
        except Exception: proc.kill()
    log("STOP", "Done")


def main():
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print("=" * 50)
    print("  Sonic-Nav Navigation  |  DOMAIN=42")
    print("=" * 50)

    start_sim()
    start_deploy()

    log("WAIT", "Deploy init...")
    if not wait_for("Init Done"):
        log("ERROR", "Deploy init failed")
        cleanup()
        sys.exit(1)
    log("DEPLOY", "Init Done!")

    robot_start()
    start_sensor_bridge()
    start_cmdvel_bridge()

    print()
    print("=" * 50)
    print("  Sonic-Nav Ready! Robot standing.")
    print("  /odom /scan /tf active")
    print()
    print("  Start Navigation + RViz:")
    print("    source /opt/ros/humble/setup.bash")
    print("    source ~/ros2_ws/install/setup.bash")
    print("    ros2 launch g1_ros2_nav bringup.launch.py")
    print()
    print("  Set initial pose:")
    print("    ros2 run nav2_util set_initial_pose -- -x 0 -y 0 -z 0 -yaw 0")
    print()
    print("  Send goal:")
    print('    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \\')
    print('    "{pose: {header: {frame_id: \'map\'}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w: 1.0}}}}"')
    print()
    print("  Ctrl+C to stop all")
    print("=" * 50)

    try:
        signal.pause()
    except AttributeError:
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
