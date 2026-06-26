#!/usr/bin/env python3
"""Sonic-Nav one-click launch. Starts sim, deploy, bridge, and keyboard control."""
import os, sys, time, signal, subprocess, argparse

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
    log("SIM", "MuJoCo in tmux...")
    subprocess.run(["tmux", "kill-session", "-t", "sonic-sim"], capture_output=True)
    subprocess.run(["tmux", "new-session", "-d", "-s", "sonic-sim",
        f"export DISPLAY=:1 PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' && "
        f"source {REPO}/.venv_sim/bin/activate && "
        f"python {REPO}/gear_sonic/scripts/run_sim_loop.py"], check=True)
    time.sleep(6)
    log("SIM", "Running (tmux a -t sonic-sim)")


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


def start_bridge():
    log("BRIDGE", "cmd_vel bridge...")
    cmd = (f"source /opt/ros/humble/setup.bash && "
           f"source ~/ros2_ws/install/setup.bash && "
           f"exec ros2 run g1_ros2_nav cmd_vel_bridge")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("bridge", proc))
    log("BRIDGE", "Running")


def start_keyboard():
    log("KB", "Starting keyboard control...")
    cmd = (f"source /opt/ros/humble/setup.bash && "
           f"exec /usr/bin/python3 {REPO}/scripts/keyboard_control.py")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV)
    processes.append(("keyboard", proc))
    log("KB", "Use WASD to control, ESC to quit")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-keyboard", action="store_true", help="Don't start keyboard control")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print("=" * 50)
    print("  Sonic-Nav Keyboard  |  DOMAIN=42")
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
    start_bridge()

    if not args.no_keyboard:
        start_keyboard()

    print()
    print("=" * 50)
    print("  Sonic-Nav Ready! Robot standing.")
    print()
    print("  Keyboard: WASD  |  1/2: speed  |  ESC: quit")
    if args.no_keyboard:
        print("  Use: ros2 topic pub /cmd_vel ...")
        print("  Or:  /usr/bin/python3 scripts/keyboard_control.py")
    print("  Ctrl+C to stop all")
    print("=" * 50)

    try:
        signal.pause()
    except AttributeError:
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
