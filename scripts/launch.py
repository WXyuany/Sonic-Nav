#!/usr/bin/env python3
import os, sys, time, signal, subprocess

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
os.chdir(REPO)

ENV = os.environ.copy()
ENV.update({
    "DISPLAY": os.environ.get("DISPLAY", ":1"),
    "ROS_DOMAIN_ID": "42",
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "PYTHONPATH": f"{REPO}:{REPO}/g1_ros2_nav:{os.environ.get('PYTHONPATH', '')}",
    "TensorRT_ROOT": os.path.expanduser("~/TensorRT"),
    "XAUTHORITY": os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority")),
})

processes = []
DEPLOY_LOG = "/tmp/sonic_deploy.log"


def log(tag, msg):
    print(f"[{tag:>6}] {msg}", flush=True)


def start_sim():
    log("SIM", "Starting MuJoCo...")
    cmd = f"source {REPO}/.venv_sim/bin/activate && exec python {REPO}/gear_sonic/scripts/run_sim_loop.py"
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("sim", proc))
    time.sleep(3)
    if proc.poll() is not None:
        log("SIM", f"CRASHED (exit={proc.returncode})")
        sys.exit(1)
    log("SIM", "Running")


def start_deploy():
    log("DEPLOY", "Starting (TRT loading)...")
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
    return proc


def wait_for(pattern, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(DEPLOY_LOG):
            with open(DEPLOY_LOG) as f:
                content = f.read()
            if pattern in content:
                return True
        time.sleep(0.5)
    return False


def start_keepalive():
    log("CTRL", "Starting keepalive (sends start + holds connection)...")
    script = '''
import os, rclpy, msgpack, time
os.environ.update({"RMW_IMPLEMENTATION":"rmw_fastrtps_cpp","ROS_LOCALHOST_ONLY":"1","ROS_DOMAIN_ID":"42"})
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
rclpy.init()
n = Node("keepalive")
p = n.create_publisher(ByteMultiArray, "ControlPolicy/upper_body_pose", 10)
time.sleep(2)
pl = {"navigate_cmd":[0,0,0],"locomotion_mode":0,"base_height_command":0.78,"toggle_policy_action":True}
m = ByteMultiArray()
m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
p.publish(m)
time.sleep(1)
pl["toggle_policy_action"] = False
print("KEEPALIVE_OK")
while True:
    m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
    p.publish(m)
    time.sleep(0.08)
'''
    proc = subprocess.Popen(
        ["bash", "-c", f"source /opt/ros/humble/setup.bash && /usr/bin/python3 -c '{script}'"],
        env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    processes.append(("keepalive", proc))
    t0 = time.time()
    while time.time() - t0 < 20:
        line = proc.stdout.readline()
        if "KEEPALIVE_OK" in line:
            log("CTRL", "Keepalive active")
            return
        if not line and proc.poll() is not None:
            break
    log("CTRL", "FAILED")


def start_bridge():
    log("BRIDGE", "Starting cmd_vel bridge...")
    cmd = f"source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && exec ros2 run g1_ros2_nav cmd_vel_bridge"
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("bridge", proc))
    log("BRIDGE", "Running")


def cleanup():
    log("STOP", "Shutting down...")
    for name, proc in reversed(processes):
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log("STOP", "All stopped")


def tail_log(n=8):
    if not os.path.exists(DEPLOY_LOG):
        return
    with open(DEPLOY_LOG) as f:
        lines = f.readlines()
    for line in lines[-n:]:
        line = line.strip()
        if line and "Loop timing" not in line and "ROS2 DEBUG" not in line:
            print(f"  {line}")


def main():
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print("=" * 45)
    print("  Sonic-Nav  |  DOMAIN=42  RMW=fastrtps")
    print("=" * 45)

    start_sim()
    start_deploy()

    log("WAIT", "Waiting for deploy Init Done (~30s for TRT load)...")
    if not wait_for("Init Done", 90):
        log("ERROR", "Deploy failed to init")
        tail_log(20)
        cleanup()
        sys.exit(1)
    log("DEPLOY", "Init Done!")

    start_bridge()
    time.sleep(2)
    start_keepalive()

    print()
    print("=" * 45)
    print("  Sonic-Nav Ready!  Robot standing.")
    print("  Keyboard:  /usr/bin/python3 scripts/keyboard_control.py")
    print("  Ctrl+C to stop")
    print("=" * 45)

    try:
        signal.pause()
    except AttributeError:
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
