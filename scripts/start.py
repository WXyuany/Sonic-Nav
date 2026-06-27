#!/usr/bin/env python3
import os, sys, time, signal, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO)
ENV = os.environ.copy()
ENV.update({"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "ROS_LOCALHOST_ONLY": "1",
            "ROS_DOMAIN_ID": "42", "DISPLAY": ":1"})
procs = []

def run_script(script, name):
    print(f"[{name}] Starting...")
    cmd = f"source /opt/ros/humble/setup.bash && exec /usr/bin/python3 {REPO}/scripts/{script}"
    p = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(3)
    print(f"[{name}] Running")

def cleanup(*_):
    print("\n[STOP] Shutting down...")
    for p in reversed(procs):
        try: p.terminate(); p.wait(timeout=5)
        except: p.kill()
    print("[STOP] Done"); sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("=" * 45)
print("  Sonic-Nav  |  DOMAIN=42")
print("=" * 45)

# 1. Sim
sim = subprocess.Popen(["bash", "-c",
    f"source {REPO}/.venv_sim/bin/activate && export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && exec python {REPO}/gear_sonic/scripts/run_sim_loop.py"],
    env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
procs.append(sim)
print("[SIM] Starting...")
time.sleep(6)
print("[SIM] Running")

# 2. Deploy (ROS2 mode)
open("/tmp/sonic_deploy.log", "w").close()
deploy = subprocess.Popen(["bash", "-c",
    f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh >/dev/null 2>&1 && cd {REPO}/gear_sonic_deploy && "
    f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
    f"--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx "
    f"--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type ros2 --output-type all "
    f"--zmq-host localhost --disable-crc-check"],
    env=ENV, stdout=open("/tmp/sonic_deploy.log", "w"), stderr=subprocess.STDOUT)
procs.append(deploy)
print("[DEPLOY] Starting...")
t0 = time.time()
while time.time() - t0 < 120:
    if os.path.exists("/tmp/sonic_deploy.log"):
        with open("/tmp/sonic_deploy.log") as f:
            if "Init Done" in f.read(): break
    time.sleep(1)
print("[DEPLOY] Init Done!")

# Auto-start control
print("[CTRL] Sending start command...")
subprocess.run(["bash", "-c",
    "source /opt/ros/humble/setup.bash && /usr/bin/python3 -c '"
    "import os,rclpy,msgpack,time;os.environ.update({\"RMW_IMPLEMENTATION\":\"rmw_fastrtps_cpp\",\"ROS_LOCALHOST_ONLY\":\"1\",\"ROS_DOMAIN_ID\":\"42\"});"
    "from rclpy.node import Node;from std_msgs.msg import ByteMultiArray;rclpy.init();n=Node(\"s\");"
    "p=n.create_publisher(ByteMultiArray,\"ControlPolicy/upper_body_pose\",10);time.sleep(3);"
    "pl={\"navigate_cmd\":[0,0,0],\"locomotion_mode\":0,\"base_height_command\":0.78,\"toggle_policy_action\":True};"
    "m=ByteMultiArray();m.data=[bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)];p.publish(m);time.sleep(2);"
    "n.destroy_node();rclpy.shutdown();print(\"OK\")'"],
    env=ENV)
print("[CTRL] Robot should be standing")

# 3. Sensors
run_script("sensor_pub.py", "SENSOR")
time.sleep(2)
run_script("mid360_pub.py", "MID360")
time.sleep(2)
run_script("camera_pub.py", "CAM")

# 4. Navigation
run_script("goal_follower.py", "NAV")

print()
print("=" * 45)
print("  Ready! Run RViz:")
print("    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42")
print("    source /opt/ros/humble/setup.bash && rviz2")
print("  Click 2D Goal Pose to navigate.")
print("  Ctrl+C to stop all.")
print("=" * 45)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    cleanup()
