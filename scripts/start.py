#!/usr/bin/env python3
import os, sys, time, signal, subprocess, shlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO)

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help, set_wbc_scene
from gear_sonic.utils.mujoco_sim.launch_cleanup import cleanup_stale_sonic_processes

SCENE_ARG = sys.argv[1] if len(sys.argv) > 1 else "default"
try:
    SCENE_SELECTION = resolve_scene(SCENE_ARG, repo_root=REPO)
except ValueError as exc:
    raise SystemExit(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc
SCENE = SCENE_SELECTION.name
SCENE_XML = SCENE_SELECTION.xml_file
os.chdir(REPO)
ENV = os.environ.copy()
ENV.update({"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "ROS_LOCALHOST_ONLY": "1",
            "ROS_DOMAIN_ID": "42", "DISPLAY": ":1", "SONIC_SKIP_LFS_PULL": "1"})
procs = []

def _format_args(args=None):
    if args is None:
        return ""
    values = [args] if isinstance(args, (str, os.PathLike)) else list(args)
    return "".join(f" {shlex.quote(str(value))}" for value in values)

def run_script(script, name, args=None):
    print(f"[{name}] Starting...")
    log_path = f"/tmp/sonic_{name.lower()}.log"
    script_path = os.path.join(REPO, "scripts", script)
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"export PYTHONPATH=\"{REPO}:{REPO}/g1_ros2_nav:$PYTHONPATH\" && "
        f"exec /usr/bin/python3 {shlex.quote(script_path)}{_format_args(args)}"
    )
    log = open(log_path, "w")
    p = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=log, stderr=subprocess.STDOUT)
    procs.append(p)
    time.sleep(3)
    if p.poll() is not None:
        print(f"[{name}] FAILED, see {log_path}")
        raise RuntimeError(f"{name} exited with code {p.returncode}")
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

stale_pids = cleanup_stale_sonic_processes()
if stale_pids:
    print(f"[CLEANUP] Stopped stale Sonic processes: {stale_pids}")

# Pre-switch scene so simulator and sensor publishers use the same XML.
set_wbc_scene(SCENE, repo_root=REPO)
print(f"[SCENE] Using {SCENE} ({SCENE_XML})")

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
deploy_ready = False
while time.time() - t0 < 120:
    if os.path.exists("/tmp/sonic_deploy.log"):
        with open("/tmp/sonic_deploy.log") as f:
            if "Init Done" in f.read():
                deploy_ready = True
                break
    time.sleep(1)
if not deploy_ready:
    raise RuntimeError("DEPLOY did not report Init Done within 120s; see /tmp/sonic_deploy.log")
print("[DEPLOY] Init Done!")

# Auto-start control
print("[CTRL] Sending start command...")
subprocess.run(["bash", "-c",
    "source /opt/ros/humble/setup.bash && /usr/bin/python3 -c '"
    "import os,rclpy,msgpack,time;os.environ.update({\"RMW_IMPLEMENTATION\":\"rmw_fastrtps_cpp\",\"ROS_LOCALHOST_ONLY\":\"1\",\"ROS_DOMAIN_ID\":\"42\"});"
    "from rclpy.node import Node;from std_msgs.msg import ByteMultiArray;rclpy.init();n=Node(\"s\");"
    "p=n.create_publisher(ByteMultiArray,\"ControlPolicy/upper_body_pose\",10);time.sleep(3);"
    "wrist=[0.0903,0.1615,-0.2411,0.7295,0.3145,0.5533,-0.2506,0.1280,-0.1522,-0.2461,0.7320,-0.2639,0.5395,0.3217];"
    "hand=[0.0]*7;"
    "pl={\"navigate_cmd\":[0,0,0],\"locomotion_mode\":0,\"base_height_command\":0.78,\"toggle_policy_action\":True,\"wrist_pose\":wrist,\"left_hand_joint\":hand,\"right_hand_joint\":hand};"
    "m=ByteMultiArray();m.data=[bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)];p.publish(m);time.sleep(2);"
    "n.destroy_node();rclpy.shutdown();print(\"OK\")'"],
    env=ENV)
print("[CTRL] Robot should be standing")

# 3. Sensors
run_script("sensor_pub.py", "SENSOR")
time.sleep(2)
run_script("mid360_pub.py", "MID360", SCENE_XML)
time.sleep(2)
run_script("camera_pub.py", "CAM", SCENE_XML)

# 4. Navigation
run_script("goal_follower.py", "NAV")

print()
print("=" * 45)
print("  Ready! Run RViz:")
print("    bash scripts/rviz.sh")
print("  Click 2D Goal Pose to navigate.")
print("  Ctrl+C to stop all.")
print("=" * 45)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    cleanup()
