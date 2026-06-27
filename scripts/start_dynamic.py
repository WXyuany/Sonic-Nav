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
print("  Sonic-Nav DYNAMIC |  DOMAIN=42")
print("=" * 45)

# 1. Sim (dynamic scene)
sim = subprocess.Popen(["bash", "-c",
    f"cp {REPO}/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml /tmp/g1_tmp.yaml && "
    f"sed -i 's|scene_43dof.xml|scene_dynamic.xml|' /tmp/g1_tmp.yaml && "
    f"cd {REPO} && source .venv_sim/bin/activate && export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && "
    f"exec python -c \"import sys; sys.path.insert(0,'{REPO}'); "
    f"from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig; "
    f"from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel; "
    f"from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model; "
    f"import yaml; cfg = SimLoopConfig().load_wbc_yaml(); cfg['ROBOT_SCENE']='gear_sonic/data/robot_model/model_data/g1/scene_dynamic.xml'; "
    f"cfg['ENV_NAME']='default'; robot=instantiate_g1_robot_model(); init_channel(cfg); "
    f"sim=SimulatorFactory.create_simulator(config=cfg, env_name='default', onscreen=True); "
    f"SimulatorFactory.start_simulator(sim, as_thread=False); \""],
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
print("    bash scripts/rviz.sh")
print("  Click 2D Goal Pose to navigate.")
print("  Ctrl+C to stop all.")
print("=" * 45)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    cleanup()
