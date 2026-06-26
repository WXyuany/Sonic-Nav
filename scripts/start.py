#!/usr/bin/env python3
"""One-click Sonic-Nav launch."""
import os, sys, time, signal, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO)
ENV = os.environ.copy()
ENV.update({"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "ROS_LOCALHOST_ONLY": "1",
            "ROS_DOMAIN_ID": "42", "DISPLAY": ":1"})

procs = []

def run(cmd, name, wait_for=None, timeout=120):
    print(f"[{name}] Starting...")
    p = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    if wait_for:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                print(f"[{name}] CRASHED"); sys.exit(1)
            time.sleep(0.5)
    print(f"[{name}] Running")

def run_script(script, name):
    print(f"[{name}] Starting...")
    cmd = f"source /opt/ros/humble/setup.bash && exec /usr/bin/python3 {REPO}/scripts/{script}"
    p = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(3)
    print(f"[{name}] Running")

def wait_deploy_log(pattern, timeout=120):
    t0 = time.time()
    log = "/tmp/sonic_deploy.log"
    while time.time() - t0 < timeout:
        if os.path.exists(log):
            with open(log) as f:
                if pattern in f.read(): return True
        time.sleep(0.5)
    return False

def cleanup(*_):
    print("\n[STOP] Shutting down...")
    for p in reversed(procs):
        try: p.terminate(); p.wait(timeout=5)
        except: p.kill()
    print("[STOP] Done")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("=" * 45)
print("  Sonic-Nav  |  DOMAIN=42")
print("=" * 45)

# 1. Sim
run(f"source {REPO}/.venv_sim/bin/activate && export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && exec python {REPO}/gear_sonic/scripts/run_sim_loop.py", "SIM", timeout=120)
time.sleep(6)

# 2. Deploy
open("/tmp/sonic_deploy.log", "w").close()
cmd = (f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh >/dev/null 2>&1 && cd {REPO}/gear_sonic_deploy && "
       f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
       f"--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx "
       f"--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type ros2 --output-type all "
       f"--zmq-host localhost --disable-crc-check")
run_log = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=open("/tmp/sonic_deploy.log", "w"), stderr=subprocess.STDOUT)
procs.append(run_log)
print("[DEPLOY] Starting...")

if not wait_deploy_log("Init Done"):
    print("[DEPLOY] Init timeout"); cleanup(); sys.exit(1)
print("[DEPLOY] Init Done!")

# 3. Sensor bridge
run_script("sensor_pub.py", "SENSOR")
time.sleep(3)

# 4. MPPI Navigation
run_script("mppi_nav.py", "MPPI")
time.sleep(3)

print()
print("=" * 45)
print("  Ready! Run RViz:")
print("    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
print("    export ROS_LOCALHOST_ONLY=1")
print("    export ROS_DOMAIN_ID=42")
print("    source /opt/ros/humble/setup.bash")
print("    rviz2")
    print("  Click 2D Goal Pose to navigate.")
    print("  For MPPI nav: use scripts/mppi_nav.py instead")
    print("  Ctrl+C to stop all.")
print("=" * 45)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    cleanup()
