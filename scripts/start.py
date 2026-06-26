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

# 2. Deploy proxy (keyboard mode + ROS2 → keys)
rm -f /tmp/proxy_ready
run = subprocess.Popen(["bash", "-c",
    f"source /opt/ros/humble/setup.bash && exec /usr/bin/python3 {REPO}/scripts/deploy_proxy.py"],
    env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
procs.append(run)
print("[PROXY] Starting...")
t0 = time.time()
while time.time() - t0 < 120:
    if os.path.exists("/tmp/proxy_ready"):
        print("[PROXY] Ready!")
        break
    if run.poll() is not None:
        print(f"[PROXY] CRASHED:\n{run.stdout.read()[-500:]}")
        cleanup(); sys.exit(1)
    time.sleep(1)

# 3. Sensor bridge
run_script("sensor_pub.py", "SENSOR")
time.sleep(3)

# 4. Goal follower
run_script("goal_follower.py", "NAV")
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
print("  Ctrl+C to stop all.")
print("=" * 45)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    cleanup()
