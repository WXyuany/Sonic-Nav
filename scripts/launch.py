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
})
DEPLOY_LOG = "/tmp/sonic_deploy.log"
processes = []


def log(tag, msg):
    print(f"[{tag:>6}] {msg}", flush=True)


def start_sim():
    log("SIM", "Starting MuJoCo...")
    sim_cmd = (
        f"source {REPO}/.venv_sim/bin/activate && "
        f"export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && "
        f"exec python {REPO}/gear_sonic/scripts/run_sim_loop.py"
    )
    try:
        subprocess.run(["tmux", "kill-session", "-t", "sonic-sim"], capture_output=True)
    except Exception:
        pass
    subprocess.run(["tmux", "new-session", "-d", "-s", "sonic-sim", sim_cmd], check=True)
    time.sleep(6)
    log("SIM", "Running (tmux: sonic-sim, attach: tmux a -t sonic-sim)")


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


def wait_for(pattern, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(DEPLOY_LOG):
            with open(DEPLOY_LOG) as f:
                if pattern in f.read():
                    return True
        time.sleep(0.5)
    return False


def start_bridge():
    log("BRIDGE", "Starting cmd_vel bridge...")
    cmd = (f"source /opt/ros/humble/setup.bash && "
           f"source ~/ros2_ws/install/setup.bash && "
           f"exec ros2 run g1_ros2_nav cmd_vel_bridge")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(("bridge", proc))
    log("BRIDGE", "Running")


def cleanup():
    log("STOP", "Shutting down...")
    subprocess.run(["tmux", "kill-session", "-t", "sonic-sim"], capture_output=True)
    for _, proc in reversed(processes):
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log("STOP", "Done")


def main():
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    print("=" * 50)
    print("  Sonic-Nav  |  DOMAIN=42  RMW=fastrtps")
    print("=" * 50)

    start_sim()
    start_deploy()

    log("WAIT", "Waiting for Init Done (~30s)...")
    if not wait_for("Init Done"):
        log("ERROR", "Deploy failed to init")
        cleanup()
        sys.exit(1)
    log("DEPLOY", "Init Done!")

    start_bridge()

    print()
    print("=" * 50)
    print("  Sonic-Nav Ready!")
    print()
    print("  Keyboard (start + walk):")
    print("    /usr/bin/python3 scripts/keyboard_control.py")
    print()
    print("  Keyboard will auto-start control,")
    print("  then W=forward S=back A/D=strafe Q/E=turn")
    print("  Ctrl+C to stop all")
    print("=" * 50)

    try:
        signal.pause()
    except AttributeError:
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
