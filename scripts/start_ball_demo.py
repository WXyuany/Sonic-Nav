#!/usr/bin/env python3
import os
import shlex
import signal
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO)

from gear_sonic.utils.mujoco_sim.launch_cleanup import cleanup_stale_sonic_processes
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help, set_wbc_scene


os.chdir(REPO)
ENV = os.environ.copy()
ENV.update({
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_DOMAIN_ID": "42",
    "DISPLAY": ":1",
    "SONIC_SKIP_LFS_PULL": "1",
})
GRASP_ASSIST_FILE = os.environ.get("SONIC_BOX_GRASP_ASSIST_FILE", "/tmp/sonic_box_grasp_assist.json")

procs = []
proc_names = {}


def add_proc(proc, name):
    procs.append(proc)
    proc_names[proc.pid] = name
    return proc


def _format_args(args=None):
    if args is None:
        return ""
    values = [args] if isinstance(args, (str, os.PathLike)) else list(args)
    return "".join(f" {shlex.quote(str(value))}" for value in values)


def run_script(script, name, args=None, startup_sleep=3.0):
    print(f"[{name}] Starting...")
    log_path = f"/tmp/sonic_{name.lower()}.log"
    script_path = os.path.join(REPO, "scripts", script)
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"export PYTHONPATH=\"{REPO}:{REPO}/g1_ros2_nav:$PYTHONPATH\" && "
        f"exec /usr/bin/python3 {shlex.quote(script_path)}{_format_args(args)}"
    )
    log = open(log_path, "w")
    proc = subprocess.Popen(["bash", "-c", cmd], env=ENV, stdout=log, stderr=subprocess.STDOUT)
    add_proc(proc, name)
    time.sleep(startup_sleep)
    if proc.poll() is not None:
        print(f"[{name}] FAILED, see {log_path}")
        raise RuntimeError(f"{name} exited with code {proc.returncode}")
    print(f"[{name}] Running")
    return proc


def stream_ball_demo_progress(demo_proc):
    log_path = "/tmp/sonic_ball_demo.log"
    offset = 0
    interesting = (
        "ZMQ publisher bound:",
        "ball contact-lock assist",
        "contact servo disabled",
        "contact servo enabled",
        "rollout log:",
        "sent start command:",
        "using ball anchor:",
        "approach reached:",
        "approach close enough:",
        "approach still far:",
        "approach failed:",
        "waiting for plausible ball anchor",
        "no plausible ball anchor",
        "IK right-hand poses applied:",
        "IK result rejected:",
        "skill graph:",
        "contact servo",
        "capture not ready",
        "palm pocket not ready",
        "workspace servo",
        "workspace alignment residual",
        "workspace still offset",
        "workspace aligned:",
        "lift reference:",
        "grasp geometry",
        "ball lifted:",
        "ball not lifted",
        "lift not stable:",
        "ball lift check",
        "phase:",
        "demo done",
        "no ball anchor received",
        "failed to apply ball anchor",
    )
    while True:
        if os.path.exists(log_path):
            with open(log_path) as log:
                log.seek(offset)
                lines = log.readlines()
                offset = log.tell()
            for raw in lines:
                line = raw.strip()
                if not any(token in line for token in interesting):
                    continue
                msg = line.split("[ball_pick_place_demo]", 1)[-1].strip()
                print(f"[BALL_DEMO] {msg}", flush=True)
                if "demo done; holding final single-hand place pose" in msg:
                    return "holding"
                if msg == "demo done":
                    return "done"

        if demo_proc.poll() is not None:
            return "exited"
        time.sleep(0.5)


def check_processes(ignore=None):
    ignore = set(ignore or [])
    for proc in list(procs):
        if proc.pid in ignore:
            continue
        code = proc.poll()
        if code is not None:
            name = proc_names.get(proc.pid, f"pid {proc.pid}")
            raise RuntimeError(f"{name} exited unexpectedly with code {code}")


def shutdown_children():
    print("\n[STOP] Shutting down...")
    for proc in reversed(procs):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("[STOP] Done")


def cleanup(*_):
    shutdown_children()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    scene_arg = sys.argv[1] if len(sys.argv) > 1 else "ball_demo"
    demo_args = sys.argv[2:]
    exit_after_demo = "--exit-after-demo" in demo_args
    demo_args = [arg for arg in demo_args if arg != "--exit-after-demo"]
    use_ball_anchor = "--no-ball-anchor" not in demo_args
    demo_args = [arg for arg in demo_args if arg != "--no-ball-anchor"]
    if use_ball_anchor and "--use-ball-anchor" not in demo_args:
        demo_args.append("--use-ball-anchor")
    if use_ball_anchor and "--require-ball-anchor" not in demo_args:
        demo_args.append("--require-ball-anchor")
    if exit_after_demo and "--hold" not in demo_args and "--no-hold" not in demo_args:
        demo_args.append("--no-hold")
    hold_final_pose = "--no-hold" not in demo_args
    if hold_final_pose and "--hold" not in demo_args:
        demo_args.append("--hold")
    try:
        selection = resolve_scene(scene_arg, repo_root=REPO)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc

    print("=" * 45)
    print(f"  Sonic Ball Pick-Place Demo | {selection.name}")
    print("=" * 45)

    stale_pids = cleanup_stale_sonic_processes()
    if stale_pids:
        print(f"[CLEANUP] Stopped stale Sonic processes: {stale_pids}")
    try:
        os.remove(GRASP_ASSIST_FILE)
    except FileNotFoundError:
        pass
    try:
        os.remove("/tmp/sonic_qpos.npy")
    except FileNotFoundError:
        pass

    set_wbc_scene(selection.abs_path, repo_root=REPO)
    print(f"[SCENE] Using {selection.name} ({selection.xml_file})")
    if not any(arg == "--scene" or str(arg).startswith("--scene=") for arg in demo_args):
        demo_args.extend(["--scene", str(selection.abs_path)])

    sim = subprocess.Popen(["bash", "-c",
        f"source {REPO}/.venv_sim/bin/activate && export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && exec python {REPO}/gear_sonic/scripts/run_sim_loop.py"],
        env=ENV, stdout=open("/tmp/sonic_sim.log", "w"), stderr=subprocess.STDOUT)
    add_proc(sim, "SIM")
    print("[SIM] Starting...")
    time.sleep(8)
    check_processes()
    print("[SIM] Running")

    open("/tmp/sonic_deploy.log", "w").close()
    deploy = subprocess.Popen(["bash", "-c",
        f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh >/dev/null 2>&1 && cd {REPO}/gear_sonic_deploy && "
        f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all "
        f"--zmq-host localhost --set-compliance 0.05 --max-close-ratio 0.85 --disable-crc-check"],
        env=ENV, stdout=open("/tmp/sonic_deploy.log", "w"), stderr=subprocess.STDOUT)
    add_proc(deploy, "DEPLOY")
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
        check_processes()
    if not deploy_ready:
        raise RuntimeError("DEPLOY did not report Init Done within 120s; see /tmp/sonic_deploy.log")
    print("[DEPLOY] Init Done!")
    print("[CTRL] Start/planner command will be sent by the ZMQ ball demo")

    run_script("perception/sensor_pub.py", "SENSOR", startup_sleep=1.0)
    run_script(
        "perception/camera_pub.py",
        "CAM",
        [str(selection.abs_path), "--fps", "20", "--depth-fps", "20"],
        startup_sleep=1.0,
    )
    if use_ball_anchor:
        run_script("perception/ball_anchor_pub.py", "BALL_ANCHOR", [str(selection.abs_path)], startup_sleep=1.0)
        run_script("tools/world_model_node.py", "WORLD_MODEL", startup_sleep=1.0)
        run_script("tools/world_model_recovery_coordinator.py", "WORLD_RECOVERY", startup_sleep=1.0)
        run_script("tools/world_model_executor.py", "WORLD_EXECUTOR", startup_sleep=1.0)

    demo_proc = run_script("manipulation/ball_pick_place_demo.py", "BALL_DEMO", demo_args, startup_sleep=1.0)
    print("[BALL_DEMO] Sequence progress follows:")
    progress_state = stream_ball_demo_progress(demo_proc)
    if hold_final_pose:
        if progress_state == "holding":
            print("[BALL_DEMO] Sequence completed; final right-hand place pose is being held until Ctrl+C.")
        elif demo_proc.poll() is not None:
            raise RuntimeError(f"BALL_DEMO exited with code {demo_proc.returncode}; see /tmp/sonic_ball_demo.log")
        else:
            print("[BALL_DEMO] Holding mode enabled; monitoring process until Ctrl+C.")
        while True:
            time.sleep(1)
            check_processes()

    print("[BALL_DEMO] Waiting for sequence to finish...")
    while demo_proc.poll() is None:
        time.sleep(1)
        check_processes(ignore={demo_proc.pid})

    if demo_proc.returncode != 0:
        raise RuntimeError(f"BALL_DEMO exited with code {demo_proc.returncode}; see /tmp/sonic_ball_demo.log")

    print()
    print("=" * 45)
    print("  Ball pick-place sequence completed.")
    print("  Ctrl+C to stop the simulator/deploy stack.")
    print("=" * 45)

    if exit_after_demo:
        shutdown_children()
        return

    while True:
        time.sleep(1)
        check_processes(ignore={demo_proc.pid})


if __name__ == "__main__":
    main()
