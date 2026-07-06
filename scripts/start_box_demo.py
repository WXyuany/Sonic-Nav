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


def stream_box_demo_progress(demo_proc):
    log_path = "/tmp/sonic_box_demo.log"
    offset = 0
    interesting = (
        "ZMQ publisher bound:",
        "box attach assist",
        "sent start command:",
        "using box anchor:",
        "approach reached:",
        "approach close enough:",
        "approach still far:",
        "approach failed:",
        "waiting for plausible box anchor",
        "no plausible box anchor",
        "IK upper-body poses applied:",
        "IK result rejected:",
        "lift reference:",
        "box lifted:",
        "box not lifted",
        "box lift check",
        "phase:",
        "demo done",
        "no box anchor received",
        "failed to apply box anchor",
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
                msg = line.split("[box_grasp_demo]", 1)[-1].strip()
                print(f"[BOX_DEMO] {msg}", flush=True)
                if "demo done; holding final box pose" in msg:
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


def cleanup(*_):
    print("\n[STOP] Shutting down...")
    for proc in reversed(procs):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("[STOP] Done")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    scene_arg = sys.argv[1] if len(sys.argv) > 1 else "box_demo"
    demo_args = sys.argv[2:]
    use_box_anchor = "--no-box-anchor" not in demo_args
    demo_args = [arg for arg in demo_args if arg != "--no-box-anchor"]
    if use_box_anchor and "--use-box-anchor" not in demo_args:
        demo_args.append("--use-box-anchor")
    if use_box_anchor and "--require-box-anchor" not in demo_args:
        demo_args.append("--require-box-anchor")
    hold_final_pose = "--no-hold" not in demo_args
    if hold_final_pose and "--hold" not in demo_args:
        demo_args.append("--hold")
    try:
        selection = resolve_scene(scene_arg, repo_root=REPO)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc

    print("=" * 45)
    print(f"  Sonic Box Demo | {selection.name}")
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
        f"--zmq-host localhost --set-compliance 0.05 --max-close-ratio 0.2 --disable-crc-check"],
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
    print("[CTRL] Start/planner command will be sent by the ZMQ box demo")

    run_script("sensor_pub.py", "SENSOR", startup_sleep=1.0)
    run_script(
        "camera_pub.py",
        "CAM",
        [str(selection.abs_path), "--fps", "20", "--depth-fps", "20"],
        startup_sleep=1.0,
    )
    if use_box_anchor:
        run_script("box_anchor_pub.py", "BOX_ANCHOR", [str(selection.abs_path)], startup_sleep=1.0)

    demo_proc = run_script("box_grasp_demo.py", "BOX_DEMO", demo_args, startup_sleep=1.0)
    print("[BOX_DEMO] Sequence progress follows:")
    progress_state = stream_box_demo_progress(demo_proc)
    if hold_final_pose:
        if progress_state == "holding":
            print("[BOX_DEMO] Sequence completed; final contact grasp pose is being held until Ctrl+C.")
        elif demo_proc.poll() is not None:
            raise RuntimeError(f"BOX_DEMO exited with code {demo_proc.returncode}; see /tmp/sonic_box_demo.log")
        else:
            print("[BOX_DEMO] Holding mode enabled; monitoring process until Ctrl+C.")
        while True:
            time.sleep(1)
            check_processes()

    print("[BOX_DEMO] Waiting for sequence to finish...")
    while demo_proc.poll() is None:
        time.sleep(1)
        check_processes(ignore={demo_proc.pid})

    if demo_proc.returncode != 0:
        raise RuntimeError(f"BOX_DEMO exited with code {demo_proc.returncode}; see /tmp/sonic_box_demo.log")

    print()
    print("=" * 45)
    print("  Box demo sequence completed.")
    print("  Ctrl+C to stop the simulator/deploy stack.")
    print("=" * 45)

    while True:
        time.sleep(1)
        check_processes(ignore={demo_proc.pid})


if __name__ == "__main__":
    main()
