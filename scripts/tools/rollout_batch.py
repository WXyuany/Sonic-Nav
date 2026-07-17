#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO = SCRIPTS_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gear_sonic.utils.mujoco_sim.launch_cleanup import cleanup_stale_sonic_processes
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, set_wbc_scene


DEFAULT_SCENES = {
    "ball": "ball_demo",
    "box": "box_demo",
}
DEFAULT_PREFIXES = {
    "ball": "ball_test",
    "box": "box_test",
}
ENV = os.environ.copy()
ENV.update(
    {
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_LOCALHOST_ONLY": "1",
        "ROS_DOMAIN_ID": "42",
        "DISPLAY": ":1",
        "SONIC_SKIP_LFS_PULL": "1",
    }
)
ANCHOR_SCRIPTS = {
    "ball": "perception/ball_anchor_pub.py",
    "box": "perception/box_anchor_pub.py",
}
DEMO_SCRIPTS = {
    "ball": "manipulation/ball_pick_place_demo.py",
    "box": "manipulation/box_grasp_demo.py",
}
DEMO_NODE_TAGS = {
    "ball": "[ball_pick_place_demo]",
    "box": "[box_grasp_demo]",
}
PROGRESS_TOKENS = (
    "rollout log:",
    "sent start command:",
    "using ball anchor:",
    "using box anchor:",
    "approach reached:",
    "approach still far:",
    "workspace alignment residual",
    "workspace servo",
    "workspace response sign update",
    "pregrasp base align disabled",
    "capture not ready",
    "palm pocket not ready",
    "lift reference:",
    "grasp geometry",
    "contact servo",
    "workspace response sign update",
    "ball lifted:",
    "box lifted:",
    "ball not lifted",
    "box not lifted",
    "lift not stable:",
    "task marked failed:",
    "phase:",
    "demo done",
)
ANCHOR_LOSS_REASONS = {
    "missing_or_implausible_anchor",
    "apply_anchor_exception",
    "object_out_of_workspace_z",
    "object_out_of_workspace_y",
}
BASE_SUMMARY_RE = re.compile(r"\b(?P<kind>ball|box)_base=\((?P<xyz>[^)]*)\)")

procs: list[subprocess.Popen] = []
proc_names: dict[int, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated Sonic real-controller rollouts without manually restarting each demo. "
            "By default one simulator/deploy stack is reused so each rollout continues from the previous state."
        )
    )
    parser.add_argument("demo", choices=sorted(DEFAULT_SCENES), help="Which real rollout launcher to run.")
    parser.add_argument("--scene", help="Scene name or XML path. Defaults to <demo>_demo.")
    parser.add_argument("--runs", type=int, default=3, help="Number of rollout runs.")
    parser.add_argument("--prefix", help="Run-id prefix. Defaults to ball_test or box_test.")
    parser.add_argument("--start-index", type=int, help="First numeric suffix. Defaults to the next free index.")
    parser.add_argument("--width", type=int, default=3, help="Zero-pad width for generated run ids.")
    parser.add_argument("--report-path", default="reports/rollouts")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--restart-each", action="store_true", help="Start a fresh simulator stack per rollout.")
    parser.add_argument("--headless", action="store_true", help="Run MuJoCo physics without onscreen GUI.")
    parser.add_argument("--camera", action="store_true", default=True, help="Start camera publisher.")
    parser.add_argument("--no-camera", dest="camera", action="store_false")
    parser.add_argument("--dynamic-place-target", action="store_true", default=True)
    parser.add_argument("--fixed-place-target", dest="dynamic_place_target", action="store_false")
    parser.add_argument("--settle-between", type=float, default=1.0)
    parser.add_argument(
        "--anchor-object-name",
        help=(
            "Object geom/site name passed to the demo anchor publisher. "
            "For generated task scenes this is usually <object_id>_geom."
        ),
    )
    parser.add_argument(
        "--anchor-place-site",
        help=(
            "Place target site name passed to the ball anchor publisher. "
            "For generated task scenes this is usually <target_id>_site."
        ),
    )
    parser.add_argument(
        "--anchor-approach-standoff",
        type=float,
        help="Approach standoff passed to the object anchor publisher for generated benchmark scenes.",
    )
    parser.add_argument(
        "--reset-each-rollout",
        action="store_true",
        help=(
            "Reset MuJoCo robot/object state between rollouts while keeping the simulator, deploy, "
            "sensor, and world-model processes alive."
        ),
    )
    parser.add_argument(
        "--reset-on-anchor-loss",
        dest="reset_on_anchor_loss",
        action="store_true",
        default=True,
        help="When a reused stack reaches an invalid object-anchor state, reset the MuJoCo scene instead of restarting.",
    )
    parser.add_argument(
        "--no-reset-on-anchor-loss",
        dest="reset_on_anchor_loss",
        action="store_false",
        help="Leave the stack in its current state when the object anchor is lost.",
    )
    parser.add_argument(
        "--max-consecutive-anchor-loss",
        type=int,
        default=2,
        help="Stop a reused batch after this many terminal anchor states. Use 0 to never auto-stop.",
    )
    parser.add_argument(
        "--terminal-anchor-min-z",
        type=float,
        default=-0.35,
        help="Base-frame object z below this value is treated as dropped/outside the tabletop workspace.",
    )
    parser.add_argument("--reset-settle", type=float, default=2.0, help="Seconds to wait after a scene reset request.")
    parser.add_argument("--sim-reset-file", default=os.environ.get("SONIC_SIM_RESET_FILE", "/tmp/sonic_sim_reset.json"))
    parser.add_argument(
        "--world-policy-backend",
        choices=["heuristic", "memory", "learned"],
        default="heuristic",
        help="Policy backend passed to tools/world_model_node.py.",
    )
    parser.add_argument("--world-policy-model", help="Policy model JSON passed to tools/world_model_node.py.")
    parser.add_argument("--world-runtime-override-file", default="", help="JSON primitive override mapping passed to tools/world_model_node.py.")
    parser.add_argument("--vlm-anchor-bridge", action="store_true", help="Start the VLM/RGB-D detection to object-anchor bridge.")
    parser.add_argument("--vlm-detections-topic", default="/sonic_world/vlm_detections")
    parser.add_argument(
        "--qwen-vl-shadow",
        action="store_true",
        help=(
            "Start local Qwen-VL, RGB-D anchor fusion, and paired privileged-anchor recording. "
            "Shadow anchors never drive /sonic_world/object_anchor."
        ),
    )
    parser.add_argument(
        "--qwen-vl-model-path",
        default="models/Qwen2.5-VL-3B-Instruct",
        help="Local Qwen2.5-VL model directory used by --qwen-vl-shadow.",
    )
    parser.add_argument("--qwen-vl-host", default="127.0.0.1")
    parser.add_argument("--qwen-vl-port", type=int, default=8000)
    parser.add_argument(
        "--qwen-vl-period",
        type=float,
        default=5.0,
        help="Seconds between Qwen detections in shadow mode; use 0 for recovery-request-only inference.",
    )
    parser.add_argument("--qwen-vl-max-new-tokens", type=int, default=512)
    parser.add_argument("--qwen-vl-device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--qwen-vl-gpu-memory-gib", type=float, default=0.0)
    parser.add_argument("--qwen-vl-depth-cache-size", type=int, default=480, help="RGB-D frames retained for delayed Qwen responses; 480 covers about 24s at 20Hz.")
    parser.add_argument("--qwen-vl-health-timeout", type=float, default=180.0)
    parser.add_argument("--qwen-vl-shadow-dir", default="reports/perception")
    parser.add_argument(
        "--qwen-vl-calibration-file",
        default="",
        help="Optional validated RGB-D translation calibration from world_model_rgbd_anchor_calibrator.py.",
    )
    parser.add_argument("--qwen-vl-audit-output", default="", help="Optional local Qwen response JSONL audit path; no image data is recorded.")
    parser.add_argument("--qwen-vl-calibration-max-samples", type=int, default=20)
    parser.add_argument("--qwen-vl-temporal-window", type=int, default=3)
    parser.add_argument("--qwen-vl-temporal-min-observations", type=int, default=3)
    parser.add_argument(
        "--visual-auto-reobserve",
        action="store_true",
        help="Request bounded perception recovery when required visual objects are absent after RGB-D fusion.",
    )
    parser.add_argument("--visual-reobserve-max-attempts", type=int, default=2)
    parser.add_argument("--visual-reobserve-cooldown-s", type=float, default=1.0)
    parser.add_argument("--visual-recovery-escalate-navigation", action="store_true")
    parser.add_argument(
        "--perception-shadow-backend",
        choices=["qwen", "grounding_dino", "hsv"],
        default="qwen",
        help="2D detector used by --qwen-vl-shadow. Grounding DINO and HSV are local detector baselines.",
    )
    parser.add_argument("--grounding-dino-model-path", default="models/grounding-dino-tiny")
    parser.add_argument("--grounding-dino-host", default="127.0.0.1")
    parser.add_argument("--grounding-dino-port", type=int, default=8001)
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-dino-object-label", default="", help="Optional task-conditioned Grounding DINO text label for the primary object.")
    parser.add_argument("--grounding-dino-target-label", default="", help="Optional task-conditioned Grounding DINO text label for the place target.")
    parser.add_argument(
        "--qwen-vl-instruction",
        default="",
        help="Optional task-conditioned Qwen detection instruction. Defaults to the active object and target IDs.",
    )
    parser.add_argument(
        "--qwen-vl-gate-report",
        help=(
            "Passing VLM anchor-evaluation report. When supplied, promote Qwen/RGB-D anchors through "
            "a validated relay into /sonic_world/object_anchor."
        ),
    )
    parser.add_argument(
        "--qwen-vl-shadow-max-pairs",
        type=int,
        default=0,
        help="Stop the shadow recorder after this many pairs; 0 records for the whole batch.",
    )
    parser.add_argument("--world-primitive-runner", action="store_true", help="Start tools/world_model_primitive_runner.py.")
    parser.add_argument(
        "--world-teacher-ball-attach",
        action="store_true",
        help="Enable ball contact-lock only for teacher-data collection; never valid for physical benchmark promotion.",
    )
    parser.add_argument(
        "--world-teacher-lift-attach",
        action="store_true",
        help="Teacher-data mode that preserves real side-grasp evidence and only attaches the ball for lift_object.",
    )
    parser.add_argument(
        "--world-primitive-backend",
        choices=["status_only", "zmq_phase"],
        default="status_only",
        help="Primitive runner backend. Use zmq_phase only when no separate demo process owns the ZMQ planner port.",
    )
    parser.add_argument(
        "--autonomous-world-execution",
        action="store_true",
        help="Do not launch the legacy demo script; publish TaskRequest and wait for world-model primitive statuses.",
    )
    parser.add_argument(
        "--autonomous-primitive-backend",
        choices=["status_only", "zmq_phase"],
        default="zmq_phase",
        help="Primitive backend used when --autonomous-world-execution starts the primitive runner.",
    )
    parser.add_argument("--task-id", default="")
    parser.add_argument("--task-verb", default="")
    parser.add_argument("--task-object-id", default="")
    parser.add_argument("--task-object-category", default="")
    parser.add_argument("--task-target-id", default="")
    parser.add_argument("--autonomous-timeout", type=float, default=120.0)
    parser.add_argument("--episode-manifest", help="Run one carry-state world-model episode manifest on a single reusable stack.")
    parser.add_argument("--episode-output-jsonl", default="", help="Episode event log; defaults below --report-path.")
    parser.add_argument(
        "--episode-start-settle",
        type=float,
        default=5.0,
        help="Seconds to keep the deployed controller and sensors stable before launching a carry-state episode.",
    )
    parser.add_argument("--episode-timeout-per-stage", type=float, default=120.0)
    parser.add_argument("--episode-continue-on-failure", action="store_true")
    parser.add_argument("--episode-stage-start", type=int, default=1, help="First manifest stage_index to execute (inclusive).")
    parser.add_argument("--episode-stage-stop", type=int, default=0, help="Last manifest stage_index to execute (inclusive); 0 runs all remaining stages.")
    parser.add_argument(
        "--episode-curriculum-stage",
        type=int,
        default=0,
        help="Reset MuJoCo and restore the selected manifest stage's freejoint objects before the episode; 0 disables curriculum reset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python",
        default="/usr/bin/python3",
        help="Python used for ROS/demo children. ROS Humble rclpy requires the system Python by default.",
    )
    parser.add_argument(
        "--fail-on-rollout-fail",
        action="store_true",
        help="Return a non-zero process exit code if any rollout task failed. By default the batch completes and reports failures.",
    )
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--no-report", dest="report", action="store_false")
    args, demo_args = parser.parse_known_args()
    args.demo_args = demo_args
    return args


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.qwen_vl_shadow and not args.camera:
        raise SystemExit("--qwen-vl-shadow requires camera publishing; remove --no-camera")
    if not 1 <= int(args.qwen_vl_port) <= 65535:
        raise SystemExit("--qwen-vl-port must be between 1 and 65535")
    if args.qwen_vl_shadow and float(args.qwen_vl_period) < 0.0:
        raise SystemExit("--qwen-vl-period must be non-negative")
    if args.qwen_vl_gate_report and not args.qwen_vl_shadow:
        raise SystemExit("--qwen-vl-gate-report requires --qwen-vl-shadow")
    if args.qwen_vl_calibration_file and not _repo_path(args.qwen_vl_calibration_file).exists():
        raise SystemExit(f"--qwen-vl-calibration-file does not exist: {_repo_path(args.qwen_vl_calibration_file)}")
    if args.episode_manifest:
        manifest = _repo_path(args.episode_manifest)
        if not manifest.exists():
            raise SystemExit(f"--episode-manifest does not exist: {manifest}")
        args.autonomous_world_execution = True
        args.world_primitive_runner = True
        args.reset_each_rollout = False
        if int(args.episode_curriculum_stage) < 0:
            raise SystemExit("--episode-curriculum-stage must be non-negative")
        if int(args.episode_stage_start) < 1 or int(args.episode_stage_stop) < 0:
            raise SystemExit("--episode-stage-start must be positive and --episode-stage-stop must be non-negative")
        if int(args.episode_stage_stop) and int(args.episode_stage_stop) < int(args.episode_stage_start):
            raise SystemExit("--episode-stage-stop must be greater than or equal to --episode-stage-start")

    scene = args.scene or DEFAULT_SCENES[args.demo]
    prefix = args.prefix or DEFAULT_PREFIXES[args.demo]
    start_index = args.start_index if args.start_index is not None else _next_index(prefix, _repo_path(args.report_path))
    extra_args = _clean_demo_args(args.demo_args)
    report_script = SCRIPT_DIR / "rollout_report.py"

    failures: list[tuple[str, int]] = []
    run_ids = [f"{prefix}_{start_index + offset:0{max(1, args.width)}d}" for offset in range(args.runs)]

    if args.restart_each:
        failures = _run_restart_each(args, scene, run_ids, extra_args)
    else:
        failures = _run_reuse_stack(args, scene, run_ids, extra_args)

    if args.report and not args.dry_run:
        report_cmd = [args.python, str(report_script), args.report_path]
        print()
        print("[BATCH] Updating rollout report...")
        subprocess.call(report_cmd, cwd=REPO, env=_child_env())

    if failures:
        print()
        print("[BATCH] Failed runs:")
        for run_id, code in failures:
            print(f"  {run_id}: exit_code={code}")
        if args.fail_on_rollout_fail or args.stop_on_fail:
            return 1
    return 0


def _run_restart_each(
    args: argparse.Namespace,
    scene: str,
    run_ids: list[str],
    extra_args: list[str],
) -> list[tuple[str, int]]:
    launcher = SCRIPTS_DIR / f"start_{args.demo}_demo.py"
    failures: list[tuple[str, int]] = []
    for index, run_id in enumerate(run_ids, start=1):
        cmd = [
            args.python,
            str(launcher),
            scene,
            "--rollout-id",
            run_id,
            "--no-hold",
            "--exit-after-demo",
            *extra_args,
        ]
        _print_run_header(args.demo, index, len(run_ids), run_id, cmd, mode="restart-each")
        if args.dry_run:
            continue
        code = subprocess.call(cmd, cwd=REPO, env=_child_env())
        if code != 0:
            failures.append((run_id, code))
            print(f"[BATCH] {run_id} failed with exit code {code}")
            if args.stop_on_fail:
                break
    return failures


def _run_reuse_stack(
    args: argparse.Namespace,
    scene: str,
    run_ids: list[str],
    extra_args: list[str],
) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    selection = resolve_scene(scene, repo_root=REPO)
    if args.episode_manifest:
        return _run_episode_stack(args, selection)
    if args.dry_run:
        for index, run_id in enumerate(run_ids, start=1):
            cmd = (
                _autonomous_cmd(args, run_id)
                if args.autonomous_world_execution
                else _demo_cmd(args, selection.abs_path, run_id, extra_args)
            )
            _print_run_header(args.demo, index, len(run_ids), run_id, cmd, mode="reuse-stack")
        return failures

    _install_signal_handlers()
    _start_stack(args, selection)
    consecutive_anchor_loss = 0
    try:
        for index, run_id in enumerate(run_ids, start=1):
            cmd = (
                _autonomous_cmd(args, run_id)
                if args.autonomous_world_execution
                else _demo_cmd(args, selection.abs_path, run_id, extra_args)
            )
            _print_run_header(args.demo, index, len(run_ids), run_id, cmd, mode="reuse-stack")
            if args.autonomous_world_execution:
                code = subprocess.call(cmd, cwd=REPO, env=_child_env())
                outcome = _read_autonomous_outcome(args, run_id)
            else:
                try:
                    proc = _run_script_cmd(cmd, f"{args.demo.upper()}_DEMO", startup_sleep=1.0)
                except RuntimeError as exc:
                    failures.append((run_id, 1))
                    print(f"[BATCH] {run_id} failed during startup: {exc}")
                    if args.reset_each_rollout:
                        _request_sim_reset(args, reason=f"{run_id}:startup_failed")
                    if args.stop_on_fail:
                        break
                    continue
                code = _stream_demo_progress(args.demo, proc)
                outcome = _read_rollout_outcome(args.demo, run_id, args.report_path)
            terminal_anchor_issue = _terminal_anchor_issue(args, outcome)
            if code != 0:
                failures.append((run_id, code))
                print(f"[BATCH] {run_id} failed with exit code {code}")
                if args.stop_on_fail and not terminal_anchor_issue:
                    break
            if terminal_anchor_issue:
                if not args.reset_each_rollout:
                    consecutive_anchor_loss += 1
                print(f"[BATCH] terminal anchor state after {run_id}: {terminal_anchor_issue}")
                if args.reset_on_anchor_loss:
                    _request_sim_reset(args, reason=f"{run_id}:{terminal_anchor_issue}")
                if not args.reset_each_rollout and (
                    args.max_consecutive_anchor_loss > 0
                    and consecutive_anchor_loss >= args.max_consecutive_anchor_loss
                ):
                    print(
                        "[BATCH] stopping: repeated terminal anchor states "
                        f"({consecutive_anchor_loss}/{args.max_consecutive_anchor_loss})"
                    )
                    break
            else:
                consecutive_anchor_loss = 0
            if args.reset_each_rollout and not terminal_anchor_issue:
                _request_sim_reset(args, reason=f"{run_id}:rollout_end")
            time.sleep(max(0.0, float(args.settle_between)))
    finally:
        _shutdown_children()
    return failures


def _run_episode_stack(args: argparse.Namespace, selection: Any) -> list[tuple[str, int]]:
    manifest = _repo_path(args.episode_manifest)
    run_id = _safe_name(Path(manifest).stem)
    cmd = _episode_cmd(args, manifest)
    if args.dry_run:
        _print_run_header(args.demo, 1, 1, run_id, cmd, mode="carry-state-episode")
        return []
    _install_signal_handlers()
    try:
        _start_stack(args, selection)
        if int(args.episode_curriculum_stage) > 0:
            _request_episode_curriculum_reset(args, manifest, int(args.episode_curriculum_stage))
        settle_s = max(0.0, float(args.episode_start_settle))
        if settle_s:
            print(f"[EPISODE] Settling controller and sensors for {settle_s:.1f}s...")
            time.sleep(settle_s)
        _print_run_header(args.demo, 1, 1, run_id, cmd, mode="carry-state-episode")
        output = _episode_output(args, manifest)
        diagnostic = output.with_suffix(output.suffix + ".launcher.log")
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        with diagnostic.open("w", encoding="utf-8") as handle:
            wrapped = [
                "bash", "-c",
                "source /opt/ros/humble/setup.bash && "
                f"export PYTHONPATH=\"{REPO}:{REPO}/g1_ros2_nav:$PYTHONPATH\" && "
                "exec " + " ".join(_shell_quote(part) for part in cmd),
            ]
            code = subprocess.run(wrapped, cwd=REPO, env=_child_env(), stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
        if code == 0 and not _has_episode_terminal(output):
            print(f"[EPISODE] invalid completion: no episode_terminal in {output}; see {diagnostic}")
            code = 2
        return [] if code == 0 else [(run_id, int(code))]
    finally:
        _shutdown_children()


def _start_stack(args: argparse.Namespace, selection: Any) -> None:
    stale_pids = cleanup_stale_sonic_processes()
    if stale_pids:
        print(f"[CLEANUP] Stopped stale Sonic processes: {stale_pids}")
    ENV["SONIC_SIM_RESET_FILE"] = str(_repo_path(args.sim_reset_file))
    _remove_if_exists("/tmp/sonic_qpos.npy")
    _remove_if_exists(os.environ.get("SONIC_BOX_GRASP_ASSIST_FILE", "/tmp/sonic_box_grasp_assist.json"))
    _remove_if_exists(_repo_path(args.sim_reset_file))

    set_wbc_scene(selection.abs_path, repo_root=REPO)
    print("=" * 72)
    print(f"[STACK] Reusing one Sonic stack for {args.runs} {args.demo} rollouts")
    print(f"[SCENE] {selection.name} ({selection.xml_file})")
    print("=" * 72)

    sim_cmd = [
        "bash",
        "-c",
        (
            f"source {REPO}/.venv_sim/bin/activate && "
            f"export PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' DISPLAY=:1 && "
            f"exec python {REPO}/gear_sonic/scripts/run_sim_loop.py{_sim_cli_args(args)}"
        ),
    ]
    _run_raw(sim_cmd, "SIM", "/tmp/sonic_sim.log", startup_sleep=8.0)
    _check_processes()

    deploy_cmd = [
        "bash",
        "-c",
        (
            f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh >/dev/null 2>&1 && "
            f"cd {REPO}/gear_sonic_deploy && "
            "exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
            "--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx "
            "--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all "
            f"--zmq-host localhost --set-compliance 0.05 --max-close-ratio {_deploy_max_close_ratio(args.demo)} --disable-crc-check"
        ),
    ]
    Path("/tmp/sonic_deploy.log").write_text("", encoding="utf-8")
    _run_raw(deploy_cmd, "DEPLOY", "/tmp/sonic_deploy.log", startup_sleep=0.0)
    print("[DEPLOY] Waiting for Init Done...")
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if "Init Done" in Path("/tmp/sonic_deploy.log").read_text(encoding="utf-8", errors="ignore"):
            print("[DEPLOY] Init Done!")
            break
        time.sleep(1.0)
        _check_processes()
    else:
        raise RuntimeError("DEPLOY did not report Init Done within 120s; see /tmp/sonic_deploy.log")

    _run_script("perception/sensor_pub.py", "SENSOR", startup_sleep=1.0)
    if args.camera:
        _run_script(
            "perception/camera_pub.py",
            "CAM",
            [str(selection.abs_path), "--fps", "20", "--depth-fps", "20"],
            startup_sleep=1.0,
        )
    if not args.episode_manifest:
        anchor_args = [str(selection.abs_path)]
        if args.anchor_object_name:
            anchor_args.extend(["--ball-name" if args.demo == "ball" else "--box-name", str(args.anchor_object_name)])
        if args.task_object_id:
            anchor_args.extend(["--ball-id" if args.demo == "ball" else "--box-id", str(args.task_object_id)])
        if args.demo == "ball" and args.anchor_place_site:
            anchor_args.extend(["--place-site", str(args.anchor_place_site)])
        if args.demo == "ball" and args.task_target_id:
            anchor_args.extend(["--place-id", str(args.task_target_id)])
        if args.anchor_approach_standoff is not None:
            anchor_args.extend(["--approach-standoff", str(float(args.anchor_approach_standoff))])
        if args.demo == "ball" and args.dynamic_place_target:
            anchor_args.append("--dynamic-place-target")
        _run_script(ANCHOR_SCRIPTS[args.demo], f"{args.demo.upper()}_ANCHOR", anchor_args, startup_sleep=1.0)
    _run_script("tools/world_model_node.py", "WORLD_MODEL", _world_model_args(args), startup_sleep=1.0)
    if args.episode_manifest:
        _run_script(
            "tools/world_model_episode_anchor.py",
            "WORLD_EPISODE_ANCHOR",
            ["--scene", str(selection.abs_path), "--manifest", str(_repo_path(args.episode_manifest)), "--rate", "2"],
            startup_sleep=1.0,
        )
        # Recovery micro-adjusts use the same bounded /cmd_vel_nav -> SONIC
        # payload path as the project's navigation stacks.
        _run_script("navigation/nav_control_adapter.py", "NAV_CONTROL_ADAPTER", startup_sleep=1.0)
    _run_script("tools/world_model_recovery_coordinator.py", "WORLD_RECOVERY", startup_sleep=1.0)
    _run_script("tools/world_model_recovery_backends.py", "WORLD_RECOVERY_BACKENDS", startup_sleep=1.0)
    if args.qwen_vl_shadow:
        _start_qwen_vl_shadow(args)
    if args.vlm_anchor_bridge:
        _run_script(
            "tools/world_model_vlm_anchor_bridge.py",
            "WORLD_VLM_ANCHOR",
            ["--detections-topic", str(args.vlm_detections_topic)],
            startup_sleep=1.0,
        )
    if args.world_primitive_runner or args.autonomous_world_execution:
        _run_script(
            "tools/world_model_primitive_runner.py",
            "WORLD_PRIMITIVE_RUNNER",
            _primitive_runner_args(args),
            startup_sleep=1.0,
        )
    executor_args = ["--require-effect-evidence"] if args.autonomous_world_execution else []
    _run_script("tools/world_model_executor.py", "WORLD_EXECUTOR", executor_args, startup_sleep=1.0)


def _start_qwen_vl_shadow(args: argparse.Namespace) -> None:
    if args.perception_shadow_backend == "qwen":
        model_path = _repo_path(args.qwen_vl_model_path)
        ready, detail = _qwen_model_ready(model_path)
        if not ready:
            raise RuntimeError(f"Qwen-VL model is not ready: {detail}")
        endpoint = _qwen_vl_endpoint(args)
        server_cmd = [
            "env",
            f"QWEN_VL_LOCAL_MODEL={model_path}",
            f"QWEN_VL_HOST={args.qwen_vl_host}",
            f"QWEN_VL_PORT={int(args.qwen_vl_port)}",
            f"QWEN_VL_MAX_NEW_TOKENS={max(1, int(args.qwen_vl_max_new_tokens))}",
            f"QWEN_VL_DEVICE={args.qwen_vl_device}",
            f"QWEN_VL_GPU_MEMORY_GIB={max(0.0, float(args.qwen_vl_gpu_memory_gib))}",
            "bash",
            str(SCRIPT_DIR / "start_local_qwen_vl.sh"),
        ]
        _run_raw(server_cmd, "QWEN_VL_SERVER", "/tmp/sonic_qwen_vl_server.log", startup_sleep=0.5)
        _wait_for_http_health(
            f"http://{args.qwen_vl_host}:{int(args.qwen_vl_port)}/healthz",
            timeout_s=float(args.qwen_vl_health_timeout),
            process_name="QWEN_VL_SERVER",
        )
        _run_script(
            "tools/world_model_qwen_vl_detector.py",
            "QWEN_VL_DETECTOR",
            [
                "--endpoint",
                endpoint,
                "--model",
                str(model_path),
                "--instruction",
                _qwen_vl_instruction(args),
                "--period",
                str(max(0.0, float(args.qwen_vl_period))),
                "--audit-output",
                str(_qwen_audit_output(args)),
            ],
            startup_sleep=1.0,
        )
    elif args.perception_shadow_backend == "hsv":
        _run_script(
            "tools/world_model_hsv_region_detector.py",
            "HSV_REGION_DETECTOR",
            [
                "--classes-json",
                json.dumps(_hsv_shadow_classes(args), separators=(",", ":")),
                "--period",
                str(max(0.0, float(args.qwen_vl_period))),
            ],
            startup_sleep=1.0,
        )
    else:
        model_path = _repo_path(args.grounding_dino_model_path)
        if not _grounding_dino_model_ready(model_path):
            raise RuntimeError(f"Grounding DINO model is not ready: {model_path}")
        server_cmd = [
            "env",
            f"GROUNDING_DINO_LOCAL_MODEL={model_path}",
            f"GROUNDING_DINO_HOST={args.grounding_dino_host}",
            f"GROUNDING_DINO_PORT={int(args.grounding_dino_port)}",
            "bash",
            str(SCRIPT_DIR / "start_local_grounding_dino.sh"),
        ]
        _run_raw(server_cmd, "GROUNDING_DINO_SERVER", "/tmp/sonic_grounding_dino_server.log", startup_sleep=0.5)
        _wait_for_http_health(
            f"http://{args.grounding_dino_host}:{int(args.grounding_dino_port)}/healthz",
            timeout_s=float(args.qwen_vl_health_timeout),
            process_name="GROUNDING_DINO_SERVER",
        )
        _run_script(
            "tools/world_model_grounding_dino_detector.py",
            "GROUNDING_DINO_DETECTOR",
            [
                "--endpoint",
                f"http://{args.grounding_dino_host}:{int(args.grounding_dino_port)}/v1/detections",
                "--classes-json",
                json.dumps(_grounding_dino_classes(args), separators=(",", ":")),
                "--box-threshold",
                str(float(args.grounding_dino_box_threshold)),
                "--text-threshold",
                str(float(args.grounding_dino_text_threshold)),
                "--period",
                str(max(0.0, float(args.qwen_vl_period))),
            ],
            startup_sleep=1.0,
        )
    _run_script(
        "tools/world_model_rgbd_anchor_backend.py",
        "QWEN_RGBD_ANCHOR",
        [
            "--anchor-topic",
            "/sonic_world/qwen_rgbd_anchor",
            *(["--calibration-file", str(_repo_path(args.qwen_vl_calibration_file))] if args.qwen_vl_calibration_file else []),
            *( ["--depth-cache-size", str(max(24, int(args.qwen_vl_depth_cache_size)))] if args.perception_shadow_backend == "qwen" else [] ),
            *_visual_expected_object_args(args),
            *( [
                "--auto-reobserve-on-missing",
                "--reobserve-max-attempts",
                str(max(0, int(args.visual_reobserve_max_attempts))),
                "--reobserve-cooldown-s",
                str(max(0.0, float(args.visual_reobserve_cooldown_s))),
                *( ["--visual-recovery-escalate-navigation"] if args.visual_recovery_escalate_navigation else [] ),
            ] if args.visual_auto_reobserve else [] ),
            *_perception_pose_filter_args(args),
        ],
        startup_sleep=1.0,
    )
    calibration_output = _qwen_calibration_output(args)
    _run_script(
        "tools/world_model_rgbd_calibration_probe.py",
        "QWEN_RGBD_CALIBRATION",
        [
            "--reference-topic",
            "/sonic_world/object_anchor" if args.episode_manifest else _privileged_anchor_topic(args.demo),
            "--max-samples",
            str(max(0, int(args.qwen_vl_calibration_max_samples))),
            "--output",
            str(calibration_output),
        ],
        startup_sleep=1.0,
    )
    temporal_topic = "/sonic_world/qwen_rgbd_anchor_temporal"
    _run_script(
        "tools/world_model_temporal_anchor_filter.py",
        "QWEN_TEMPORAL_ANCHOR",
        [
            "--input-topic",
            "/sonic_world/qwen_rgbd_anchor",
            "--output-topic",
            temporal_topic,
            "--window-size",
            str(max(1, int(args.qwen_vl_temporal_window))),
            "--min-observations",
            str(max(1, int(args.qwen_vl_temporal_min_observations))),
        ],
        startup_sleep=1.0,
    )
    if args.qwen_vl_gate_report:
        _run_script(
            "tools/world_model_vlm_anchor_gate.py",
            "QWEN_VL_GATE",
            ["--gate-report", str(_repo_path(args.qwen_vl_gate_report))],
            startup_sleep=1.0,
        )
    reference_output, prediction_output = _qwen_shadow_outputs(args)
    _run_script(
        "tools/world_model_shadow_anchor_recorder.py",
        "QWEN_SHADOW_RECORDER",
        [
            "--reference-topic",
            "/sonic_world/object_anchor" if args.episode_manifest else _privileged_anchor_topic(args.demo),
            "--prediction-topic",
            temporal_topic,
            "--reference-output",
            str(reference_output),
            "--prediction-output",
            str(prediction_output),
            "--max-pairs",
            str(max(0, int(args.qwen_vl_shadow_max_pairs))),
        ],
        startup_sleep=1.0,
    )
    print(
        f"[{args.perception_shadow_backend.upper()}_SHADOW] recording privileged and visual/RGB-D anchors to "
        f"{reference_output} and {prediction_output}; planner input remains privileged"
    )


def _qwen_vl_endpoint(args: argparse.Namespace) -> str:
    return f"http://{args.qwen_vl_host}:{int(args.qwen_vl_port)}/v1/chat/completions"


def _qwen_vl_instruction(args: argparse.Namespace) -> str:
    if str(args.qwen_vl_instruction).strip():
        return str(args.qwen_vl_instruction).strip()
    if args.episode_manifest:
        try:
            manifest = json.loads(_repo_path(args.episode_manifest).read_text(encoding="utf-8"))
            stages = manifest.get("stages") if isinstance(manifest.get("stages"), list) else []
            first = stages[0] if stages and isinstance(stages[0], dict) else {}
            request = first.get("request") if isinstance(first.get("request"), dict) else {}
            object_id = str(request.get("object_id") or "")
            category = str(request.get("object_category") or "")
            if not category:
                for item in first.get("objects") or []:
                    if isinstance(item, dict) and str(item.get("object_id") or "") == object_id:
                        category = str(item.get("category") or "object")
                        break
            if object_id:
                return f"Locate only visible {category or 'object'} object_id='{object_id}' category='{category or 'object'}'. Ignore robot and background."
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    default_object_id = "demo_ball_visual" if args.demo == "ball" else "demo_box_visual"
    object_id = str(args.task_object_id or default_object_id)
    category = str(args.task_object_category or args.demo)
    target_id = str(args.task_target_id or ("place_target" if args.demo == "ball" else ""))
    object_description = category
    target_description = "destination region"
    # Generated benchmark spheres/targets use green materials. Keep this visual prior scoped to task-suite ids;
    # standalone ball demos retain their original prompt and real deployments can override the instruction.
    if args.task_object_id and category == "ball":
        object_description = "green ball"
        target_description = "green circular target region"
    target_clause = (
        f" Also locate the visible {target_description} with object_id='{target_id}' and category='place_target'."
        if target_id
        else ""
    )
    return (
        f"Locate only the visible {object_description} with object_id='{object_id}' and category='{category}'."
        f"{target_clause} Ignore the robot, background, and unrelated objects."
    )


def _hsv_shadow_classes(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.demo == "ball":
        return [
            {
                "object_id": str(args.task_object_id or "demo_ball_visual"),
                "category": str(args.task_object_category or "ball"),
                "hsv_lower": [95, 80, 60],
                "hsv_upper": [115, 255, 255],
                "min_area": 8,
                "max_area": 800,
                "support": "table",
                "shape": "sphere",
            },
            {
                "object_id": str(args.task_target_id or "place_target"),
                "category": "place_target",
                "hsv_lower": [40, 45, 45],
                "hsv_upper": [75, 255, 255],
                "min_area": 12,
                "max_area": 4000,
                "support": "table",
                "shape": "target",
            },
        ]
    return [
        {
            "object_id": str(args.task_object_id or "demo_box_visual"),
            "category": str(args.task_object_category or "box"),
            "hsv_lower": [10, 60, 60],
            "hsv_upper": [35, 255, 255],
            "min_area": 20,
            "max_area": 12000,
            "support": "table",
            "shape": "box",
        }
    ]


def _grounding_dino_classes(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.demo == "ball":
        specs = [
            {
                "object_id": str(args.task_object_id or "demo_ball_visual"),
                "category": str(args.task_object_category or "ball"),
                "label": str(args.grounding_dino_object_label or "blue ball"),
                "max_area": 2400,
                "support": "table",
                "shape": "sphere",
            },
            {
                "object_id": str(args.task_target_id or "place_target"),
                "category": "place_target",
                "label": str(args.grounding_dino_target_label or "green target"),
                "max_area": 10000,
                "support": "table",
                "shape": "target",
            },
        ]
        return _expand_grounding_dino_labels(specs)
    return _expand_grounding_dino_labels([
        {
            "object_id": str(args.task_object_id or "demo_box_visual"),
            "category": str(args.task_object_category or "box"),
            "label": str(args.grounding_dino_object_label or "box"),
            "max_area": 12000,
            "support": "table",
            "shape": "box",
        }
    ])


def _expand_grounding_dino_labels(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for spec in specs:
        labels = [item.strip() for item in str(spec.get("label") or "object").split("|") if item.strip()]
        expanded.extend({**spec, "label": label} for label in labels)
    return expanded


def _visual_expected_object_args(args: argparse.Namespace) -> list[str]:
    object_id = str(args.task_object_id or ("demo_ball_visual" if args.demo == "ball" else "demo_box_visual"))
    values = ["--expected-object-id", object_id]
    if args.demo == "ball":
        values.extend(["--expected-object-id", str(args.task_target_id or "place_target")])
    return values


def _perception_pose_filter_args(args: argparse.Namespace) -> list[str]:
    if args.perception_shadow_backend not in {"hsv", "grounding_dino"}:
        return []
    if args.demo == "ball":
        return ["--map-z-min", "0.65", "--map-z-max", "1.05"]
    return ["--map-z-min", "0.55", "--map-z-max", "1.15"]


def _grounding_dino_model_ready(model_path: Path) -> bool:
    return (model_path / "config.json").is_file() and any(
        (model_path / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def _qwen_shadow_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = _repo_path(args.qwen_vl_shadow_dir)
    prefix = str(args.prefix or DEFAULT_PREFIXES[args.demo])
    return (
        output_dir / f"{prefix}_privileged_anchors.jsonl",
        output_dir / f"{prefix}_qwen_rgbd_anchors.jsonl",
    )


def _qwen_calibration_output(args: argparse.Namespace) -> Path:
    output_dir = _repo_path(args.qwen_vl_shadow_dir)
    prefix = str(args.prefix or DEFAULT_PREFIXES[args.demo])
    return output_dir / f"{prefix}_rgbd_calibration.jsonl"


def _qwen_audit_output(args: argparse.Namespace) -> Path:
    if str(args.qwen_vl_audit_output).strip():
        return _repo_path(args.qwen_vl_audit_output)
    output_dir = _repo_path(args.qwen_vl_shadow_dir)
    prefix = str(args.prefix or DEFAULT_PREFIXES[args.demo])
    return output_dir / f"{prefix}_qwen_response_audit.jsonl"


def _privileged_anchor_topic(demo: str) -> str:
    return f"/sonic_demo/{demo}_anchor"


def _qwen_model_ready(model_path: Path) -> tuple[bool, str]:
    config = model_path / "config.json"
    if not config.is_file():
        return False, str(config)
    single_file = model_path / "model.safetensors"
    if single_file.is_file():
        return True, str(model_path)
    index = model_path / "model.safetensors.index.json"
    if not index.is_file():
        return False, str(index)
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        shards = {str(value) for value in weight_map.values()} if isinstance(weight_map, dict) else set()
    except (OSError, json.JSONDecodeError):
        return False, str(index)
    if not shards:
        return False, str(index)
    missing = sorted(shard for shard in shards if not (model_path / shard).is_file())
    if missing:
        return False, f"missing {len(missing)} model shard(s), first: {model_path / missing[0]}"
    return True, str(model_path)


def _wait_for_http_health(url: str, *, timeout_s: float, process_name: str) -> None:
    print(f"[{process_name}] Waiting for {url}...")
    deadline = time.monotonic() + max(1.0, timeout_s)
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:  # nosec B310 - fixed local endpoint from CLI host/port
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                print(f"[{process_name}] Ready")
                return
            last_error = f"unexpected health response: {payload!r}"
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        _check_processes()
        time.sleep(0.5)
    raise RuntimeError(f"{process_name} health check timed out after {timeout_s:.1f}s: {last_error}")


def _demo_cmd(
    args: argparse.Namespace,
    scene_path: Path,
    run_id: str,
    extra_args: list[str],
) -> list[str]:
    anchor_flags = (
        ["--use-ball-anchor", "--require-ball-anchor"]
        if args.demo == "ball"
        else ["--use-box-anchor", "--require-box-anchor"]
    )
    return [
        args.python,
        str(SCRIPTS_DIR / DEMO_SCRIPTS[args.demo]),
        "--rollout-id",
        run_id,
        "--no-hold",
        "--scene",
        str(scene_path),
        *anchor_flags,
        *extra_args,
    ]


def _run_script(script: str, name: str, args: list[str] | None = None, *, startup_sleep: float = 1.0) -> subprocess.Popen:
    script_path = SCRIPTS_DIR / script
    cmd = [
        "bash",
        "-c",
        (
            "source /opt/ros/humble/setup.bash && "
            f"export PYTHONPATH=\"{REPO}:{REPO}/g1_ros2_nav:$PYTHONPATH\" && "
            f"exec /usr/bin/python3 {_shell_quote(script_path)}{_format_args(args)}"
        ),
    ]
    return _run_raw(cmd, name, f"/tmp/sonic_{name.lower()}.log", startup_sleep=startup_sleep)


def _sim_cli_args(args: argparse.Namespace) -> str:
    flags: list[str] = []
    if args.headless:
        flags.append("--no-enable-onscreen")
    return _format_args(flags)


def _world_model_args(args: argparse.Namespace) -> list[str]:
    flags = ["--policy-backend", str(args.world_policy_backend)]
    if args.world_policy_model:
        flags.extend(["--policy-model", str(args.world_policy_model)])
    if args.world_runtime_override_file:
        flags.extend(["--runtime-override-file", str(args.world_runtime_override_file)])
    return flags


def _primitive_runner_args(args: argparse.Namespace) -> list[str]:
    backend = str(args.autonomous_primitive_backend if args.autonomous_world_execution else args.world_primitive_backend)
    out = [
        "--backend",
        backend,
        "--demo-kind",
        "auto" if args.episode_manifest else str(args.demo),
        "--scene",
        str(args.scene or ""),
    ]
    if args.autonomous_world_execution:
        out.extend(["--effect-observer", "mujoco_qpos"])
    if args.episode_manifest:
        out.append("--prefer-object-anchor")
    if args.world_teacher_ball_attach:
        if args.demo != "ball":
            raise ValueError("--world-teacher-ball-attach is supported only for ball rollouts")
        # ``--ball-attach`` starts with a dash, so it must be attached to the
        # repeatable wrapper option rather than parsed as a runner flag.
        out.extend([
            "--teacher-assisted",
            "--demo-arg=--ball-attach",
            "--demo-arg=--teacher-pregrasp-attach",
        ])
    if args.world_teacher_lift_attach:
        if args.demo != "ball":
            raise ValueError("--world-teacher-lift-attach is supported only for ball rollouts")
        out.extend([
            "--teacher-assisted",
            "--teacher-assist-skill=manip.lift_object",
            "--demo-arg=--ball-attach",
            "--demo-arg=--teacher-lift-attach",
        ])
    return out


def _episode_cmd(args: argparse.Namespace, manifest: Path) -> list[str]:
    output = _episode_output(args, manifest)
    cmd = [
        args.python,
        str(SCRIPT_DIR / "world_model_autonomous_episode.py"),
        "--manifest", str(manifest),
        "--output-jsonl", str(output),
        "--timeout-per-stage", str(float(args.episode_timeout_per_stage)),
        "--stage-start", str(int(args.episode_stage_start)),
    ]
    if int(args.episode_stage_stop) > 0:
        cmd.extend(["--stage-stop", str(int(args.episode_stage_stop))])
    if args.episode_continue_on_failure:
        cmd.append("--continue-on-failure")
    return cmd


def _episode_output(args: argparse.Namespace, manifest: Path) -> Path:
    return _repo_path(args.episode_output_jsonl) if args.episode_output_jsonl else _repo_path(args.report_path) / f"episode_{_safe_name(manifest.stem)}.jsonl"


def _has_episode_terminal(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "episode_terminal":
            return True
    return False


def _autonomous_cmd(args: argparse.Namespace, run_id: str) -> list[str]:
    task_id = str(args.task_id or run_id)
    verb = str(args.task_verb or _default_task_verb(args.demo))
    cmd = [
        args.python,
        str(SCRIPT_DIR / "world_model_autonomous_task.py"),
        "--run-id",
        run_id,
        "--task-id",
        task_id,
        "--task",
        verb,
        "--timeout",
        str(float(args.autonomous_timeout)),
        "--output-jsonl",
        str(_repo_path(args.report_path) / f"{args.demo}_{run_id}.jsonl"),
    ]
    if args.task_object_id:
        cmd.extend(["--object-id", str(args.task_object_id)])
    if args.task_object_category:
        cmd.extend(["--object-category", str(args.task_object_category)])
    if args.task_target_id:
        cmd.extend(["--target-id", str(args.task_target_id)])
    return cmd


def _default_task_verb(demo: str) -> str:
    return "pick_place" if demo == "ball" else "pick"


def _run_script_cmd(cmd: list[str], name: str, *, startup_sleep: float = 1.0) -> subprocess.Popen:
    wrapped = [
        "bash",
        "-c",
        (
            "source /opt/ros/humble/setup.bash && "
            f"export PYTHONPATH=\"{REPO}:{REPO}/g1_ros2_nav:$PYTHONPATH\" && "
            "exec " + " ".join(_shell_quote(part) for part in cmd)
        ),
    ]
    return _run_raw(wrapped, name, f"/tmp/sonic_{name.lower()}.log", startup_sleep=startup_sleep)


def _run_raw(cmd: list[str], name: str, log_path: str, *, startup_sleep: float) -> subprocess.Popen:
    print(f"[{name}] Starting...")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    procs.append(proc)
    proc_names[proc.pid] = name
    if startup_sleep > 0.0:
        time.sleep(startup_sleep)
    if proc.poll() is not None:
        print(f"[{name}] FAILED, see {log_path}")
        _forget_process(proc)
        raise RuntimeError(f"{name} exited with code {proc.returncode}")
    print(f"[{name}] Running")
    return proc


def _stream_demo_progress(demo: str, proc: subprocess.Popen) -> int:
    log_path = f"/tmp/sonic_{demo.upper()}_DEMO".lower() + ".log"
    node_tag = DEMO_NODE_TAGS[demo]
    offset = 0
    while proc.poll() is None:
        offset = _print_new_progress(log_path, offset, node_tag=node_tag, prefix=f"{demo.upper()}_DEMO")
        _check_processes(ignore={proc.pid})
        time.sleep(0.5)
    offset = _print_new_progress(log_path, offset, node_tag=node_tag, prefix=f"{demo.upper()}_DEMO")
    _forget_process(proc)
    return int(proc.returncode or 0)


def _print_new_progress(log_path: str, offset: int, *, node_tag: str, prefix: str) -> int:
    if not os.path.exists(log_path):
        return offset
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        offset = handle.tell()
    for raw in lines:
        line = raw.strip()
        if not any(token in line for token in PROGRESS_TOKENS):
            continue
        msg = line.split(node_tag, 1)[-1].strip()
        print(f"[{prefix}] {msg}", flush=True)
    return offset


def _check_processes(ignore: set[int] | None = None) -> None:
    ignore = ignore or set()
    for proc in list(procs):
        if proc.pid in ignore:
            continue
        code = proc.poll()
        if code is not None:
            name = proc_names.get(proc.pid, f"pid {proc.pid}")
            if name == "QWEN_SHADOW_RECORDER" and code == 0:
                print("[QWEN_SHADOW_RECORDER] Pair limit reached")
                _forget_process(proc)
                continue
            raise RuntimeError(f"{name} exited unexpectedly with code {code}")


def _forget_process(proc: subprocess.Popen) -> None:
    try:
        procs.remove(proc)
    except ValueError:
        pass
    proc_names.pop(proc.pid, None)


def _shutdown_children() -> None:
    print("\n[STOP] Shutting down reusable Sonic stack...")
    for proc in reversed(procs):
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    print("[STOP] Done")


def _install_signal_handlers() -> None:
    def _cleanup(*_args: object) -> None:
        _shutdown_children()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)


def _deploy_max_close_ratio(demo: str) -> str:
    return "0.85" if demo == "ball" else "0.2"


def _read_rollout_outcome(demo: str, run_id: str, report_path: str) -> dict[str, Any]:
    path = _repo_path(report_path) / f"{demo}_{run_id}.jsonl"
    outcome: dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "task_status": None,
        "task_reason": None,
        "task_error": None,
        "anchor_reason": None,
        "anchor_base": None,
    }
    if not path.exists():
        return outcome

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return outcome

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "task_end":
            outcome["task_status"] = event.get("status")
            outcome["task_reason"] = event.get("reason")
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            outcome["task_error"] = metadata.get("error")
        elif event.get("event") == "anchor_update":
            outcome["anchor_reason"] = event.get("reason")
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            base = _parse_anchor_base_from_summary(str(metadata.get("summary") or ""), demo)
            if base is not None:
                outcome["anchor_base"] = base
    return outcome


def _read_autonomous_outcome(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    return _read_rollout_outcome(str(args.demo), run_id, str(args.report_path))


def _parse_anchor_base_from_summary(summary: str, demo: str) -> tuple[float, float, float] | None:
    for match in BASE_SUMMARY_RE.finditer(summary):
        if match.group("kind") != demo:
            continue
        parts = [part.strip() for part in match.group("xyz").split(",")]
        if len(parts) != 3:
            continue
        try:
            xyz = tuple(float(part) for part in parts)
        except ValueError:
            continue
        if all(value == value for value in xyz):
            return xyz  # type: ignore[return-value]
    return None


def _terminal_anchor_issue(args: argparse.Namespace, outcome: dict[str, Any]) -> str | None:
    reason = outcome.get("anchor_reason")
    if reason in ANCHOR_LOSS_REASONS:
        return str(reason)

    task_reason = outcome.get("task_reason")
    if task_reason in ANCHOR_LOSS_REASONS:
        return str(task_reason)

    task_error = str(outcome.get("task_error") or "")
    if "anchor is required" in task_error and "no anchor" in task_error:
        return "missing_or_implausible_anchor"

    base = outcome.get("anchor_base")
    if base is not None:
        z = float(base[2])
        if z < float(args.terminal_anchor_min_z):
            return f"{args.demo}_out_of_workspace_z={z:.2f}"
    return None


def _request_sim_reset(args: argparse.Namespace, *, reason: str) -> None:
    path = _repo_path(args.sim_reset_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": True,
        "action": "reset_scene",
        "reason": reason,
        "stamp": time.time(),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    print(f"[BATCH] requested MuJoCo scene reset via {path}")
    time.sleep(max(0.0, float(args.reset_settle)))


def _request_episode_curriculum_reset(args: argparse.Namespace, manifest_path: Path, stage_index: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), list) else []
    stage = next((item for item in stages if int(item.get("stage_index") or 0) == int(stage_index)), None)
    if not isinstance(stage, dict):
        raise ValueError(f"curriculum stage {stage_index} is not present in {manifest_path}")
    registry = stage.get("object_registry") if isinstance(stage.get("object_registry"), dict) else {}
    records = registry.get("records") if isinstance(registry.get("records"), list) else []
    poses = {
        str(item.get("object_id") or item.get("id") or ""): item.get("pose_map")
        for item in stage.get("objects", [])
        if isinstance(item, dict)
    }
    freejoints = []
    for record in records:
        if not isinstance(record, dict):
            continue
        object_id = str(record.get("object_id") or "")
        pose = poses.get(object_id)
        position = pose.get("position") if isinstance(pose, dict) else None
        joint = str(record.get("joint_name") or "")
        if joint and isinstance(position, list) and len(position) >= 3:
            freejoints.append({"joint_name": joint, "position": [float(value) for value in position[:3]]})
    payload = {
        "enabled": True,
        "action": "reset_all",
        "freejoints": freejoints,
        "reason": f"episode_curriculum_stage_{stage_index}",
        "stamp": time.time(),
    }
    path = _repo_path(args.sim_reset_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    print(f"[EPISODE] requested curriculum reset stage={stage_index} freejoints={len(freejoints)}")
    time.sleep(max(0.0, float(args.reset_settle)))


def _format_args(args: list[str] | None = None) -> str:
    if not args:
        return ""
    return "".join(f" {_shell_quote(value)}" for value in args)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def _remove_if_exists(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _print_run_header(
    demo: str,
    index: int,
    total: int,
    run_id: str,
    cmd: list[str],
    *,
    mode: str,
) -> None:
    print()
    print("=" * 72)
    print(f"[BATCH] {demo} rollout {index}/{total}: {run_id} ({mode})")
    print("[BATCH] " + " ".join(_shell_quote(part) for part in cmd))
    print("=" * 72)


def _clean_demo_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        values = values[1:]
    return [value for value in values if value != "--exit-after-demo"]


def _next_index(prefix: str, report_path: Path) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    best = 0
    if report_path.is_dir():
        for path in sorted(report_path.glob("*.jsonl")):
            run_id = _run_id_from_jsonl(path)
            match = pattern.match(run_id or "")
            if match:
                best = max(best, int(match.group(1)))
    return best + 1


def _run_id_from_jsonl(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("run_id"):
                    return str(payload["run_id"])
                return None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _repo_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else REPO / raw


def _child_env() -> dict[str, str]:
    # Keep ad-hoc task/demo subprocesses in the same DDS domain and localhost
    # mode as the long-lived stack started through _run_raw.
    env = ENV.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _shell_quote(value: object) -> str:
    text = str(value)
    if not text or any(ch.isspace() or ch in "'\"$`\\!" for ch in text):
        return "'" + text.replace("'", "'\"'\"'") + "'"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
