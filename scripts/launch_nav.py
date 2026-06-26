#!/usr/bin/env python3
import os, sys, time, signal, subprocess, threading

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
os.chdir(REPO)
ENV = os.environ.copy()
ENV.update({"ROS_DOMAIN_ID":"42","RMW_IMPLEMENTATION":"rmw_fastrtps_cpp","ROS_LOCALHOST_ONLY":"1",
            "TensorRT_ROOT":os.path.expanduser("~/TensorRT")})
DEPLOY_LOG = "/tmp/sonic_deploy.log"
processes = []

def log(tag,msg):
    print(f"[{tag:>6}] {msg}",flush=True)

def start_sim():
    log("SIM","MuJoCo in tmux...")
    subprocess.run(["tmux","kill-session","-t","sonic-sim"],capture_output=True)
    subprocess.run(["tmux","new-session","-d","-s","sonic-sim",
        f"export DISPLAY=:1 PYTHONPATH='{REPO}:{REPO}/g1_ros2_nav' && source {REPO}/.venv_sim/bin/activate && python {REPO}/gear_sonic/scripts/run_sim_loop.py"],check=True)
    time.sleep(10)
    log("SIM","Running")

def start_deploy():
    log("DEPLOY","Starting...")
    open(DEPLOY_LOG,"w").close()
    cmd = (f"source {REPO}/gear_sonic_deploy/scripts/setup_env.sh >/dev/null 2>&1 && cd {REPO}/gear_sonic_deploy && "
           f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
           f"--obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx "
           f"--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type ros2 --output-type all "
           f"--zmq-host localhost --disable-crc-check")
    proc = subprocess.Popen(["bash","-c",cmd],env=ENV,stdout=open(DEPLOY_LOG,"w"),stderr=subprocess.STDOUT)
    processes.append(("deploy",proc))

def robot_start():
    log("CTRL","Sending start...")
    r = subprocess.run(["bash","-c",
        "source /opt/ros/humble/setup.bash && /usr/bin/python3 -c '"
        "import os,rclpy,msgpack,time;os.environ.update({\"RMW_IMPLEMENTATION\":\"rmw_fastrtps_cpp\",\"ROS_LOCALHOST_ONLY\":\"1\",\"ROS_DOMAIN_ID\":\"42\"});"
        "from rclpy.node import Node;from std_msgs.msg import ByteMultiArray;rclpy.init();n=Node(\"s\");"
        "p=n.create_publisher(ByteMultiArray,\"ControlPolicy/upper_body_pose\",10);time.sleep(3);"
        "pl={\"navigate_cmd\":[0,0,0],\"locomotion_mode\":0,\"base_height_command\":0.78,\"toggle_policy_action\":True};"
        "m=ByteMultiArray();m.data=[bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)];p.publish(m);time.sleep(2);"
        "n.destroy_node();rclpy.shutdown();print(\"OK\")'"],
        env=ENV,capture_output=True,text=True,timeout=30)
    log("CTRL","Robot standing" if "OK" in r.stdout else f"Failed: {r.stderr[:100]}")

def start_sensor():
    log("SENSOR","Starting /odom /scan /tf bridge...")
    proc = subprocess.Popen(["bash","-c",
        f"source /opt/ros/humble/setup.bash && source {REPO}/.venv_sim/bin/activate && exec python {REPO}/g1_ros2_nav/scripts/sensor_bridge.py"],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    processes.append(("sensor",proc))
    time.sleep(4)
    if proc.poll() is not None:
        err = proc.stderr.read().decode()
        log("SENSOR",f"CRASHED: {err[-200:]}")
        return
    log("SENSOR","Running")

def start_cmdvel():
    log("CMDVEL","Bridge...")
    proc = subprocess.Popen(["bash","-c",
        "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && exec ros2 run g1_ros2_nav cmd_vel_bridge"],
        env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    processes.append(("cmdvel",proc))
    log("CMDVEL","Running")

def wait_for(p,timeout=120):
    t0=time.time()
    while time.time()-t0<timeout:
        if os.path.exists(DEPLOY_LOG):
            with open(DEPLOY_LOG) as f:
                if p in f.read(): return True
        time.sleep(0.5)
    return False

def cleanup():
    log("STOP","Shutting down...")
    subprocess.run(["tmux","kill-session","-t","sonic-sim"],capture_output=True)
    for _,p in reversed(processes):
        if p is None: continue
        try: p.terminate();p.wait(timeout=5)
        except: p.kill()
    log("STOP","Done")

def main():
    signal.signal(signal.SIGINT,lambda *_: (cleanup(),sys.exit(0)))
    print("="*50+"\n  Sonic-Nav  |  DOMAIN=42\n"+"="*50)
    start_sim()
    start_deploy()
    log("WAIT","Deploy init...")
    if not wait_for("Init Done"): log("ERROR","Failed");cleanup();sys.exit(1)
    log("DEPLOY","Init Done!")
    robot_start()
    start_sensor()
    start_cmdvel()
    print("\n"+"="*50+"\n  Ready! Robot standing. /odom /scan /tf active.\n  ros2 launch g1_ros2_nav bringup.launch.py\n  Ctrl+C to stop\n"+"="*50)
    try: signal.pause()
    except: pass
    while True: time.sleep(1)

if __name__=="__main__":
    main()
