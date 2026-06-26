#!/usr/bin/env python3
import os, sys, time, signal, subprocess, threading, math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
import mujoco

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

class SensorBridge(Node):
    def __init__(self):
        super().__init__("sensor_bridge")
        from g1_ros2_nav.lidar_sim import LidarSim
        xml = os.path.join(REPO,"gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
        self._model = mujoco.MjModel.from_xml_path(xml)
        self._data = mujoco.MjData(self._model)
        self._lidar = LidarSim(self._model,self._data)
        self._odom = self.create_publisher(Odometry,"/odom",10)
        self._scan = self.create_publisher(LaserScan,"/scan",10)
        self._tf = TransformBroadcaster(self)
        self._timer = self.create_timer(0.05,self._publish)
        self.get_logger().info("Sensor bridge started — /odom /scan /tf")

    def _publish(self):
        try:
            qpos = np.load("/tmp/sonic_qpos.npy")
            self._data.qpos[:len(qpos)] = qpos
        except:
            return
        mujoco.mj_forward(self._model,self._data)
        now = self.get_clock().now().to_msg()
        h = Header(stamp=now,frame_id="odom")
        pos = self._data.qpos[0:3].copy()
        quat = self._data.qpos[3:7].copy()
        t = TransformStamped()
        t.header=h; t.child_frame_id="base_link"
        t.transform.translation.x=float(pos[0]); t.transform.translation.y=float(pos[1]); t.transform.translation.z=float(pos[2])
        t.transform.rotation.w=float(quat[0]); t.transform.rotation.x=float(quat[1]); t.transform.rotation.y=float(quat[2]); t.transform.rotation.z=float(quat[3])
        self._tf.sendTransform(t)
        yaw = math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
        odom = Odometry()
        odom.header=h; odom.child_frame_id="base_link"
        odom.pose.pose.position.x=float(pos[0]); odom.pose.pose.position.y=float(pos[1])
        cy=math.cos(yaw/2); sy=math.sin(yaw/2)
        odom.pose.pose.orientation=Quaternion(w=cy,z=sy)
        self._odom.publish(odom)
        self._lidar.step()
        d = self._lidar
        scan = LaserScan()
        scan.header=Header(stamp=now,frame_id="lidar_link")
        scan.angle_min=0.0; scan.angle_max=2*math.pi-d.angles[1]
        scan.angle_increment=float(d.angles[1]-d.angles[0])
        scan.range_min=float(d.min_range); scan.range_max=float(d.max_range)
        scan.ranges=[float(r) for r in d.ranges]
        self._scan.publish(scan)

def run_bridge():
    rclpy.init(args=sys.argv)
    bridge = SensorBridge()
    try: rclpy.spin(bridge)
    except: pass
    bridge.destroy_node()

def main():
    signal.signal(signal.SIGINT,lambda *_: (cleanup(),sys.exit(0)))
    print("="*50+"\n  Sonic-Nav  |  DOMAIN=42\n"+"="*50)
    start_sim()
    start_deploy()
    log("WAIT","Deploy init...")
    if not wait_for("Init Done"): log("ERROR","Failed");cleanup();sys.exit(1)
    log("DEPLOY","Init Done!")
    robot_start()
    threading.Thread(target=run_bridge,daemon=True).start()
    time.sleep(4)
    log("BRIDGE","/odom /scan /tf active")
    print("\n"+"="*50+"\n  Ready! Robot standing.\n  ros2 launch g1_ros2_nav bringup.launch.py\n  Ctrl+C to stop\n"+"="*50)
    try: signal.pause()
    except: pass

if __name__=="__main__":
    main()
