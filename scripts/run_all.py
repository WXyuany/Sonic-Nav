#!/usr/bin/env -S /usr/bin/python3
import os,sys,pty,math,time,select,subprocess
import rclpy,msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

REPO=os.path.expanduser("~/GR00T-WholeBodyControl")
master,slave=pty.openpty()
env=os.environ.copy()
env.update({"DISPLAY":":1","ROS_DOMAIN_ID":"42","TensorRT_ROOT":os.path.expanduser("~/TensorRT")})
cmd=(f"cd {REPO}/gear_sonic_deploy && source scripts/setup_env.sh >/dev/null 2>&1 && "
     f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx "
     f"reference/example/ --obs-config policy/release/observation_config.yaml "
     f"--encoder-file policy/release/model_encoder.onnx "
     f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
     f"--input-type keyboard --output-type all --zmq-host localhost --disable-crc-check")
proc=subprocess.Popen(["bash","-c",cmd],stdin=slave,stdout=slave,stderr=slave,env=env,close_fds=True)
os.close(slave)

def send(k):
    os.write(master,k.encode())

def read_deploy(timeout=0):
    r,_,_ = select.select([master],[],[],timeout)
    if r:
        try:
            data=os.read(master,4096)
            sys.stdout.buffer.write(data); sys.stdout.flush()
        except: pass

# Wait for Init Done
out=b""
while b"Init Done" not in out:
    try: out+=os.read(master,4096)
    except: time.sleep(0.5)
print("Init Done!")

# Start control + planner
send("]")
time.sleep(2)
send("\n")
time.sleep(2)
print("Planner enabled, ready for goals")

# ROS2 goal follower
class GF(Node):
    def __init__(self):
        super().__init__('gf')
        self.goal=None
        self.create_subscription(PoseStamped,'/goal_pose',self._g,10)
        self.get_logger().info('Goal follower ready. Set 2D Goal Pose in RViz.')
    def _g(self,m):
        self.goal=(m.pose.position.x,m.pose.position.y)
        self.get_logger().info(f'Goal: ({self.goal[0]:.2f},{self.goal[1]:.2f})')
    def tick(self):
        if self.goal is None: return
        gx,gy=self.goal
        dist=math.hypot(gx,gy)
        if dist<0.3:
            self.goal=None
            send(" ")
            self.get_logger().info('Goal reached!')
            return
        angle=math.atan2(gy,gx)
        if abs(angle)>0.15:
            send("q" if angle>0 else "e")
        else:
            send("w")

rclpy.init()
gf=GF()
last_tick=time.time()

try:
    while True:
        read_deploy(0.01)
        if time.time()-last_tick>0.1:
            gf.tick()
            rclpy.spin_once(gf,timeout_sec=0.01)
            last_tick=time.time()
except KeyboardInterrupt:
    pass
gf.destroy_node()
rclpy.shutdown()
proc.terminate()
