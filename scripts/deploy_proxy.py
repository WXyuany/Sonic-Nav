#!/usr/bin/env -S /usr/bin/python3
import os, sys, pty, time, select, subprocess, fcntl, rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
master, slave = pty.openpty()
fl = fcntl.fcntl(master, fcntl.F_GETFL)
fcntl.fcntl(master, fcntl.F_SETFL, fl | os.O_NONBLOCK)

env = os.environ.copy()
env.update({"DISPLAY":":1", "TensorRT_ROOT": os.path.expanduser("~/TensorRT")})
cmd = (f"cd {REPO}/gear_sonic_deploy && source scripts/setup_env.sh >/dev/null 2>&1 && "
       f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx "
       f"reference/example/ --obs-config policy/release/observation_config.yaml "
       f"--encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx "
       f"--input-type keyboard --output-type all --zmq-host localhost --disable-crc-check")
proc = subprocess.Popen(["bash","-c",cmd], stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
os.close(slave)

print("[PROXY] Waiting for deploy Init Done...")
out = b""
while b"Init Done" not in out:
    if select.select([master],[],[],0.5)[0]:
        try: out += os.read(master, 4096)
        except: pass
print("[PROXY] Init Done!")

# Start control
os.write(master, b"]")
time.sleep(2)
os.write(master, b"\n")
time.sleep(6)
open("/tmp/proxy_ready","w").close()
print("[PROXY] Control + Planner ON, listening for ROS2 commands...")

class Bridge(Node):
    def __init__(self):
        super().__init__('proxy_bridge')
        self.create_subscription(ByteMultiArray, 'ControlPolicy/upper_body_pose', self.on_cmd, 10)
        self.prev_vx = self.prev_vy = self.prev_vw = 0.0
    def on_cmd(self, m):
        pl = msgpack.unpackb(bytes(m.data))
        vx = pl.get('navigate_cmd',[0,0,0])[0]
        vy = pl.get('navigate_cmd',[0,0,0])[1]
        vw = pl.get('navigate_cmd',[0,0,0])[2]
        if abs(vx) > 0.01 and vx > 0: os.write(master, b"w")
        elif abs(vx) > 0.01: os.write(master, b"s")
        elif abs(vy) > 0.01 and vy > 0: os.write(master, b"a")
        elif abs(vy) > 0.01: os.write(master, b"d")
        elif abs(vw) > 0.01 and vw > 0: os.write(master, b"q")
        elif abs(vw) > 0.01: os.write(master, b"e")
        else: os.write(master, b" ")

rclpy.init()
bridge = Bridge()
try:
    while True:
        rclpy.spin_once(bridge, timeout_sec=0.05)
        select.select([master],[],[],0.01)
except KeyboardInterrupt:
    pass
proc.terminate()
