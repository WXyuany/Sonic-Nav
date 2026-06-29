#!/usr/bin/env -S /usr/bin/python3
import os, sys, time, math, termios, tty, select
import rclpy, msgpack, numpy as np
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

LEFT  = np.array([0.15, 0.20, 1.05])
RIGHT = np.array([0.15, -0.20, 1.05])
HEAD  = np.array([0.04, 0.0, 1.35])
SPEED = 0.05
ARM   = 'left'
PAUSED = True

class PICOSim(Node):
    def __init__(self):
        super().__init__('pico_sim')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.l = LEFT.copy(); self.r = RIGHT.copy(); self.h = HEAD.copy()
        self._print_help()

    def _print_help(self):
        print("\n  PICO Sim  Space:start  1/2/3:hand  WASD:move  Q/E:up/down  Esc:quit")
        print("  Status: " + ("ACTIVE" if not PAUSED else "PAUSED (press Space)"))

    def send(self):
        global PAUSED
        if PAUSED: return
        pl = {"navigate_cmd":[0,0,0],"locomotion_mode":0,"base_height_command":0.78,
              "toggle_policy_action":False,
              "wrist_pose":list(self.l)+[1,0,0,0]+list(self.r)+[1,0,0,0]}
        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)

    def on_key(self, k):
        global ARM
        if k == '\x1b': return False
        if k == ' ':
            global PAUSED; PAUSED = not PAUSED
            print("  Status:", "ACTIVE" if not PAUSED else "PAUSED")
            return True
        if PAUSED: return True
        if k in '123':
            ARM = {'1':'left','2':'right','3':'head'}[k]
            print("  Active:", ARM)
            return True
        arr = {'left':self.l, 'right':self.r, 'head':self.h}[ARM]
        if   k == 'w': arr[0] += SPEED
        elif k == 's': arr[0] -= SPEED
        elif k == 'a': arr[1] += SPEED
        elif k == 'd': arr[1] -= SPEED
        elif k == 'q': arr[2] += SPEED
        elif k == 'e': arr[2] -= SPEED
        return True

def main():
    try: old = termios.tcgetattr(sys.stdin); tty.setcbreak(sys.stdin)
    except termios.error: print("Must run in terminal"); sys.exit(1)
    rclpy.init()
    pico = PICOSim()
    running, last = True, time.time()
    import threading
    threading.Thread(target=lambda: [rclpy.spin_once(pico, timeout_sec=0.02) for _ in iter(int,1) if running and rclpy.ok()], daemon=True).start()
    try:
        while running:
            if select.select([sys.stdin],[],[],0.05)[0]: running = pico.on_key(sys.stdin.read(1))
            if time.time()-last > 0.1: pico.send(); last = time.time()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        pico.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
