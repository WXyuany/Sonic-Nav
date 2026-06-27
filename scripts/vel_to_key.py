#!/usr/bin/env -S /usr/bin/python3
import os, sys, time, subprocess, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
os.environ.setdefault("RMW_IMPLEMENTATION","rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY","1")
os.environ.setdefault("ROS_DOMAIN_ID","42")
TMUX="sonic-nav"
PANE="0.1"
class VelToKey(Node):
    def __init__(self):
        super().__init__("vel_to_key")
        self.create_subscription(Twist,"/cmd_vel",self._cb,10)
        self.get_logger().info("Ready")
    def _cb(self,msg):
        vx=msg.linear.x;vy=msg.linear.y;vw=msg.angular.z
        k=" ";ax=abs(vx);ay=abs(vy);aw=abs(vw)
        if ax>0.01 and ax>=ay and ax>=aw*2: k="w" if vx>0 else "s"
        elif ay>0.01 and ay>ax and ay>=aw*2: k="a" if vy>0 else "d"
        elif aw>0.01: k="q" if vw>0 else "e"
        subprocess.run(["tmux","send-keys","-t",f"{TMUX}:{PANE}",k],capture_output=True,timeout=1)
def main():
    rclpy.init();n=VelToKey()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node();rclpy.shutdown()
if __name__=="__main__": main()
