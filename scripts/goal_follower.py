#!/usr/bin/env -S /usr/bin/python3
import os,math,time,rclpy,msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ByteMultiArray

class GoalFollower(Node):
    def __init__(self):
        super().__init__('gf')
        self.pub=self.create_publisher(ByteMultiArray,'ControlPolicy/upper_body_pose',10)
        self.create_subscription(PoseStamped,'/goal_pose',self.on_goal,10)
        self.create_subscription(Odometry,'/odom',self.on_odom,10)
        self.goal=None
        self.rx=0.0;self.ry=0.0;self.ryaw=0.0
        self.started=False
        self.timer=self.create_timer(0.1,self.tick)
        self.get_logger().info('Ready. Set 2D Goal in RViz.')

    def on_goal(self,m):
        self.goal=(m.pose.position.x,m.pose.position.y)
        self.get_logger().info(f'Goal: ({self.goal[0]:.1f},{self.goal[1]:.1f})')

    def on_odom(self,m):
        self.rx=m.pose.pose.position.x
        self.ry=m.pose.pose.position.y
        q=m.pose.pose.orientation
        self.ryaw=math.atan2(2*(q.w*q.z),1-2*q.z*q.z)

    def tick(self):
        pl={'toggle_policy_action':not self.started,'locomotion_mode':0,'base_height_command':0.78,'navigate_cmd':[0,0,0]}
        self.started=True
        if self.goal is None: pass
        else:
            dx=self.goal[0]-self.rx;dy=self.goal[1]-self.ry
            dist=math.hypot(dx,dy)
            if dist<0.5: self.goal=None;self.get_logger().info('Reached!')
            else:
                target=math.atan2(dy,dx)
                err=target-self.ryaw
                err=math.atan2(math.sin(err),math.cos(err))
                if abs(err)>0.3: pl['navigate_cmd']=[0,0,0.5 if err>0 else -0.5]
                else: pl['navigate_cmd']=[min(0.5,dist*0.3),0,0]
        m=ByteMultiArray();m.data=[bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)]
        self.pub.publish(m)

def main():
    rclpy.init();n=GoalFollower()
    try: rclpy.spin(n)
    except: pass
    n.destroy_node();rclpy.shutdown()

if __name__=='__main__': main()
