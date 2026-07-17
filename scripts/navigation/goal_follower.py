#!/usr/bin/env -S /usr/bin/python3
import os,math,time,rclpy,msgpack,numpy as np
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ByteMultiArray
from tf2_ros import Buffer, TransformException, TransformListener
from rclpy.time import Time

NEUTRAL_WRIST_POSE = [
    0.0903, 0.1615, -0.2411, 0.7295, 0.3145, 0.5533, -0.2506,
    0.1280, -0.1522, -0.2461, 0.7320, -0.2639, 0.5395, 0.3217,
]
NEUTRAL_HAND_JOINTS = [0.0] * 7

class GoalFollower(Node):
    def __init__(self):
        super().__init__('gf')
        self.pub=self.create_publisher(ByteMultiArray,'ControlPolicy/upper_body_pose',10)
        self.tf_buffer=Buffer();self.tf_listener=TransformListener(self.tf_buffer,self)
        self.create_subscription(PoseStamped,'/goal_pose',self.on_goal,10)
        self.create_subscription(Odometry,'/odom',self.on_odom,10)
        self.goal=None
        self.rx=0.0;self.ry=0.0;self.ryaw=0.0
        self.timer=self.create_timer(0.1,self.tick)
        self.get_logger().info('Ready. Set 2D Goal in RViz.')

    def on_goal(self,m):
        self.goal=self._goal_to_odom(m)
        self.get_logger().info(f'Goal: ({self.goal[0]:.1f},{self.goal[1]:.1f})')

    def _goal_to_odom(self,m):
        frame=m.header.frame_id or 'odom'
        if frame=='odom': return (m.pose.position.x,m.pose.position.y)
        try:
            tf=self.tf_buffer.lookup_transform('odom',frame,Time())
        except TransformException as e:
            self.get_logger().warn(f'No TF {frame}->odom for goal, using raw goal: {e}')
            return (m.pose.position.x,m.pose.position.y)
        q=tf.transform.rotation;t=tf.transform.translation
        r=np.array([[1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y-q.z*q.w),2*(q.x*q.z+q.y*q.w)],
                    [2*(q.x*q.y+q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.y*q.z-q.x*q.w)],
                    [2*(q.x*q.z-q.y*q.w),2*(q.y*q.z+q.x*q.w),1-2*(q.x*q.x+q.y*q.y)]],dtype=np.float32)
        p=np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z],dtype=np.float32)
        out=p@r.T+np.array([t.x,t.y,t.z],dtype=np.float32)
        return (float(out[0]),float(out[1]))

    def on_odom(self,m):
        self.rx=m.pose.pose.position.x
        self.ry=m.pose.pose.position.y
        q=m.pose.pose.orientation
        self.ryaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

    def tick(self):
        pl={'toggle_policy_action':False,'locomotion_mode':0,'base_height_command':0.78,'navigate_cmd':[0,0,0],
            'wrist_pose':NEUTRAL_WRIST_POSE,'left_hand_joint':NEUTRAL_HAND_JOINTS,
            'right_hand_joint':NEUTRAL_HAND_JOINTS}
        if self.goal is None: pass
        else:
            dx=self.goal[0]-self.rx;dy=self.goal[1]-self.ry
            dist=math.hypot(dx,dy)
            if dist<0.5: self.goal=None;self.get_logger().info('Reached!')
            else:
                target=math.atan2(dy,dx)
                err=target-self.ryaw
                err=math.atan2(math.sin(err),math.cos(err))
                turn = max(-0.5, min(0.5, err * 1.0))
                fwd = max(0.0, min(0.8, dist*0.5 - abs(err)*0.5))
                pl['navigate_cmd'] = [fwd, 0, turn]
        m=ByteMultiArray();m.data=[bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)]
        self.pub.publish(m)

def main():
    rclpy.init();n=GoalFollower()
    try: rclpy.spin(n)
    except: pass
    n.destroy_node();rclpy.shutdown()

if __name__=='__main__': main()
