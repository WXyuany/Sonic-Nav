#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy
from rclpy.executors import ExternalShutdownException
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'g1_ros2_nav'))
from g1_ros2_nav.tmp_io import load_npy_if_ready
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
rclpy.init()
n=Node('sensors')
odom_pub=n.create_publisher(Odometry,'/odom',10)
tf=TransformBroadcaster(n)
publish_map_odom = os.environ.get('SONIC_PUBLISH_MAP_ODOM', '1') != '0'
def pub():
    q=load_npy_if_ready('/tmp/sonic_qpos.npy')
    if q is None: return
    now=n.get_clock().now().to_msg(); h=Header(stamp=now,frame_id='odom')
    p,qu=q[0:3],q[3:7]
    if publish_map_odom:
        tm=TransformStamped();tm.header=Header(stamp=now,frame_id='map');tm.child_frame_id='odom';tm.transform.rotation.w=1.0;tf.sendTransform(tm)
    t=TransformStamped();t.header=h;t.child_frame_id='base_link'
    t.transform.translation.x=float(p[0]);t.transform.translation.y=float(p[1]);t.transform.translation.z=float(p[2])
    t.transform.rotation.w=float(qu[0]);t.transform.rotation.x=float(qu[1]);t.transform.rotation.y=float(qu[2]);t.transform.rotation.z=float(qu[3]);tf.sendTransform(t)
    yaw=math.atan2(2*(qu[0]*qu[3]+qu[1]*qu[2]),1-2*(qu[2]**2+qu[3]**2))
    o=Odometry();o.header=h;o.child_frame_id='base_link'
    o.pose.pose.position.x=float(p[0]);o.pose.pose.position.y=float(p[1]);o.pose.pose.position.z=float(p[2]);o.pose.pose.orientation=Quaternion(w=math.cos(yaw/2),z=math.sin(yaw/2));odom_pub.publish(o)
n.create_timer(0.02,pub)
print('Sensors: /odom /tf')
try: rclpy.spin(n)
except (KeyboardInterrupt, ExternalShutdownException): pass
