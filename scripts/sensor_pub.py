#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
rclpy.init()
n=Node('sensors')
odom_pub=n.create_publisher(Odometry,'/odom',10)
scan_pub=n.create_publisher(LaserScan,'/scan',10)
tf=TransformBroadcaster(n)
def pub():
    try: q=np.load('/tmp/sonic_qpos.npy')
    except: return
    now=n.get_clock().now().to_msg(); h=Header(stamp=now,frame_id='odom')
    p,qu=q[0:3],q[3:7]
    tm=TransformStamped();tm.header=Header(stamp=now,frame_id='map');tm.child_frame_id='odom';tm.transform.rotation.w=1.0;tf.sendTransform(tm)
    t=TransformStamped();t.header=h;t.child_frame_id='base_link'
    t.transform.translation.x=float(p[0]);t.transform.translation.y=float(p[1])
    t.transform.rotation.w=float(qu[0]);t.transform.rotation.x=float(qu[1]);t.transform.rotation.y=float(qu[2]);t.transform.rotation.z=float(qu[3]);tf.sendTransform(t)
    yaw=math.atan2(2*(qu[0]*qu[3]+qu[1]*qu[2]),1-2*(qu[2]**2+qu[3]**2))
    o=Odometry();o.header=h;o.child_frame_id='base_link'
    o.pose.pose.position.x=float(p[0]);o.pose.pose.position.y=float(p[1]);o.pose.pose.orientation=Quaternion(w=math.cos(yaw/2),z=math.sin(yaw/2));odom_pub.publish(o)
    try:
        r=np.load('/tmp/sonic_lidar.npy')
        s=LaserScan();s.header=Header(stamp=now,frame_id='lidar_link');s.angle_min=0.0;s.angle_max=2*math.pi*(1-1/len(r))
        s.angle_increment=2*math.pi/len(r);s.range_min=0.1;s.range_max=30.0;s.ranges=[float(x) for x in r];scan_pub.publish(s)
    except: pass
n.create_timer(0.02,pub)
print('Sensors: /odom /scan /tf')
try: rclpy.spin(n)
except: pass
