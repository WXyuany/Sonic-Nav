#!/usr/bin/env -S /usr/bin/python3
import os,math,time,numpy as np,rclpy
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped,Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
rclpy.init()
n=Node('sensors')
op=n.create_publisher(Odometry,'/odom',10)
tf=TransformBroadcaster(n)
def pub():
    try: q=np.load('/tmp/sonic_qpos.npy')
    except: return
    now=n.get_clock().now().to_msg()
    h=Header(stamp=now,frame_id='odom')
    p=q[0:3];qu=q[3:7]
    tm=TransformStamped();tm.header=Header(stamp=now,frame_id='map');tm.child_frame_id='odom';tm.transform.rotation.w=1.0
    tf.sendTransform(tm)
    t=TransformStamped();t.header=h;t.child_frame_id='base_link'
    t.transform.translation.x=float(p[0]);t.transform.translation.y=float(p[1]);t.transform.translation.z=0.0
    t.transform.rotation.w=float(qu[0]);t.transform.rotation.x=float(qu[1]);t.transform.rotation.y=float(qu[2]);t.transform.rotation.z=float(qu[3])
    tf.sendTransform(t)
    yaw=math.atan2(2*(qu[0]*qu[3]+qu[1]*qu[2]),1-2*(qu[2]**2+qu[3]**2))
    o=Odometry();o.header=h;o.child_frame_id='base_link'
    o.pose.pose.position.x=float(p[0]);o.pose.pose.position.y=float(p[1])
    o.pose.pose.orientation=Quaternion(w=math.cos(yaw/2),z=math.sin(yaw/2))
    op.publish(o)
n.create_timer(0.05,pub)
print('Sensors running - /odom /tf')
try: rclpy.spin(n)
except: pass
