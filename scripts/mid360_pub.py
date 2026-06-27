#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy, mujoco
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/g1_ros2_nav')
from g1_ros2_nav.lidar_sim import Mid360Sim
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml = REPO + '/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mid360 = Mid360Sim(model, data)

rclpy.init()
n = Node('mid360')
pc_pub = n.create_publisher(PointCloud2, '/mid360_points', 10)
tf_bc = TransformBroadcaster(n)

def pub():
    try:
        q = np.load('/tmp/sonic_qpos.npy')
        data.qpos[:len(q)] = q
    except: return
    mujoco.mj_forward(model, data)
    now = n.get_clock().now().to_msg()
    # Publish base_link -> lidar_link TF
    tl = TransformStamped()
    tl.header = Header(stamp=now, frame_id='base_link')
    tl.child_frame_id = 'lidar_link'
    tl.transform.translation.z = 0.30
    tl.transform.rotation.w = 1.0
    tf_bc.sendTransform(tl)
    mid360.step()
    pts = mid360.points
    pc = PointCloud2()
    pc.header = Header(stamp=n.get_clock().now().to_msg(), frame_id='lidar_link')
    pc.height = 1; pc.width = len(pts)
    pc.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                 PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                 PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    pc.point_step = 12; pc.row_step = pc.point_step * pc.width
    pc.is_bigendian = False; pc.is_dense = True
    pc.data = pts.tobytes()
    pc_pub.publish(pc)

n.create_timer(0.3, pub)
print('Mid360: /mid360_points')
try: rclpy.spin(n)
except: pass
