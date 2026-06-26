#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, time, numpy as np, rclpy, mujoco
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/g1_ros2_nav')
from g1_ros2_nav.lidar_sim import LidarSim, Mid360Sim
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml = REPO + '/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
lidar = LidarSim(model, data)
mid360 = Mid360Sim(model, data)

rclpy.init()
n = Node('sensors')
odom_pub = n.create_publisher(Odometry, '/odom', 10)
scan_pub = n.create_publisher(LaserScan, '/scan', 10)
pc_pub = n.create_publisher(PointCloud2, '/mid360_points', 10)
tf = TransformBroadcaster(n)

def pub():
    try:
        q = np.load('/tmp/sonic_qpos.npy')
        data.qpos[:len(q)] = q
    except:
        return
    mujoco.mj_forward(model, data)
    now = n.get_clock().now().to_msg()
    h = Header(stamp=now, frame_id='odom')
    p = data.qpos[0:3]; qu = data.qpos[3:7]

    tm = TransformStamped(); tm.header = Header(stamp=now, frame_id='map')
    tm.child_frame_id = 'odom'; tm.transform.rotation.w = 1.0
    tf.sendTransform(tm)
    t = TransformStamped(); t.header = h; t.child_frame_id = 'base_link'
    t.transform.translation.x = float(p[0]); t.transform.translation.y = float(p[1])
    t.transform.rotation.w = float(qu[0]); t.transform.rotation.x = float(qu[1])
    t.transform.rotation.y = float(qu[2]); t.transform.rotation.z = float(qu[3])
    tf.sendTransform(t)
    # lidar_link on torso
    tl = TransformStamped(); tl.header = h; tl.child_frame_id = 'lidar_link'
    tl.transform.translation.z = 0.30; tl.transform.rotation.w = 1.0
    tf.sendTransform(tl)
    yaw = math.atan2(2*(qu[0]*qu[3]+qu[1]*qu[2]), 1-2*(qu[2]**2+qu[3]**2))
    o = Odometry(); o.header = h; o.child_frame_id = 'base_link'
    o.pose.pose.position.x = float(p[0]); o.pose.pose.position.y = float(p[1])
    o.pose.pose.orientation = Quaternion(w=math.cos(yaw/2), z=math.sin(yaw/2))
    odom_pub.publish(o)

    lidar.step()
    scan = LaserScan()
    scan.header = Header(stamp=now, frame_id='lidar_link')
    scan.angle_min = 0.0; scan.angle_max = 2*math.pi - lidar.angles[1]
    scan.angle_increment = float(lidar.angles[1] - lidar.angles[0])
    scan.range_min = float(lidar.min_range); scan.range_max = float(lidar.max_range)
    scan.ranges = [float(r) for r in lidar.ranges]
    scan_pub.publish(scan)

def pub_mid360():
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

n.create_timer(0.05, pub)
n.create_timer(0.2, pub_mid360)
print('Sensors: /odom /scan /mid360_points /tf')
try: rclpy.spin(n)
except: pass
