#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy, mujoco
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/g1_ros2_nav')
from g1_ros2_nav.lidar_sim import Mid360Sim
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml = REPO + '/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mid360 = Mid360Sim(model, data)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
renderer = None

def _get_renderer():
    global renderer
    if renderer is None:
        renderer = mujoco.Renderer(model, 640, 480)
    return renderer

rclpy.init()
n = Node('mid360')
pc_pub = n.create_publisher(PointCloud2, '/mid360_points', 10)
rgb_pub = n.create_publisher(Image, '/camera/color/image_raw', 10)
depth_pub = n.create_publisher(Image, '/camera/depth/image_raw', 10)
ci_pub = n.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
tf_bc = TransformBroadcaster(n)

def pub_pointcloud():
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

def pub_camera():
    try:
        r = _get_renderer()
        r.update_scene(data, camera=cam_id)
        rgb = r.render()
    except Exception as e:
        return
    now = n.get_clock().now().to_msg()
    nc = Header(stamp=now, frame_id='base_link')
    nc.child_frame_id = 'head_camera'
    nc.transform.translation.x = 0.06; nc.transform.translation.z = 0.45
    nc.transform.rotation.w = 1.0
    tf_bc.sendTransform(nc)

    rgb_msg = Image()
    rgb_msg.header = Header(stamp=now, frame_id='head_camera')
    rgb_msg.height = 480; rgb_msg.width = 640
    rgb_msg.encoding = 'rgb8'; rgb_msg.step = 640 * 3
    rgb_msg.data = rgb.tobytes()
    rgb_pub.publish(rgb_msg)

    r.enable_depth_rendering()
    depth = r.render()
    r.disable_depth_rendering()
    depth_msg = Image()
    depth_msg.header = Header(stamp=now, frame_id='head_camera')
    depth_msg.height = 480; depth_msg.width = 640
    depth_msg.encoding = '32FC1'; depth_msg.step = 640 * 4
    depth_msg.data = depth.astype(np.float32).tobytes()
    depth_pub.publish(depth_msg)

    ci = CameraInfo()
    ci.header = Header(stamp=now, frame_id='head_camera')
    ci.height = 480; ci.width = 640
    fx = 640 / (2 * math.tan(math.radians(87) / 2))
    ci.k = [fx, 0.0, 320.0, 0.0, fx, 240.0, 0.0, 0.0, 1.0]
    ci_pub.publish(ci)

n.create_timer(0.3, pub_pointcloud)
n.create_timer(0.3, pub_camera)
print('Sensors: /mid360_points /camera/*')
try: rclpy.spin(n)
except: pass
