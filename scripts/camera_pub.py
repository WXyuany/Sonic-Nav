#!/usr/bin/env -S /usr/bin/python3
import os, os.path, sys, math, numpy as np
os.environ['DISPLAY'] = ':1'
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
os.environ.setdefault('ROS_DOMAIN_ID', '42')

import rclpy, mujoco
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml = REPO + '/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
W, H = 640, 480

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')

rclpy.init()
n = Node('camera')
rgb_pub = n.create_publisher(Image, '/camera/color/image_raw', 10)
depth_pub = n.create_publisher(Image, '/camera/depth/image_raw', 10)
ci_pub = n.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
tf_bc = TransformBroadcaster(n)
renderer = None

def _get_renderer():
    global renderer
    if renderer is None:
        renderer = mujoco.Renderer(model, W, H)
    return renderer

def pub():
    try:
        q = np.load('/tmp/sonic_qpos.npy')
        data.qpos[:len(q)] = q
    except: return
    mujoco.mj_forward(model, data)
    now = n.get_clock().now().to_msg()

    nc = TransformStamped()
    nc.header = Header(stamp=now, frame_id='base_link'); nc.child_frame_id = 'head_camera'
    nc.transform.translation.x = 0.06; nc.transform.translation.z = 0.45
    nc.transform.rotation.w = 1.0
    tf_bc.sendTransform(nc)

    try:
        r = _get_renderer()
        r.update_scene(data, camera=cam_id)
        rgb = r.render()
    except Exception as e:
        return

    rgb_msg = Image(); rgb_msg.header = Header(stamp=now, frame_id='head_camera')
    rgb_msg.height = H; rgb_msg.width = W
    rgb_msg.encoding = 'rgb8'; rgb_msg.step = W * 3
    rgb_msg.data = rgb.tobytes()
    rgb_pub.publish(rgb_msg)

    try:
        r.enable_depth_rendering()
        depth = r.render()
        r.disable_depth_rendering()
        d = Image(); d.header = Header(stamp=now, frame_id='head_camera')
        d.height = H; d.width = W
        d.encoding = '32FC1'; d.step = W * 4
        d.data = depth.astype(np.float32).tobytes()
        depth_pub.publish(d)
    except: pass

    ci = CameraInfo(); ci.header = Header(stamp=now, frame_id='head_camera')
    ci.height = H; ci.width = W
    fx = W / (2 * math.tan(math.radians(87) / 2))
    ci.k = [fx, 0.0, W/2, 0.0, fx, H/2, 0.0, 0.0, 1.0]
    ci_pub.publish(ci)

n.create_timer(0.5, pub)
print('Camera: /camera/color /camera/depth')
try: rclpy.spin(n)
except: pass
