#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy, mujoco
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
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
renderer = mujoco.Renderer(model, W, H)

rclpy.init()
n = Node('camera')
rgb_pub = n.create_publisher(Image, '/camera/color/image_raw', 10)
depth_pub = n.create_publisher(Image, '/camera/depth/image_raw', 10)
ci_pub = n.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
tf_bc = TransformBroadcaster(n)

def pub():
    try:
        q = np.load('/tmp/sonic_qpos.npy')
        data.qpos[:len(q)] = q
    except: return
    mujoco.mj_forward(model, data)
    now = n.get_clock().now().to_msg()

    nc = Header(stamp=now, frame_id='base_link')
    nc.child_frame_id = 'head_camera'
    nc.transform.translation.x = 0.06
    nc.transform.translation.z = 0.45
    nc.transform.rotation.w = 1.0
    tf_bc.sendTransform(nc)

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()

    rgb_msg = Image()
    rgb_msg.header = Header(stamp=now, frame_id='head_camera')
    rgb_msg.height = H; rgb_msg.width = W
    rgb_msg.encoding = 'rgb8'; rgb_msg.step = W * 3
    rgb_msg.data = rgb.tobytes()
    rgb_pub.publish(rgb_msg)

    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    depth_msg = Image()
    depth_msg.header = Header(stamp=now, frame_id='head_camera')
    depth_msg.height = H; depth_msg.width = W
    depth_msg.encoding = '32FC1'; depth_msg.step = W * 4
    depth_msg.data = depth.astype(np.float32).tobytes()
    depth_pub.publish(depth_msg)

    ci = CameraInfo()
    ci.header = Header(stamp=now, frame_id='head_camera')
    ci.height = H; ci.width = W
    fx = W / (2 * math.tan(math.radians(87) / 2))
    ci.k = [fx, 0.0, W/2, 0.0, fx, H/2, 0.0, 0.0, 1.0]
    ci_pub.publish(ci)

n.create_timer(0.2, pub)
print('Camera: /camera/color /camera/depth')
try: rclpy.spin(n)
except: pass
