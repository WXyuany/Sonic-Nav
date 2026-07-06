#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, numpy as np, rclpy, mujoco
from rclpy.executors import ExternalShutdownException
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/g1_ros2_nav')
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help
from g1_ros2_nav.lidar_sim import Mid360Sim
from g1_ros2_nav.tmp_io import load_npy_if_ready
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

scene_arg = sys.argv[1] if len(sys.argv) > 1 else 'scene_43dof.xml'
try:
    scene = resolve_scene(scene_arg, repo_root=REPO)
except ValueError as exc:
    raise SystemExit(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc
xml = str(scene.abs_path)
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
mid360 = Mid360Sim(model, data)
lidar_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lidar')
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

rclpy.init()
n = Node('mid360')
pc_pub = n.create_publisher(PointCloud2, '/mid360_points', 10)
scan_pub = n.create_publisher(LaserScan, '/scan', 10)
tf_bc = TransformBroadcaster(n)

def quat_from_matrix(m):
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(w=0.25 * s, x=(m[2, 1] - m[1, 2]) / s,
                          y=(m[0, 2] - m[2, 0]) / s, z=(m[1, 0] - m[0, 1]) / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return Quaternion(w=(m[2, 1] - m[1, 2]) / s, x=0.25 * s,
                          y=(m[0, 1] + m[1, 0]) / s, z=(m[0, 2] + m[2, 0]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return Quaternion(w=(m[0, 2] - m[2, 0]) / s, x=(m[0, 1] + m[1, 0]) / s,
                          y=0.25 * s, z=(m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return Quaternion(w=(m[1, 0] - m[0, 1]) / s, x=(m[0, 2] + m[2, 0]) / s,
                      y=(m[1, 2] + m[2, 1]) / s, z=0.25 * s)

def pub():
    q = load_npy_if_ready('/tmp/sonic_qpos.npy')
    if q is None: return
    data.qpos[:len(q)] = q
    mujoco.mj_forward(model, data)
    now = n.get_clock().now().to_msg()
    lidar_pos = data.site_xpos[lidar_site_id]
    base_pos = data.xpos[base_body_id]
    base_rot = data.xmat[base_body_id].reshape(3, 3)
    lidar_rot = data.site_xmat[lidar_site_id].reshape(3, 3)
    rel_pos = base_rot.T @ (lidar_pos - base_pos)
    rel_rot = base_rot.T @ lidar_rot

    tl = TransformStamped()
    tl.header = Header(stamp=now, frame_id='base_link')
    tl.child_frame_id = 'lidar_link'
    tl.transform.translation.x = float(rel_pos[0])
    tl.transform.translation.y = float(rel_pos[1])
    tl.transform.translation.z = float(rel_pos[2])
    tl.transform.rotation = quat_from_matrix(rel_rot)
    tf_bc.sendTransform(tl)

    mid360.step()
    pts = mid360.points
    pc = PointCloud2()
    pc.header = Header(stamp=now, frame_id='lidar_link')
    pc.height = 1; pc.width = len(pts)
    pc.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                 PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                 PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    pc.point_step = 12; pc.row_step = pc.point_step * pc.width
    pc.is_bigendian = False; pc.is_dense = True
    pc.data = pts.tobytes()
    pc_pub.publish(pc)

    scan = LaserScan()
    scan.header = Header(stamp=now, frame_id='lidar_link')
    scan.angle_min = 0.0
    scan.angle_max = float(2 * math.pi - mid360.angles[1])
    scan.angle_increment = float(mid360.angles[1] - mid360.angles[0])
    scan.time_increment = 0.0
    scan.scan_time = 0.1
    scan.range_min = float(mid360.min_range)
    scan.range_max = float(mid360.max_range)
    scan.ranges = [float(r) for r in mid360.ranges]
    scan_pub.publish(scan)

n.create_timer(0.1, pub)
print('Mid360: /mid360_points /scan')
try:
    rclpy.spin(n)
except (KeyboardInterrupt, ExternalShutdownException):
    pass
