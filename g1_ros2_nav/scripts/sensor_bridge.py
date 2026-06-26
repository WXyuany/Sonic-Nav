#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, time, numpy as np, mujoco, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

REPO = os.path.expanduser("~/GR00T-WholeBodyControl")
sys.path.insert(0, os.path.join(REPO, "g1_ros2_nav"))
from g1_ros2_nav.lidar_sim import LidarSim

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")


class SensorBridge(Node):
    def __init__(self):
        super().__init__("sensor_bridge")
        xml = os.path.join(REPO, "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
        self._model = mujoco.MjModel.from_xml_path(xml)
        self._data = mujoco.MjData(self._model)
        self._lidar = LidarSim(self._model, self._data)
        self._odom = self.create_publisher(Odometry, "/odom", 10)
        self._scan = self.create_publisher(LaserScan, "/scan", 10)
        self._tf = TransformBroadcaster(self)
        self._timer = self.create_timer(0.05, self._publish)
        self.get_logger().info("Sensor bridge started")

    def _publish(self):
        try:
            qpos = np.load("/tmp/sonic_qpos.npy")
            self._data.qpos[:len(qpos)] = qpos
        except Exception:
            return
        mujoco.mj_forward(self._model, self._data)
        now = self.get_clock().now().to_msg()
        h = Header(stamp=now, frame_id="odom")
        pos = self._data.qpos[0:3].copy()
        quat = self._data.qpos[3:7].copy()
        t = TransformStamped()
        t.header = h
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        self._tf.sendTransform(t)
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        odom = Odometry()
        odom.header = h
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)
        odom.pose.pose.orientation = Quaternion(w=cy, z=sy)
        self._odom.publish(odom)
        self._lidar.step()
        d = self._lidar
        scan = LaserScan()
        scan.header = Header(stamp=now, frame_id="lidar_link")
        scan.angle_min = 0.0
        scan.angle_max = 2 * math.pi - d.angles[1]
        scan.angle_increment = float(d.angles[1] - d.angles[0])
        scan.range_min = float(d.min_range)
        scan.range_max = float(d.max_range)
        scan.ranges = [float(r) for r in d.ranges]
        self._scan.publish(scan)


def main():
    rclpy.init()
    bridge = SensorBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
