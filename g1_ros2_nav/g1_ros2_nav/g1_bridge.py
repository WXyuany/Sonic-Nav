import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return Quaternion(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


class G1ROSBridge(Node):
    def __init__(self, sim_env=None):
        super().__init__("g1_bridge")
        self._sim_env = sim_env

        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._scan_timer = self.create_timer(0.05, self._publish_scan)
        self._odom_timer = self.create_timer(0.02, self._publish_odom_tf)

        self.get_logger().info("G1 ROS2 Bridge started")

    def _publish_odom_tf(self):
        if self._sim_env is None:
            return

        mj_data = self._sim_env.mj_data
        pelvis_id = self._sim_env.root_body_id
        pos = mj_data.qpos[pelvis_id * 7 : pelvis_id * 7 + 3].copy()
        quat = mj_data.qpos[pelvis_id * 7 + 3 : pelvis_id * 7 + 7].copy()

        now = self.get_clock().now().to_msg()
        header = Header(stamp=now, frame_id="odom")

        t = TransformStamped()
        t.header = header
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        self._tf_broadcaster.sendTransform(t)

        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
        )
        odom = Odometry()
        odom.header = header
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        self._odom_pub.publish(odom)

    def _publish_scan(self):
        if self._sim_env is None or self._sim_env.lidar_sim is None:
            return

        self._sim_env.lidar_step()
        lidar_data = self._sim_env.get_lidar_data()
        if lidar_data is None:
            return

        scan = LaserScan()
        scan.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="lidar_link")
        scan.angle_min = float(lidar_data["angles"][0])
        scan.angle_max = float(lidar_data["angles"][-1])
        scan.angle_increment = float(lidar_data["angles"][1] - lidar_data["angles"][0])
        scan.range_min = float(lidar_data["range_min"])
        scan.range_max = float(lidar_data["range_max"])
        scan.ranges = [float(r) for r in lidar_data["ranges"]]
        self._scan_pub.publish(scan)


def main():
    rclpy.init()
    node = G1ROSBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
