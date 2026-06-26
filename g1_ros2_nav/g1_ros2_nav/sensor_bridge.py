import os, sys, math, time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

os.environ.setdefault("RMW_IMPLEMENTATION","rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY","1")
os.environ.setdefault("ROS_DOMAIN_ID","42")

class SensorBridge(Node):
    def __init__(self):
        super().__init__("sensor_bridge")
        self._odom = self.create_publisher(Odometry,"/odom",10)
        self._scan = self.create_publisher(LaserScan,"/scan",10)
        self._tf = TransformBroadcaster(self)
        self._timer = self.create_timer(0.05,self._publish)
        self.get_logger().info("Sensor bridge started")

    def _publish(self):
        now = self.get_clock().now().to_msg()
        h = Header(stamp=now,frame_id="odom")
        t = TransformStamped()
        t.header=h; t.child_frame_id="base_link"
        t.transform.translation.z=0.8; t.transform.rotation.w=1.0
        self._tf.sendTransform(t)
        odom = Odometry()
        odom.header=h; odom.child_frame_id="base_link"
        odom.pose.pose.position.z=0.8; odom.pose.pose.orientation=Quaternion(w=1.0)
        self._odom.publish(odom)
        scan = LaserScan()
        scan.header=Header(stamp=now,frame_id="lidar_link")
        scan.angle_min=0.0; scan.angle_max=6.283; scan.angle_increment=0.01745
        scan.range_min=0.1; scan.range_max=30.0; scan.ranges=[2.0]*360
        self._scan.publish(scan)

def main():
    rclpy.init()
    n = SensorBridge()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()

if __name__=="__main__": main()
