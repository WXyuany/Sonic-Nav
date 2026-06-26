import msgpack
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import ByteMultiArray


class CmdVelBridge(Node):
    def __init__(self, locomotion_mode=0, base_height=0.78):
        super().__init__("cmd_vel_bridge")
        self._locomotion_mode = locomotion_mode
        self._base_height = base_height
        self._last_cmd = [0.0, 0.0, 0.0]

        self._cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10
        )
        self._goal_pub = self.create_publisher(
            ByteMultiArray, "ControlPolicy/upper_body_pose", 10
        )
        self._timer = self.create_timer(0.2, self._timer_callback)

        self.get_logger().info("CmdVel bridge started → ControlPolicy/upper_body_pose")

    def _cmd_vel_callback(self, msg: Twist):
        self._last_cmd = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]

    def _timer_callback(self):
        payload = {
            "navigate_cmd": self._last_cmd,
            "locomotion_mode": self._locomotion_mode,
            "base_height_command": self._base_height,
            "toggle_policy_action": False,
        }
        packed = msgpack.packb(payload, use_bin_type=True)
        out = ByteMultiArray()
        out.data = [bytes([b]) for b in packed]
        self._goal_pub.publish(out)


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
