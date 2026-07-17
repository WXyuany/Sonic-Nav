#!/usr/bin/env -S /usr/bin/python3
import os
import sys
import time

import msgpack
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = os.path.dirname(SCRIPTS_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.nav.control import ControlConfig, SonicControlPayloadBuilder, VelocityLimiter
from gear_sonic.nav.params import load_config, overlay_env_scalars


os.environ.update({
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_DOMAIN_ID": "42",
})


DEFAULTS = {
    "publish_rate": 25.0,
    "command_timeout": 0.35,
    "locomotion_mode": 0,
    "base_height": 0.78,
    "upper_body_mode": "navigation",
    "limits": {
        "max_v": 0.65,
        "max_w": 0.85,
        "max_dv": 0.08,
        "max_dw": 0.12,
        "v_deadband": 0.025,
        "w_deadband": 0.030,
    },
    "topics": {
        "cmd_vel": "/cmd_vel_nav",
        "sonic_payload": "ControlPolicy/upper_body_pose",
        "safe_cmd_vel": "/sonic_nav/cmd_vel_safe",
        "diagnostics": "/sonic_nav/diagnostics",
        "mode": "/sonic_nav/upper_body_mode",
    },
}


ENV_OVERRIDES = {
    "SONIC_CONTROL_MAX_V": ("limits.max_v", float),
    "SONIC_CONTROL_MAX_W": ("limits.max_w", float),
    "SONIC_CONTROL_MAX_DV": ("limits.max_dv", float),
    "SONIC_CONTROL_MAX_DW": ("limits.max_dw", float),
    "SONIC_CONTROL_V_DEADBAND": ("limits.v_deadband", float),
    "SONIC_CONTROL_W_DEADBAND": ("limits.w_deadband", float),
    "SONIC_CONTROL_TIMEOUT": ("command_timeout", float),
    "SONIC_CONTROL_LOCOMOTION_MODE": ("locomotion_mode", int),
    "SONIC_CONTROL_BASE_HEIGHT": ("base_height", float),
}


class NavControlAdapter(Node):
    def __init__(self):
        super().__init__("nav_control_adapter")
        cfg = overlay_env_scalars(load_config("control", DEFAULTS, "SONIC_CONTROL_CONFIG"), ENV_OVERRIDES)
        self.cfg = ControlConfig.from_dict(cfg)
        self.topics = cfg["topics"]
        self.limiter = VelocityLimiter(self.cfg)
        self.payload_builder = SonicControlPayloadBuilder(self.cfg)
        self.last_cmd = (0.0, 0.0)
        self.last_cmd_time = 0.0
        self.upper_body_mode = self.cfg.upper_body_mode
        self.timed_out = True

        self.sonic_pub = self.create_publisher(ByteMultiArray, self.topics["sonic_payload"], 10)
        self.safe_cmd_pub = self.create_publisher(TwistStamped, self.topics["safe_cmd_vel"], 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, self.topics["diagnostics"], 10)
        self.create_subscription(Twist, self.topics["cmd_vel"], self.on_cmd, 10)
        self.create_subscription(String, self.topics["mode"], self.on_mode, 10)
        self.timer = self.create_timer(1.0 / max(1.0, self.cfg.publish_rate), self.tick)
        self.get_logger().info(
            f"Adapter ready: {self.topics['cmd_vel']} -> {self.topics['sonic_payload']} "
            f"(upper_body={self.upper_body_mode}, timeout={self.cfg.command_timeout:.2f}s)"
        )

    def on_cmd(self, msg: Twist):
        self.last_cmd = (float(msg.linear.x), float(msg.angular.z))
        self.last_cmd_time = time.monotonic()
        self.timed_out = False

    def on_mode(self, msg: String):
        mode = msg.data.strip().lower()
        if mode not in {"navigation", "locked", "idle", "manipulation"}:
            self.get_logger().warn(f"Unknown upper body mode '{msg.data}', keeping {self.upper_body_mode}")
            return
        self.upper_body_mode = mode
        self.get_logger().info(f"Upper body mode: {mode}")

    def tick(self):
        now = time.monotonic()
        stale = now - self.last_cmd_time > self.cfg.command_timeout
        if stale:
            self.last_cmd = (0.0, 0.0)
            if not self.timed_out:
                self.limiter.reset()
            self.timed_out = True

        v, w = self.limiter.limit(self.last_cmd[0], self.last_cmd[1], slew=not stale)
        self._publish_sonic(v, w)
        self._publish_safe_cmd(v, w)
        self._publish_diag(v, w, stale)

    def _publish_sonic(self, v: float, w: float):
        payload = self.payload_builder.payload(v, w, upper_body_mode=self.upper_body_mode)
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in msgpack.packb(payload, use_bin_type=True)]
        self.sonic_pub.publish(msg)

    def _publish_safe_cmd(self, v: float, w: float):
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.safe_cmd_pub.publish(msg)

    def _publish_diag(self, v: float, w: float, stale: bool):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "sonic_nav_control_adapter"
        status.hardware_id = "sonic_g1_sim"
        status.level = DiagnosticStatus.WARN if stale else DiagnosticStatus.OK
        status.message = "cmd_vel timeout" if stale else "ok"
        status.values = [
            KeyValue(key="cmd_v", value=f"{v:.3f}"),
            KeyValue(key="cmd_w", value=f"{w:.3f}"),
            KeyValue(key="upper_body_mode", value=self.upper_body_mode),
            KeyValue(key="locomotion_mode", value=str(self.cfg.locomotion_mode)),
            KeyValue(key="base_height", value=f"{self.cfg.base_height:.3f}"),
        ]
        arr.status.append(status)
        self.diag_pub.publish(arr)


def main():
    rclpy.init()
    node = NavControlAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
