#!/usr/bin/env python3
"""Keyboard control for Sonic-Nav robot. WASD to move, Q to quit."""

import os
import sys
import time
import threading
import termios
import tty
import select
import msgpack
import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

os.environ.setdefault("ROS_DOMAIN_ID", "42")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")


class KeyboardController(Node):
    def __init__(self):
        super().__init__("keyboard_control")
        self._pub = self.create_publisher(
            ByteMultiArray, "ControlPolicy/upper_body_pose", 10
        )
        self._vx = 0.0
        self._vy = 0.0
        self._vw = 0.0
        self._speed = 0.3
        self._started = False

        self.get_logger().info("Keyboard control ready")
        self._print_help()

    def _print_help(self):
        print()
        print("  ╔══════════════════════════════╗")
        print("  ║   Sonic-Nav Keyboard Ctrl   ║")
        print("  ╠══════════════════════════════╣")
        print("  ║  W/S    : forward / back    ║")
        print("  ║  A/D    : left / right      ║")
        print("  ║  Q/E    : turn left / right ║")
        print("  ║  1/2    : speed -/+         ║")
        print("  ║  SPACE  : start control     ║")
        print("  ║  ESC    : quit              ║")
        print("  ╚══════════════════════════════╝")
        print(f"  Speed: {self._speed:.1f} m/s")
        print()

    def send_cmd(self):
        payload = {
            "navigate_cmd": [self._vx, self._vy, self._vw],
            "locomotion_mode": 0,
            "base_height_command": 0.78,
            "toggle_policy_action": not self._started,
        }
        if not self._started and (self._vx != 0 or self._vy != 0 or self._vw != 0):
            self._started = True
            self.get_logger().info("Control started!")

        packed = msgpack.packb(payload, use_bin_type=True)
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in packed]
        self._pub.publish(msg)

    def handle_key(self, key):
        if key == "\x1b":  # ESC
            return False
        elif key in ("w", "W"):
            self._vx = self._speed
        elif key in ("s", "S"):
            self._vx = -self._speed
        elif key in ("a", "A"):
            self._vy = self._speed
        elif key in ("d", "D"):
            self._vy = -self._speed
        elif key in ("q", "Q"):
            self._vw = 0.5
        elif key in ("e", "E"):
            self._vw = -0.5
        elif key == "1":
            self._speed = max(0.1, self._speed - 0.1)
            print(f"  Speed: {self._speed:.1f}")
        elif key == "2":
            self._speed = min(1.5, self._speed + 0.1)
            print(f"  Speed: {self._speed:.1f}")
        elif key == " ":
            self._started = True
            self._vx = 0.0
            self._vy = 0.0
            self._vw = 0.0
            self.get_logger().info("Starting control...")
        else:
            self._vx = 0.0
            self._vy = 0.0
            self._vw = 0.0
        return True


def get_key():
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return None


def main():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    rclpy.init()
    ctrl = KeyboardController()
    running = True
    last_send = time.time()

    try:
        while running and rclpy.ok():
            key = get_key()
            if key:
                running = ctrl.handle_key(key)

            if time.time() - last_send > 0.1:
                ctrl.send_cmd()
                last_send = time.time()

            rclpy.spin_once(ctrl, timeout_sec=0.01)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        ctrl.destroy_node()
        rclpy.shutdown()
        print("\nBye!")


if __name__ == "__main__":
    main()
