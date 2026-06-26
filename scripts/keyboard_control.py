#!/usr/bin/env -S /usr/bin/python3 -u
import os, sys, time, termios, tty, select, msgpack, rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")


class KeyboardControl(Node):
    def __init__(self):
        super().__init__("keyboard_control")
        self._pub = self.create_publisher(ByteMultiArray, "ControlPolicy/upper_body_pose", 10)
        self._vx = 0.0
        self._vy = 0.0
        self._vw = 0.0
        self._speed = 0.3
        self._started = False
        self._print_help()

    def _print_help(self):
        print()
        print("  Sonic-Nav Keyboard  |  Starting control...")
        print("  W/S: fwd/back  A/D: strafe  Q/E: turn")
        print("  1/2: speed -/+  SPACE: stop  ESC: quit")
        print()

    def send_cmd(self):
        if not self._started:
            self._started = True
            pl = {"navigate_cmd": [0, 0, 0], "locomotion_mode": 0,
                  "base_height_command": 0.78, "toggle_policy_action": True}
            m = ByteMultiArray()
            m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
            self._pub.publish(m)
            self.get_logger().info("Control start sent, robot standing")
            return

        pl = {"navigate_cmd": [self._vx, self._vy, self._vw],
              "locomotion_mode": 0, "base_height_command": 0.78,
              "toggle_policy_action": False}
        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self._pub.publish(m)

    def on_key(self, k):
        if k == "\x1b":
            return False
        k = k.lower()
        if k == "w":
            self._vx = self._speed
            self._vy = self._vw = 0.0
        elif k == "s":
            self._vx = -self._speed
            self._vy = self._vw = 0.0
        elif k == "a":
            self._vy = self._speed
            self._vx = self._vw = 0.0
        elif k == "d":
            self._vy = -self._speed
            self._vx = self._vw = 0.0
        elif k == "q":
            self._vw = 0.5
            self._vx = self._vy = 0.0
        elif k == "e":
            self._vw = -0.5
            self._vx = self._vy = 0.0
        elif k == " ":
            self._vx = self._vy = self._vw = 0.0
        elif k == "1":
            self._speed = max(0.1, self._speed - 0.1)
            print(f"  Speed: {self._speed:.1f}")
        elif k == "2":
            self._speed = min(1.5, self._speed + 0.1)
            print(f"  Speed: {self._speed:.1f}")
        return True


def main():
    try:
        old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
    except termios.error:
        print("Must run in a terminal", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    ctrl = KeyboardControl()
    running = True
    last = 0.0
    t0 = time.time()

    for _ in range(30):
        rclpy.spin_once(ctrl, timeout_sec=0.1)

    try:
        while running and rclpy.ok():
            if select.select([sys.stdin], [], [], 0.01)[0]:
                running = ctrl.on_key(sys.stdin.read(1))
            now = time.time()
            interval = 0.05 if time.time() - t0 < 3.0 else 0.1
            if now - last > interval:
                ctrl.send_cmd()
                last = now
            rclpy.spin_once(ctrl, timeout_sec=0.005)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        ctrl.destroy_node()
        rclpy.shutdown()
        print("\nDone")


if __name__ == "__main__":
    main()
