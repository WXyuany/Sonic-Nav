#!/usr/bin/env -S /usr/bin/python3 -u
import os, sys, time, termios, tty, select, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")


class TeleopTwist(Node):
    def __init__(self):
        super().__init__("g1_teleop")
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._vx = 0.0
        self._vy = 0.0
        self._vw = 0.0
        self._speed = 0.3
        self._print_help()

    def _print_help(self):
        print()
        print("  G1 Teleop  |  W/S: fwd/back  A/D: strafe  Q/E: turn")
        print("  1/2: speed -/+  SPACE: stop  ESC: quit")
        print(f"  Speed: {self._speed:.1f} m/s")
        print()

    def publish(self):
        msg = Twist()
        msg.linear.x = self._vx
        msg.linear.y = self._vy
        msg.angular.z = self._vw
        self._pub.publish(msg)

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
    teleop = TeleopTwist()
    running = True
    last = time.time()

    try:
        while running and rclpy.ok():
            if select.select([sys.stdin], [], [], 0.02)[0]:
                running = teleop.on_key(sys.stdin.read(1))
            now = time.time()
            if now - last > 0.1:
                teleop.publish()
                last = now
            rclpy.spin_once(teleop, timeout_sec=0.02)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        teleop.destroy_node()
        rclpy.shutdown()
        print("\nDone")


if __name__ == "__main__":
    main()
