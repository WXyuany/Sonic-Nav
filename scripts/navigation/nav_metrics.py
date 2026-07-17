#!/usr/bin/env -S /usr/bin/python3
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO = os.path.dirname(SCRIPTS_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.nav.metrics import NavigationMetrics
from gear_sonic.nav.params import REPO_ROOT, load_config, overlay_env_scalars


os.environ.update({
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_DOMAIN_ID": "42",
})


DEFAULTS = {
    "goal_tolerance": 0.45,
    "collision_radius": 0.18,
    "stuck_speed": 0.035,
    "stuck_cmd": 0.08,
    "stuck_timeout": 3.0,
    "summary_dir": "logs/nav_metrics",
    "sample_period": 0.10,
    "local_costmap_threshold": 35,
    "topics": {
        "odom": "/odom",
        "goal": "/goal_pose",
        "cmd_vel": "/sonic_nav/cmd_vel_safe",
        "local_costmap": "/local_costmap",
        "summary": "/sonic_nav/metrics_summary",
    },
}


ENV_OVERRIDES = {
    "SONIC_NAV_METRICS_GOAL_TOL": ("goal_tolerance", float),
    "SONIC_NAV_METRICS_COLLISION_RADIUS": ("collision_radius", float),
    "SONIC_NAV_METRICS_STUCK_SPEED": ("stuck_speed", float),
    "SONIC_NAV_METRICS_STUCK_CMD": ("stuck_cmd", float),
    "SONIC_NAV_METRICS_STUCK_TIMEOUT": ("stuck_timeout", float),
    "SONIC_NAV_METRICS_SAMPLE_PERIOD": ("sample_period", float),
}


class NavMetricsNode(Node):
    def __init__(self):
        super().__init__("nav_metrics")
        cfg = overlay_env_scalars(load_config("eval", DEFAULTS, "SONIC_NAV_METRICS_CONFIG"), ENV_OVERRIDES)
        self.cfg = cfg
        self.topics = cfg["topics"]
        self.metrics = NavigationMetrics(
            goal_tolerance=float(cfg["goal_tolerance"]),
            collision_radius=float(cfg["collision_radius"]),
            stuck_speed=float(cfg["stuck_speed"]),
            stuck_cmd=float(cfg["stuck_cmd"]),
            stuck_timeout=float(cfg["stuck_timeout"]),
        )
        self.pose = None
        self.pending_goal = None
        self.last_cmd = (0.0, 0.0)
        self.clearance = None
        self.last_summary_pub = 0.0
        self.run_index = 0
        self.costmap_threshold = int(cfg.get("local_costmap_threshold", 35))
        self.summary_dir = Path(cfg["summary_dir"]).expanduser()
        if not self.summary_dir.is_absolute():
            self.summary_dir = REPO_ROOT / self.summary_dir
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.summary_pub = self.create_publisher(String, self.topics["summary"], 10)
        self.create_subscription(Odometry, self.topics["odom"], self.on_odom, 20)
        self.create_subscription(PoseStamped, self.topics["goal"], self.on_goal, 10)
        self.create_subscription(TwistStamped, self.topics["cmd_vel"], self.on_cmd, 20)
        self.create_subscription(OccupancyGrid, self.topics["local_costmap"], self.on_costmap, 10)
        self.timer = self.create_timer(max(0.02, float(cfg["sample_period"])), self.tick)
        self.get_logger().info(f"Navigation metrics ready; writing summaries to {self.summary_dir}")

    def on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def on_goal(self, msg: PoseStamped):
        goal = self._goal_to_odom(msg)
        if self.metrics.active:
            self._finish("superseded")
        self.pending_goal = goal
        if self.pose is not None:
            self._start_pending_goal()

    def on_cmd(self, msg: TwistStamped):
        self.last_cmd = (float(msg.twist.linear.x), float(msg.twist.angular.z))

    def on_costmap(self, msg: OccupancyGrid):
        if msg.info.width == 0 or msg.info.height == 0:
            self.clearance = None
            return
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        idx = np.argwhere(data >= self.costmap_threshold)
        if len(idx) == 0:
            self.clearance = None
            return
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        x = ox + (idx[:, 1].astype(np.float32) + 0.5) * res
        y = oy + (idx[:, 0].astype(np.float32) + 0.5) * res
        self.clearance = float(np.min(np.hypot(x, y)))

    def tick(self):
        if self.pending_goal is not None and self.pose is not None:
            self._start_pending_goal()
        if not self.metrics.active or self.pose is None:
            return
        now = time.monotonic()
        self.metrics.update(now, self.pose, self.last_cmd, self.clearance)
        if now - self.last_summary_pub > 1.0:
            self._publish_summary(now)
            self.last_summary_pub = now
        if self.metrics.reached:
            self._finish("reached")

    def close(self):
        if self.metrics.active:
            self._finish("shutdown")

    def _start_pending_goal(self):
        self.run_index += 1
        now = time.monotonic()
        self.metrics.start(now, self.pose, self.pending_goal)
        self.get_logger().info(
            f"Metrics run {self.run_index}: goal=({self.pending_goal[0]:.2f}, {self.pending_goal[1]:.2f})"
        )
        self.pending_goal = None
        self._publish_summary(now)

    def _finish(self, reason: str):
        now = time.monotonic()
        summary = self.metrics.summary(now)
        summary["reason"] = reason
        summary["run_index"] = self.run_index
        summary["finished_wall_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record = {
            "summary": summary,
            "history": self.metrics.history,
        }
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = self.summary_dir / f"nav_{stamp}_{self.run_index:03d}_{reason}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.get_logger().info(f"Metrics {reason}: {out}")
        self._publish_summary(now, extra={"reason": reason, "file": str(out)})
        self.metrics.active = False

    def _publish_summary(self, now: float, extra: dict | None = None):
        summary = self.metrics.summary(now)
        if extra:
            summary.update(extra)
        msg = String()
        msg.data = json.dumps(summary, separators=(",", ":"))
        self.summary_pub.publish(msg)

    def _goal_to_odom(self, msg: PoseStamped):
        frame = msg.header.frame_id or "odom"
        if frame == "odom":
            return (float(msg.pose.position.x), float(msg.pose.position.y))
        try:
            tf = self.tf_buffer.lookup_transform("odom", frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"No TF {frame}->odom for metrics goal, using raw goal: {exc}")
            return (float(msg.pose.position.x), float(msg.pose.position.y))
        q = tf.transform.rotation
        t = tf.transform.translation
        rot = self._quat_to_rot(q.w, q.x, q.y, q.z)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        out = p @ rot.T + np.array([t.x, t.y, t.z], dtype=np.float32)
        return (float(out[0]), float(out[1]))

    @staticmethod
    def _quat_to_rot(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)


def main():
    rclpy.init()
    node = NavMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
