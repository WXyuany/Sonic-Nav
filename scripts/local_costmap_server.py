#!/usr/bin/env -S /usr/bin/python3
import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.nav.costmap import (
    LocalCostmapConfig,
    build_local_costmap,
    filter_base_points,
    occupied_points_from_grid,
    voxel_downsample_2d,
)
from gear_sonic.nav.params import load_config, overlay_env_scalars


os.environ.update({
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_DOMAIN_ID": "42",
})


DEFAULTS = {
    "resolution": 0.06,
    "forward_range": 6.0,
    "backward_range": 1.2,
    "lateral_range": 3.2,
    "obstacle_radius": 0.05,
    "inflation_radius": 0.36,
    "occupied_threshold": 35,
    "cloud_filter": {
        "robot_radius": 0.42,
        "max_range": 7.0,
        "min_z": -0.55,
        "max_z": 0.85,
        "voxel": 0.08,
        "max_points": 1400,
        "min_x": -1.2,
    },
    "topics": {
        "input_cloud": "/mid360_points",
        "local_costmap": "/local_costmap",
        "inflated_points": "/local_costmap/occupied_points",
        "diagnostics": "/sonic_nav/costmap_status",
    },
}


ENV_OVERRIDES = {
    "SONIC_COSTMAP_RESOLUTION": ("resolution", float),
    "SONIC_COSTMAP_FORWARD": ("forward_range", float),
    "SONIC_COSTMAP_BACKWARD": ("backward_range", float),
    "SONIC_COSTMAP_LATERAL": ("lateral_range", float),
    "SONIC_COSTMAP_INFLATION": ("inflation_radius", float),
    "SONIC_COSTMAP_OCC_THRESHOLD": ("occupied_threshold", int),
}


class LocalCostmapServer(Node):
    def __init__(self):
        super().__init__("local_costmap_server")
        cfg = overlay_env_scalars(load_config("costmap", DEFAULTS, "SONIC_COSTMAP_CONFIG"), ENV_OVERRIDES)
        self.cfg = LocalCostmapConfig.from_dict(cfg)
        self.filter_cfg = cfg["cloud_filter"]
        self.topics = cfg["topics"]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_pub = self.create_publisher(OccupancyGrid, self.topics["local_costmap"], 10)
        self.points_pub = self.create_publisher(PointCloud2, self.topics["inflated_points"], 10)
        self.status_pub = self.create_publisher(String, self.topics["diagnostics"], 10)
        self.create_subscription(PointCloud2, self.topics["input_cloud"], self.on_cloud, 10)
        self.get_logger().info(
            f"Local costmap ready: {self.cfg.width}x{self.cfg.height} at {self.cfg.resolution:.2f}m "
            f"from {self.topics['input_cloud']}"
        )

    def on_cloud(self, msg: PointCloud2):
        if msg.point_step != 12 or len(msg.fields) < 3:
            self.get_logger().warn("Unsupported PointCloud2 layout; expected packed float32 xyz")
            return
        xyz = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        pts3 = self._cloud_to_base(msg, xyz.astype(np.float32))
        pts = filter_base_points(
            pts3,
            robot_radius=float(self.filter_cfg["robot_radius"]),
            max_range=float(self.filter_cfg["max_range"]),
            min_z=float(self.filter_cfg["min_z"]),
            max_z=float(self.filter_cfg["max_z"]),
            min_x=float(self.filter_cfg.get("min_x", -1.2)),
        )
        pts = voxel_downsample_2d(
            pts,
            float(self.filter_cfg["voxel"]),
            int(self.filter_cfg["max_points"]),
        )
        grid = build_local_costmap(pts, self.cfg)
        self.map_pub.publish(self._to_grid_msg(grid))
        inflated = occupied_points_from_grid(grid, self.cfg)
        self.points_pub.publish(self._to_cloud_msg(inflated))
        status = String()
        status.data = f"raw={len(xyz)} filtered={len(pts)} occupied={len(inflated)}"
        self.status_pub.publish(status)

    def _to_grid_msg(self, grid: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.cfg.resolution)
        msg.info.width = int(grid.shape[1])
        msg.info.height = int(grid.shape[0])
        ox, oy = self.cfg.origin
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        msg.data = [int(v) for v in grid.reshape(-1)]
        return msg

    def _to_cloud_msg(self, points_xy: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        if len(points_xy) == 0:
            xyz = np.zeros((0, 3), dtype=np.float32)
        else:
            xyz = np.zeros((len(points_xy), 3), dtype=np.float32)
            xyz[:, :2] = points_xy
        msg.height = 1
        msg.width = int(len(xyz))
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = xyz.astype(np.float32).tobytes()
        return msg

    def _cloud_to_base(self, msg: PointCloud2, pts3: np.ndarray) -> np.ndarray:
        if msg.header.frame_id in ("", "base_link"):
            return pts3
        try:
            tf = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"No TF {msg.header.frame_id}->base_link yet, using raw cloud: {exc}",
                throttle_duration_sec=1.0,
            )
            return pts3
        q = tf.transform.rotation
        t = tf.transform.translation
        rot = self._quat_to_rot(q.w, q.x, q.y, q.z)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (pts3 @ rot.T + trans).astype(np.float32)

    @staticmethod
    def _quat_to_rot(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)


def main():
    rclpy.init()
    node = LocalCostmapServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
