#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import mujoco
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "g1_ros2_nav"))

from g1_ros2_nav.tmp_io import load_npy_if_ready
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help


OPTICAL_FROM_MJ_CAMERA = np.diag([1.0, -1.0, -1.0])


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 numbers, got {text!r}")
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _as_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _yaw_quat(pose, yaw: float):
    pose.orientation.w = math.cos(yaw * 0.5)
    pose.orientation.z = math.sin(yaw * 0.5)


def _marker_color(marker: Marker, rgba: tuple[float, float, float, float]):
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish the demo ball and tabletop place target as known map/base/camera anchors."
    )
    parser.add_argument("scene", nargs="?", default=os.environ.get("SONIC_BALL_SCENE", "ball_demo"))
    parser.add_argument("--ball-name", default=os.environ.get("SONIC_BALL_NAME", "demo_ball_visual"))
    parser.add_argument("--ball-id", default=os.environ.get("SONIC_BALL_ID", ""))
    parser.add_argument("--place-site", default=os.environ.get("SONIC_BALL_PLACE_SITE", "demo_ball_place_target"))
    parser.add_argument("--place-id", default=os.environ.get("SONIC_BALL_PLACE_ID", "place_target"))
    parser.add_argument("--camera", default=os.environ.get("SONIC_CAMERA_NAME", "head_camera"))
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_depth_optical_frame")
    parser.add_argument("--anchor-topic", default="/sonic_demo/ball_anchor")
    parser.add_argument("--fallback-ball-pos", type=_parse_vec3, default=_parse_vec3("1.62,-0.28,0.840"))
    parser.add_argument("--fallback-ball-radius", type=float, default=0.045)
    parser.add_argument("--fallback-place-pos", type=_parse_vec3, default=_parse_vec3("1.62,-0.08,0.840"))
    parser.add_argument("--place-offset-y", type=float, default=0.14)
    parser.add_argument("--dynamic-place-target", action="store_true")
    parser.add_argument("--dynamic-place-delta-y", type=float, default=0.20)
    parser.add_argument("--image-width", type=int, default=_env_int("SONIC_CAMERA_WIDTH", 320))
    parser.add_argument("--image-height", type=int, default=_env_int("SONIC_CAMERA_HEIGHT", 240))
    parser.add_argument("--max-qpos-age", type=float, default=2.0)
    parser.add_argument("--walk-speed", type=float, default=0.24)
    parser.add_argument("--approach-standoff", type=float, default=0.56)
    parser.add_argument("--min-walk-duration", type=float, default=0.6)
    parser.add_argument("--max-walk-duration", type=float, default=10.0)
    return parser.parse_args()


class BallAnchorPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ball_anchor_pub")
        self.args = args
        try:
            self.scene = resolve_scene(args.scene, repo_root=REPO)
        except ValueError as exc:
            raise RuntimeError(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc

        self.model = mujoco.MjModel.from_xml_path(str(self.scene.abs_path))
        self.data = mujoco.MjData(self.model)
        self._started_at = time.time()
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        self.ball_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, args.ball_name)
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, args.ball_name)
        self.place_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, args.place_site)
        if self.base_body_id < 0:
            raise RuntimeError("body 'pelvis' not found; cannot compute ball pose in base_link")
        if self.camera_id < 0:
            self.get_logger().warn(f"camera '{args.camera}' not found; camera-depth anchor will use NaNs")
        if self.ball_site_id < 0 and self.ball_geom_id < 0:
            self.get_logger().warn(
                f"ball '{args.ball_name}' not found in {self.scene.xml_file}; using fallback ball pose"
            )

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        live_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.anchor_pub = self.create_publisher(String, args.anchor_topic, latched_qos)
        self.pose_pub = self.create_publisher(PoseStamped, "/sonic_demo/ball_pose", latched_qos)
        self.place_pub = self.create_publisher(PoseStamped, "/sonic_demo/ball_place_pose", latched_qos)
        self.pick_pub = self.create_publisher(PoseStamped, "/sonic_demo/ball_pick_base_pose", latched_qos)
        self.marker_pub = self.create_publisher(Marker, "/sonic_demo/ball_marker", latched_qos)
        self.place_marker_pub = self.create_publisher(Marker, "/sonic_demo/ball_place_marker", latched_qos)
        self.base_point_pub = self.create_publisher(PointStamped, "/sonic_demo/ball_point_base", live_qos)
        self.place_point_pub = self.create_publisher(PointStamped, "/sonic_demo/ball_place_point_base", live_qos)
        self.camera_point_pub = self.create_publisher(PointStamped, "/sonic_demo/ball_point_camera", live_qos)
        self.timer = self.create_timer(1.0 / max(1.0, float(args.rate)), self._publish)
        self._place_map: np.ndarray | None = None
        self._reported = False
        self.get_logger().info(
            f"Ball anchor ready: scene={self.scene.xml_file} ball={args.ball_name} topic={args.anchor_topic}"
        )

    def _sync_qpos(self):
        qpos_path = "/tmp/sonic_qpos.npy"
        if self.args.max_qpos_age > 0.0:
            try:
                mtime = os.path.getmtime(qpos_path)
                if mtime < self._started_at - 0.2 or time.time() - mtime > self.args.max_qpos_age:
                    q = None
                else:
                    q = load_npy_if_ready(qpos_path)
            except OSError:
                q = None
        else:
            q = load_npy_if_ready(qpos_path)
        if q is not None and len(q) == self.model.nq:
            self.data.qpos[:] = q[:]
        mujoco.mj_forward(self.model, self.data)

    def _ball_pose_and_radius(self) -> tuple[np.ndarray, float, str]:
        if self.ball_site_id >= 0:
            center = self.data.site_xpos[self.ball_site_id].astype(np.float64).copy()
            radius = float(np.max(self.model.site_size[self.ball_site_id]))
            return center, max(0.02, radius), "site"
        if self.ball_geom_id >= 0:
            center = self.data.geom_xpos[self.ball_geom_id].astype(np.float64).copy()
            radius = float(self.model.geom_size[self.ball_geom_id][0])
            return center, max(0.02, radius), "geom"
        return self.args.fallback_ball_pos.copy(), float(self.args.fallback_ball_radius), "fallback"

    def _place_pose(self, ball_map: np.ndarray) -> np.ndarray:
        if self.args.dynamic_place_target:
            delta_y = abs(float(self.args.dynamic_place_delta_y))
            reference_y = float(self.args.fallback_place_pos[1])
            if self.place_site_id >= 0:
                reference_y = float(self.data.site_xpos[self.place_site_id][1])
            place = ball_map.copy()
            place[1] = float(ball_map[1] + (delta_y if ball_map[1] < reference_y else -delta_y))
            return place
        if self._place_map is not None:
            return self._place_map.copy()
        if self.place_site_id >= 0:
            place = self.data.site_xpos[self.place_site_id].astype(np.float64).copy()
            place[2] = float(ball_map[2])
        elif np.all(np.isfinite(self.args.fallback_place_pos)):
            place = self.args.fallback_place_pos.copy()
        else:
            place = ball_map + np.asarray([0.0, float(self.args.place_offset_y), 0.0], dtype=np.float64)
        self._place_map = place
        return place.copy()

    def _camera_point(self, point_map: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
        if self.camera_id < 0:
            return np.asarray([math.nan, math.nan, math.nan]), {"u": None, "v": None, "depth": None}

        cam_pos = self.data.cam_xpos[self.camera_id]
        cam_rot = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        point_mj = cam_rot.T @ (point_map - cam_pos)
        point_optical = OPTICAL_FROM_MJ_CAMERA.T @ point_mj
        depth = float(point_optical[2])
        pixel = {"u": None, "v": None, "depth": depth if math.isfinite(depth) else None}
        if depth > 1e-6:
            fovy = math.radians(float(self.model.cam_fovy[self.camera_id]))
            fy = self.args.image_height / (2.0 * math.tan(fovy * 0.5))
            fx = fy
            pixel["u"] = float(fx * point_optical[0] / depth + self.args.image_width * 0.5)
            pixel["v"] = float(fy * point_optical[1] / depth + self.args.image_height * 0.5)
        return point_optical.astype(np.float64), pixel

    def _make_pick_plan(
        self,
        ball_base: np.ndarray,
        ball_map: np.ndarray,
        place_map: np.ndarray,
        place_base: np.ndarray,
        radius: float,
    ) -> dict:
        forward = max(0.0, float(ball_base[0]))
        standoff = max(0.10, float(self.args.approach_standoff))
        walk_speed = max(0.05, float(self.args.walk_speed))
        walk_distance = max(0.0, forward - standoff)
        walk_duration = _clamp(
            walk_distance / walk_speed,
            float(self.args.min_walk_duration),
            float(self.args.max_walk_duration),
        )
        base_pos = self.data.xpos[self.base_body_id]
        yaw_to_ball = math.atan2(float(ball_map[1] - base_pos[1]), float(ball_map[0] - base_pos[0]))
        base_target_x = float(ball_map[0] - standoff * math.cos(yaw_to_ball))
        base_target_y = float(ball_map[1] - standoff * math.sin(yaw_to_ball))
        return {
            "walk_speed": float(walk_speed),
            "walk_duration": float(walk_duration),
            "approach_target_x": float(standoff),
            "ball_radius": float(radius),
            "target_y": float(ball_base[1]),
            "reach_x": float(_clamp(ball_base[0], 0.34, 0.58)),
            "reach_y": float(_clamp(ball_base[1], -0.30, 0.22)),
            "reach_z": float(_clamp(ball_base[2], -0.08, 0.12)),
            "place_y": float(_clamp(place_base[1], -0.18, 0.28)),
            "place_z": float(_clamp(place_base[2], -0.08, 0.16)),
            "base_target_map": [base_target_x, base_target_y, float(yaw_to_ball)],
            "place_center_map": _as_float_list(place_map),
            "place_point_base": _as_float_list(place_base),
            "place_name": self.args.place_id,
        }

    def _publish(self):
        self._sync_qpos()
        now = self.get_clock().now().to_msg()
        header = Header(stamp=now, frame_id=self.args.map_frame)
        ball_map, radius, source = self._ball_pose_and_radius()
        place_map = self._place_pose(ball_map)

        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        ball_base = base_rot.T @ (ball_map - base_pos)
        place_base = base_rot.T @ (place_map - base_pos)
        ball_camera, pixel = self._camera_point(ball_map)
        grasp = self._make_pick_plan(ball_base, ball_map, place_map, place_base, radius)

        ball_pose = PoseStamped()
        ball_pose.header = header
        ball_pose.pose.position.x = float(ball_map[0])
        ball_pose.pose.position.y = float(ball_map[1])
        ball_pose.pose.position.z = float(ball_map[2])
        ball_pose.pose.orientation.w = 1.0
        self.pose_pub.publish(ball_pose)

        place_pose = PoseStamped()
        place_pose.header = header
        place_pose.pose.position.x = float(place_map[0])
        place_pose.pose.position.y = float(place_map[1])
        place_pose.pose.position.z = float(place_map[2])
        place_pose.pose.orientation.w = 1.0
        self.place_pub.publish(place_pose)

        base_point = PointStamped()
        base_point.header = Header(stamp=now, frame_id=self.args.base_frame)
        base_point.point.x = float(ball_base[0])
        base_point.point.y = float(ball_base[1])
        base_point.point.z = float(ball_base[2])
        self.base_point_pub.publish(base_point)

        place_point = PointStamped()
        place_point.header = Header(stamp=now, frame_id=self.args.base_frame)
        place_point.point.x = float(place_base[0])
        place_point.point.y = float(place_base[1])
        place_point.point.z = float(place_base[2])
        self.place_point_pub.publish(place_point)

        camera_point = PointStamped()
        camera_point.header = Header(stamp=now, frame_id=self.args.camera_frame)
        camera_point.point.x = float(ball_camera[0])
        camera_point.point.y = float(ball_camera[1])
        camera_point.point.z = float(ball_camera[2])
        self.camera_point_pub.publish(camera_point)

        pick_pose = PoseStamped()
        pick_pose.header = header
        pick_pose.pose.position.x = float(grasp["base_target_map"][0])
        pick_pose.pose.position.y = float(grasp["base_target_map"][1])
        _yaw_quat(pick_pose.pose, float(grasp["base_target_map"][2]))
        self.pick_pub.publish(pick_pose)

        marker = Marker()
        marker.header = header
        marker.ns = "sonic_demo_ball"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = ball_pose.pose
        marker.scale.x = marker.scale.y = marker.scale.z = float(2.0 * radius)
        _marker_color(marker, (0.15, 0.55, 0.95, 0.82))
        self.marker_pub.publish(marker)

        place_marker = Marker()
        place_marker.header = header
        place_marker.ns = "sonic_demo_ball"
        place_marker.id = 2
        place_marker.type = Marker.CYLINDER
        place_marker.action = Marker.ADD
        place_marker.pose = place_pose.pose
        place_marker.scale.x = place_marker.scale.y = float(2.8 * radius)
        place_marker.scale.z = 0.010
        _marker_color(place_marker, (0.15, 0.90, 0.35, 0.35))
        self.place_marker_pub.publish(place_marker)

        anchor = {
            "stamp": {"sec": int(now.sec), "nanosec": int(now.nanosec)},
            "scene": self.scene.name,
            "ball_name": self.args.ball_id or self.args.ball_name,
            "source": source,
            "frame_id": self.args.map_frame,
            "ball_center_map": _as_float_list(ball_map),
            "ball_radius": float(radius),
            "ball_size": [float(2.0 * radius)] * 3,
            "ball_point_base": _as_float_list(ball_base),
            "ball_point_camera_depth": _as_float_list(ball_camera),
            "ball_pixel": pixel,
            "place_center_map": _as_float_list(place_map),
            "place_point_base": _as_float_list(place_base),
            "grasp": grasp,
        }
        msg = String()
        msg.data = json.dumps(anchor, separators=(",", ":"))
        self.anchor_pub.publish(msg)

        if not self._reported:
            self.get_logger().info(
                "ball anchor: "
                f"map={anchor['ball_center_map']} base={anchor['ball_point_base']} "
                f"place_base={anchor['place_point_base']} walk={grasp['walk_duration']:.2f}s"
            )
            self._reported = True


def main():
    args = parse_args()
    rclpy.init()
    node = BallAnchorPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
