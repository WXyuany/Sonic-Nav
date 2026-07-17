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
        description="Publish the demo box as a known map/base/camera anchor instead of running vision."
    )
    parser.add_argument("scene", nargs="?", default=os.environ.get("SONIC_BOX_SCENE", "box_demo"))
    parser.add_argument("--box-name", default=os.environ.get("SONIC_BOX_NAME", "demo_box_visual"))
    parser.add_argument("--box-id", default=os.environ.get("SONIC_BOX_ID", ""))
    parser.add_argument("--camera", default=os.environ.get("SONIC_CAMERA_NAME", "head_camera"))
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_depth_optical_frame")
    parser.add_argument("--anchor-topic", default="/sonic_demo/box_anchor")
    parser.add_argument("--fallback-box-pos", type=_parse_vec3, default=_parse_vec3("1.64,0.0,0.775"))
    parser.add_argument("--fallback-box-size", type=_parse_vec3, default=_parse_vec3("0.24,0.19,0.20"))
    parser.add_argument("--image-width", type=int, default=_env_int("SONIC_CAMERA_WIDTH", 320))
    parser.add_argument("--image-height", type=int, default=_env_int("SONIC_CAMERA_HEIGHT", 240))
    parser.add_argument("--max-qpos-age", type=float, default=2.0)
    parser.add_argument("--walk-speed", type=float, default=0.26)
    parser.add_argument("--approach-standoff", type=float, default=0.45)
    parser.add_argument("--wrist-forward-offset", type=float, default=0.02)
    parser.add_argument("--open-margin", type=float, default=0.085)
    parser.add_argument("--clamp-margin", type=float, default=0.006)
    parser.add_argument("--forearm-overlap", type=float, default=0.0)
    parser.add_argument("--visual-grasp-z-offset", type=float, default=0.000)
    parser.add_argument("--wrist-height-offset", type=float, default=0.0)
    parser.add_argument("--min-walk-duration", type=float, default=0.6)
    parser.add_argument("--max-walk-duration", type=float, default=8.0)
    return parser.parse_args()


class BoxAnchorPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("box_anchor_pub")
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
        self.box_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, args.box_name)
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, args.box_name)
        if self.base_body_id < 0:
            raise RuntimeError("body 'pelvis' not found; cannot compute box pose in base_link")
        if self.camera_id < 0:
            self.get_logger().warn(f"camera '{args.camera}' not found; camera-depth anchor will use NaNs")
        if self.box_site_id < 0 and self.box_geom_id < 0:
            self.get_logger().warn(
                f"box '{args.box_name}' not found in {self.scene.xml_file}; using fallback box pose"
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
        self.pose_pub = self.create_publisher(PoseStamped, "/sonic_demo/box_pose", latched_qos)
        self.grasp_pub = self.create_publisher(PoseStamped, "/sonic_demo/box_grasp_base_pose", latched_qos)
        self.marker_pub = self.create_publisher(Marker, "/sonic_demo/box_marker", latched_qos)
        self.base_point_pub = self.create_publisher(PointStamped, "/sonic_demo/box_point_base", live_qos)
        self.camera_point_pub = self.create_publisher(PointStamped, "/sonic_demo/box_point_camera", live_qos)
        self.timer = self.create_timer(1.0 / max(1.0, float(args.rate)), self._publish)
        self._reported = False
        self.get_logger().info(
            f"Box anchor ready: scene={self.scene.xml_file} box={args.box_name} topic={args.anchor_topic}"
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

    def _box_pose_and_size(self) -> tuple[np.ndarray, np.ndarray, str]:
        if self.box_site_id >= 0:
            center = self.data.site_xpos[self.box_site_id].astype(np.float64).copy()
            size = (2.0 * self.model.site_size[self.box_site_id]).astype(np.float64).copy()
            size = np.maximum(size, np.asarray([0.05, 0.05, 0.05], dtype=np.float64))
            return center, size, "site"
        if self.box_geom_id >= 0:
            center = self.data.geom_xpos[self.box_geom_id].astype(np.float64).copy()
            size = (2.0 * self.model.geom_size[self.box_geom_id]).astype(np.float64).copy()
            size = np.maximum(size, np.asarray([0.05, 0.05, 0.05], dtype=np.float64))
            return center, size, "geom"
        return self.args.fallback_box_pos.copy(), self.args.fallback_box_size.copy(), "fallback"

    def _camera_point(self, box_center: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
        if self.camera_id < 0:
            return np.asarray([math.nan, math.nan, math.nan]), {"u": None, "v": None, "depth": None}

        cam_pos = self.data.cam_xpos[self.camera_id]
        cam_rot = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        point_mj = cam_rot.T @ (box_center - cam_pos)
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

    def _make_grasp_plan(self, box_base: np.ndarray, box_map: np.ndarray, box_size: np.ndarray) -> dict:
        forward = max(0.0, float(box_base[0]))
        lateral = float(box_base[1])
        vertical = float(box_base[2])
        standoff = max(0.10, float(self.args.approach_standoff))
        walk_speed = max(0.05, float(self.args.walk_speed))
        walk_distance = max(0.0, forward - standoff)
        walk_duration = _clamp(
            walk_distance / walk_speed,
            float(self.args.min_walk_duration),
            float(self.args.max_walk_duration),
        )

        half_y = max(0.03, float(box_size[1]) * 0.5)
        half_z = max(0.03, float(box_size[2]) * 0.5)
        reach_x = _clamp(forward + float(self.args.wrist_forward_offset), 0.32, 0.62)
        clamp_y = _clamp(half_y + float(self.args.clamp_margin), 0.08, 0.16)
        open_y = _clamp(half_y + float(self.args.open_margin), clamp_y + 0.08, 0.34)
        reach_z = _clamp(vertical, -0.12, 0.08)
        visual_z = float(self.args.visual_grasp_z_offset)
        clamp_visual_z = _clamp(reach_z + visual_z, -0.04, 0.12)
        lift_z = _clamp(clamp_visual_z + min(0.10, half_z * 0.70), 0.02, 0.16)
        clear_z = _clamp(lift_z - 0.01, -0.03, 0.10)
        carry_x = _clamp(reach_x - 0.18, 0.25, 0.34)
        carry_z = _clamp(lift_z - 0.02, 0.02, 0.12)
        clamp_assist_x = _clamp(forward, 0.34, 0.55)
        clamp_assist_z = clamp_visual_z
        clear_assist_x = _clamp(forward - 0.03, 0.32, 0.52)
        clear_assist_z = _clamp(lift_z, -0.02, 0.12)
        assist_x = _clamp(carry_x, 0.25, 0.34)
        assist_z = _clamp(carry_z, -0.04, 0.08)
        wrist_target_y = _clamp(lateral, -0.12, 0.12)

        base_pos = self.data.xpos[self.base_body_id]
        yaw_to_box = math.atan2(float(box_map[1] - base_pos[1]), float(box_map[0] - base_pos[0]))
        base_target_x = float(box_map[0] - standoff * math.cos(yaw_to_box))
        base_target_y = float(box_map[1] - standoff * math.sin(yaw_to_box))
        return {
            "walk_speed": walk_speed,
            "walk_duration": float(walk_duration),
            "approach_target_x": float(standoff),
            "reach_x": float(reach_x),
            "open_y": float(open_y),
            "clamp_y": float(clamp_y),
            "reach_z": float(reach_z),
            "clear_z": float(clear_z),
            "lift_z": float(lift_z),
            "target_y": float(wrist_target_y),
            "carry_x": float(carry_x),
            "carry_z": float(carry_z),
            "assist_x": float(assist_x),
            "assist_z": float(assist_z),
            "clamp_assist_x": float(clamp_assist_x),
            "clamp_assist_z": float(clamp_assist_z),
            "clear_assist_x": float(clear_assist_x),
            "clear_assist_z": float(clear_assist_z),
            "box_half_y": float(half_y),
            "left_edge_y": float(lateral + half_y),
            "right_edge_y": float(lateral - half_y),
            "left_clamp_y": float(wrist_target_y + clamp_y),
            "right_clamp_y": float(wrist_target_y - clamp_y),
            "approach_standoff": float(standoff),
            "visual_grasp_z_offset": float(visual_z),
            "lateral_error": lateral,
            "base_target_map": [base_target_x, base_target_y, float(yaw_to_box)],
        }

    def _publish(self):
        self._sync_qpos()
        now = self.get_clock().now().to_msg()
        header = Header(stamp=now, frame_id=self.args.map_frame)
        box_map, box_size, source = self._box_pose_and_size()

        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        box_base = base_rot.T @ (box_map - base_pos)
        box_camera, pixel = self._camera_point(box_map)
        grasp = self._make_grasp_plan(box_base, box_map, box_size)

        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(box_map[0])
        pose.pose.position.y = float(box_map[1])
        pose.pose.position.z = float(box_map[2])
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        base_point = PointStamped()
        base_point.header = Header(stamp=now, frame_id=self.args.base_frame)
        base_point.point.x = float(box_base[0])
        base_point.point.y = float(box_base[1])
        base_point.point.z = float(box_base[2])
        self.base_point_pub.publish(base_point)

        camera_point = PointStamped()
        camera_point.header = Header(stamp=now, frame_id=self.args.camera_frame)
        camera_point.point.x = float(box_camera[0])
        camera_point.point.y = float(box_camera[1])
        camera_point.point.z = float(box_camera[2])
        self.camera_point_pub.publish(camera_point)

        grasp_pose = PoseStamped()
        grasp_pose.header = header
        grasp_pose.pose.position.x = float(grasp["base_target_map"][0])
        grasp_pose.pose.position.y = float(grasp["base_target_map"][1])
        _yaw_quat(grasp_pose.pose, float(grasp["base_target_map"][2]))
        self.grasp_pub.publish(grasp_pose)

        marker = Marker()
        marker.header = header
        marker.ns = "sonic_demo"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = float(box_size[0])
        marker.scale.y = float(box_size[1])
        marker.scale.z = float(box_size[2])
        _marker_color(marker, (0.95, 0.55, 0.20, 0.75))
        self.marker_pub.publish(marker)

        anchor = {
            "stamp": {"sec": int(now.sec), "nanosec": int(now.nanosec)},
            "scene": self.scene.name,
            "box_name": self.args.box_id or self.args.box_name,
            "source": source,
            "frame_id": self.args.map_frame,
            "box_center_map": _as_float_list(box_map),
            "box_size": _as_float_list(box_size),
            "box_point_base": _as_float_list(box_base),
            "box_point_camera_depth": _as_float_list(box_camera),
            "box_pixel": pixel,
            "grasp": grasp,
        }
        msg = String()
        msg.data = json.dumps(anchor, separators=(",", ":"))
        self.anchor_pub.publish(msg)

        if not self._reported:
            self.get_logger().info(
                "box anchor: "
                f"map={anchor['box_center_map']} base={anchor['box_point_base']} "
                f"camera={anchor['box_point_camera_depth']} walk={grasp['walk_duration']:.2f}s "
                f"reach_x={grasp['reach_x']:.2f}"
            )
            self._reported = True


def main():
    args = parse_args()
    rclpy.init()
    node = BoxAnchorPublisher(args)
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
