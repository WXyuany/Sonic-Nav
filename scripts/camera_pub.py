#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import mujoco
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "g1_ros2_nav"))

from g1_ros2_nav.tmp_io import load_npy_if_ready
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help


OPTICAL_FROM_MJ_CAMERA = np.diag([1.0, -1.0, -1.0])


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _quat_from_matrix(m: np.ndarray) -> Quaternion:
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = Quaternion(
            w=0.25 * s,
            x=(m[2, 1] - m[1, 2]) / s,
            y=(m[0, 2] - m[2, 0]) / s,
            z=(m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        q = Quaternion(
            w=(m[2, 1] - m[1, 2]) / s,
            x=0.25 * s,
            y=(m[0, 1] + m[1, 0]) / s,
            z=(m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        q = Quaternion(
            w=(m[0, 2] - m[2, 0]) / s,
            x=(m[0, 1] + m[1, 0]) / s,
            y=0.25 * s,
            z=(m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        q = Quaternion(
            w=(m[1, 0] - m[0, 1]) / s,
            x=(m[0, 2] + m[2, 0]) / s,
            y=(m[1, 2] + m[2, 1]) / s,
            z=0.25 * s,
        )

    norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if norm > 1e-9:
        q.w /= norm
        q.x /= norm
        q.y /= norm
        q.z /= norm
    return q


def _camera_info(header: Header, width: int, height: int, fovy_deg: float) -> CameraInfo:
    fovy = math.radians(max(1.0, min(179.0, fovy_deg)))
    fy = height / (2.0 * math.tan(fovy * 0.5))
    fx = fy
    cx = width * 0.5
    cy = height * 0.5

    msg = CameraInfo()
    msg.header = header
    msg.height = int(height)
    msg.width = int(width)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish MuJoCo head camera RGB/depth images to ROS2.")
    parser.add_argument("scene", nargs="?", default=os.environ.get("SONIC_CAMERA_SCENE", "scene_43dof.xml"))
    parser.add_argument("--camera", default=os.environ.get("SONIC_CAMERA_NAME", "head_camera"))
    parser.add_argument("--width", type=int, default=_env_int("SONIC_CAMERA_WIDTH", 320))
    parser.add_argument("--height", type=int, default=_env_int("SONIC_CAMERA_HEIGHT", 240))
    parser.add_argument("--fps", type=float, default=_env_float("SONIC_CAMERA_FPS", 20.0))
    parser.add_argument("--depth-fps", type=float, default=_env_float("SONIC_CAMERA_DEPTH_FPS", 20.0))
    parser.add_argument("--camera-frame", default=os.environ.get("SONIC_CAMERA_FRAME", "head_camera"))
    parser.add_argument(
        "--optical-frame",
        default=os.environ.get("SONIC_CAMERA_OPTICAL_FRAME", "camera_depth_optical_frame"),
    )
    parser.add_argument("--base-frame", default=os.environ.get("SONIC_CAMERA_BASE_FRAME", "base_link"))
    parser.add_argument("--max-depth", type=float, default=_env_float("SONIC_CAMERA_MAX_DEPTH", 6.0))
    parser.add_argument("--max-qpos-age", type=float, default=_env_float("SONIC_QPOS_MAX_AGE", 2.0))
    parser.add_argument("--no-depth", dest="publish_depth", action="store_false")
    parser.set_defaults(publish_depth=os.environ.get("SONIC_CAMERA_NO_DEPTH", "0") != "1")
    return parser.parse_args()


class CameraPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("camera")
        self.args = args
        try:
            self.scene = resolve_scene(args.scene, repo_root=REPO)
        except ValueError as exc:
            raise RuntimeError(f"{exc}\n\nAvailable scenes:\n{scene_help()}") from exc

        self.model = mujoco.MjModel.from_xml_path(str(self.scene.abs_path))
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if self.cam_id < 0:
            raise RuntimeError(f"camera '{args.camera}' not found in {self.scene.xml_file}")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.base_body_id < 0:
            raise RuntimeError("body 'pelvis' not found; cannot publish camera TF")

        self.width = max(64, int(args.width))
        self.height = max(48, int(args.height))
        self.period = 1.0 / max(1.0, float(args.fps))
        self.depth_period = 1.0 / max(1.0, float(args.depth_fps))
        self.last_depth_pub = -1e9
        self.rgb_renderer = None
        self.depth_renderer = None
        self.depth_renderer_ready = False

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", qos)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", qos)
        self.color_ci_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", qos)
        self.depth_ci_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", qos)
        self.tf_bc = TransformBroadcaster(self)
        self.timer = self.create_timer(self.period, self._publish)

        fovy = float(self.model.cam_fovy[self.cam_id])
        self.get_logger().info(
            f"Camera ready: {self.scene.xml_file}:{args.camera} "
            f"{self.width}x{self.height} rgb={1.0 / self.period:.1f}Hz "
            f"depth={1.0 / self.depth_period:.1f}Hz fovy={fovy:.1f}"
        )

    def _rgb(self):
        if self.rgb_renderer is None:
            self.rgb_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        return self.rgb_renderer

    def _depth(self):
        if self.depth_renderer is None:
            self.depth_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        if not self.depth_renderer_ready:
            self.depth_renderer.enable_depth_rendering()
            self.depth_renderer_ready = True
        return self.depth_renderer

    def _sync_qpos(self):
        qpos_path = "/tmp/sonic_qpos.npy"
        if self.args.max_qpos_age > 0.0:
            try:
                if time.time() - os.path.getmtime(qpos_path) > self.args.max_qpos_age:
                    q = None
                else:
                    q = load_npy_if_ready(qpos_path)
            except OSError:
                q = None
        else:
            q = load_npy_if_ready(qpos_path)
        if q is not None:
            n = min(len(q), self.model.nq)
            self.data.qpos[:n] = q[:n]
        mujoco.mj_forward(self.model, self.data)

    def _publish_tf(self, now):
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        rel_pos = base_rot.T @ (cam_pos - base_pos)
        rel_rot = base_rot.T @ cam_rot

        t = TransformStamped()
        t.header = Header(stamp=now, frame_id=self.args.base_frame)
        t.child_frame_id = self.args.camera_frame
        t.transform.translation.x = float(rel_pos[0])
        t.transform.translation.y = float(rel_pos[1])
        t.transform.translation.z = float(rel_pos[2])
        t.transform.rotation = _quat_from_matrix(rel_rot)
        self.tf_bc.sendTransform(t)

        optical = TransformStamped()
        optical.header = Header(stamp=now, frame_id=self.args.camera_frame)
        optical.child_frame_id = self.args.optical_frame
        optical.transform.rotation.w = 0.0
        optical.transform.rotation.x = 1.0
        self.tf_bc.sendTransform(optical)

    def _publish(self):
        try:
            self._sync_qpos()
            now = self.get_clock().now().to_msg()
            header = Header(stamp=now, frame_id=self.args.optical_frame)
            self._publish_tf(now)

            rgb_renderer = self._rgb()
            rgb_renderer.update_scene(self.data, camera=self.cam_id)
            rgb = rgb_renderer.render()
            rgb_msg = Image()
            rgb_msg.header = header
            rgb_msg.height = self.height
            rgb_msg.width = self.width
            rgb_msg.encoding = "rgb8"
            rgb_msg.is_bigendian = False
            rgb_msg.step = self.width * 3
            rgb_msg.data = rgb.tobytes()
            self.rgb_pub.publish(rgb_msg)

            fovy = float(self.model.cam_fovy[self.cam_id])
            ci = _camera_info(header, self.width, self.height, fovy)
            self.color_ci_pub.publish(ci)

            t_now = self.get_clock().now().nanoseconds * 1e-9
            if self.args.publish_depth and t_now - self.last_depth_pub >= self.depth_period:
                depth_renderer = self._depth()
                depth_renderer.update_scene(self.data, camera=self.cam_id)
                depth = depth_renderer.render().astype(np.float32, copy=False)
                if self.args.max_depth > 0.0:
                    depth = depth.copy()
                    depth[(depth <= 0.0) | (depth > self.args.max_depth)] = np.nan

                depth_msg = Image()
                depth_msg.header = header
                depth_msg.height = self.height
                depth_msg.width = self.width
                depth_msg.encoding = "32FC1"
                depth_msg.is_bigendian = False
                depth_msg.step = self.width * 4
                depth_msg.data = depth.tobytes()
                self.depth_pub.publish(depth_msg)
                self.depth_ci_pub.publish(ci)
                self.last_depth_pub = t_now
        except Exception as exc:
            self.get_logger().warn(f"camera publish skipped: {exc}")


def main():
    args = parse_args()
    rclpy.init()
    node = CameraPublisher(args)
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
