#!/usr/bin/env -S /usr/bin/python3
import math
import os
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

os.environ.update({
    'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
    'ROS_LOCALHOST_ONLY': '1',
    'ROS_DOMAIN_ID': '42',
})

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, scene_help


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


RESOLUTION = _env_float('SONIC_MAP_RESOLUTION', 0.06)
INFLATION = _env_float('SONIC_MAP_INFLATION', 0.34)
MARGIN = _env_float('SONIC_MAP_MARGIN', 0.8)
MIN_OBS_HEIGHT = _env_float('SONIC_MAP_MIN_OBS_HEIGHT', 0.12)
MAX_OBS_Z = _env_float('SONIC_MAP_MAX_OBS_Z', 1.65)


@dataclass(frozen=True)
class RasterObstacle:
    kind: str
    center: np.ndarray
    rot2: np.ndarray
    half: tuple[float, float]
    radius: float


def _robot_body_mask(model):
    skip = np.zeros(model.nbody, dtype=bool)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    if pelvis_id < 0:
        return skip
    for body_id in range(model.nbody):
        cur = body_id
        while cur != 0:
            if cur == pelvis_id:
                skip[body_id] = True
                break
            cur = int(model.body_parentid[cur])
    return skip


def _geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f'geom_{geom_id}'


def _z_span(model, data, geom_id):
    geom_type = int(model.geom_type[geom_id])
    center_z = float(data.geom_xpos[geom_id][2])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        radius_z = float(size[2])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius_z = float(size[1])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius_z = float(size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radius_z = float(size[2])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius_z = float(size[0] + size[1])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        aabb = model.geom_aabb[geom_id]
        radius_z = float(aabb[5]) if np.isfinite(aabb[5]) and aabb[5] > 0.0 else float(size[2])
    else:
        radius_z = float(np.max(size))
    return center_z - radius_z, center_z + radius_z


def _collect_obstacles(model, data, inflation):
    skip_bodies = _robot_body_mask(model)
    obstacles = []
    names = []
    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if int(model.geom_contype[geom_id]) == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if skip_bodies[body_id]:
            continue
        z_min, z_max = _z_span(model, data, geom_id)
        if z_max < MIN_OBS_HEIGHT or z_min > MAX_OBS_Z:
            continue

        center = data.geom_xpos[geom_id].astype(np.float32)
        rot = data.geom_xmat[geom_id].reshape(3, 3).astype(np.float32)
        rot2 = rot[:2, :2]
        size = model.geom_size[geom_id]
        kind = 'ellipse'
        half = (0.0, 0.0)
        radius = 0.0

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            kind = 'box'
            half = (float(size[0] + inflation), float(size[1] + inflation))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            kind = 'circle'
            radius = float(size[0] + inflation)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            kind = 'circle'
            radius = float(size[0] + inflation)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            kind = 'circle'
            radius = float(size[0] + size[1] + inflation)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            half = (float(size[0] + inflation), float(size[1] + inflation))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            aabb = model.geom_aabb[geom_id]
            local_center = aabb[:3].astype(np.float32)
            local_half = aabb[3:6].astype(np.float32)
            if np.isfinite(local_center).all() and np.isfinite(local_half).all() and np.max(local_half) > 0.0:
                center = (data.geom_xpos[geom_id] + rot @ local_center).astype(np.float32)
                kind = 'box'
                half = (float(local_half[0] + inflation), float(local_half[1] + inflation))
            else:
                radius = float(max(size[0], size[1], 0.25) + inflation)
                kind = 'circle'
        else:
            radius = float(max(size[0], size[1], 0.2) + inflation)
            kind = 'circle'

        obstacles.append(RasterObstacle(kind, center, rot2, half, radius))
        names.append(_geom_name(model, geom_id))
    return obstacles, names


def _obstacle_bounds(obstacle):
    cx, cy = float(obstacle.center[0]), float(obstacle.center[1])
    if obstacle.kind == 'circle':
        r = obstacle.radius
        return cx - r, cx + r, cy - r, cy + r
    hx, hy = obstacle.half
    corners = np.array([
        [-hx, -hy],
        [-hx, hy],
        [hx, -hy],
        [hx, hy],
    ], dtype=np.float32)
    pts = corners @ obstacle.rot2.T + obstacle.center[:2]
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def _map_bounds(model, obstacles):
    if obstacles:
        bounds = np.array([_obstacle_bounds(obs) for obs in obstacles], dtype=np.float32)
        min_x = float(bounds[:, 0].min() - MARGIN)
        max_x = float(bounds[:, 1].max() + MARGIN)
        min_y = float(bounds[:, 2].min() - MARGIN)
        max_y = float(bounds[:, 3].max() + MARGIN)
    else:
        center = np.asarray(model.stat.center, dtype=np.float32)
        extent = float(model.stat.extent)
        min_x = float(center[0] - extent)
        max_x = float(center[0] + extent)
        min_y = float(center[1] - extent)
        max_y = float(center[1] + extent)

    min_x = math.floor(min_x / RESOLUTION) * RESOLUTION
    min_y = math.floor(min_y / RESOLUTION) * RESOLUTION
    max_x = math.ceil(max_x / RESOLUTION) * RESOLUTION
    max_y = math.ceil(max_y / RESOLUTION) * RESOLUTION
    return min_x, max_x, min_y, max_y


def _mark_circle(grid, origin_x, origin_y, resolution, obstacle):
    r = obstacle.radius
    cx, cy = float(obstacle.center[0]), float(obstacle.center[1])
    ix0 = max(0, int(math.floor((cx - r - origin_x) / resolution)))
    ix1 = min(grid.shape[1] - 1, int(math.ceil((cx + r - origin_x) / resolution)))
    iy0 = max(0, int(math.floor((cy - r - origin_y) / resolution)))
    iy1 = min(grid.shape[0] - 1, int(math.ceil((cy + r - origin_y) / resolution)))
    if ix1 < ix0 or iy1 < iy0:
        return
    xs = origin_x + (np.arange(ix0, ix1 + 1, dtype=np.float32) + 0.5) * resolution
    ys = origin_y + (np.arange(iy0, iy1 + 1, dtype=np.float32) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    grid[iy0:iy1 + 1, ix0:ix1 + 1][mask] = 100


def _mark_oriented_box(grid, origin_x, origin_y, resolution, obstacle):
    hx, hy = obstacle.half
    min_x, max_x, min_y, max_y = _obstacle_bounds(obstacle)
    ix0 = max(0, int(math.floor((min_x - origin_x) / resolution)))
    ix1 = min(grid.shape[1] - 1, int(math.ceil((max_x - origin_x) / resolution)))
    iy0 = max(0, int(math.floor((min_y - origin_y) / resolution)))
    iy1 = min(grid.shape[0] - 1, int(math.ceil((max_y - origin_y) / resolution)))
    if ix1 < ix0 or iy1 < iy0:
        return
    xs = origin_x + (np.arange(ix0, ix1 + 1, dtype=np.float32) + 0.5) * resolution
    ys = origin_y + (np.arange(iy0, iy1 + 1, dtype=np.float32) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    dx = xx - float(obstacle.center[0])
    dy = yy - float(obstacle.center[1])
    local_x = dx * float(obstacle.rot2[0, 0]) + dy * float(obstacle.rot2[1, 0])
    local_y = dx * float(obstacle.rot2[0, 1]) + dy * float(obstacle.rot2[1, 1])
    mask = (np.abs(local_x) <= hx) & (np.abs(local_y) <= hy)
    grid[iy0:iy1 + 1, ix0:ix1 + 1][mask] = 100


def _build_grid(model, data):
    obstacles, names = _collect_obstacles(model, data, INFLATION)
    min_x, max_x, min_y, max_y = _map_bounds(model, obstacles)
    width = max(1, int(math.ceil((max_x - min_x) / RESOLUTION)))
    height = max(1, int(math.ceil((max_y - min_y) / RESOLUTION)))
    grid = np.zeros((height, width), dtype=np.int8)
    for obstacle in obstacles:
        if obstacle.kind == 'circle':
            _mark_circle(grid, min_x, min_y, RESOLUTION, obstacle)
        else:
            _mark_oriented_box(grid, min_x, min_y, RESOLUTION, obstacle)
    return grid, min_x, min_y, names


def _to_msg(grid, origin_x, origin_y):
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = float(RESOLUTION)
    msg.info.width = int(grid.shape[1])
    msg.info.height = int(grid.shape[0])
    msg.info.origin.position.x = float(origin_x)
    msg.info.origin.position.y = float(origin_y)
    msg.info.origin.orientation.w = 1.0
    msg.data = [int(v) for v in grid.reshape(-1)]
    return msg


class SceneMapServer(Node):
    def __init__(self, scene_arg):
        super().__init__('scene_map_server')
        try:
            self.scene = resolve_scene(scene_arg, repo_root=REPO)
        except ValueError as exc:
            raise RuntimeError(f'{exc}\n\nAvailable scenes:\n{scene_help()}') from exc

        model = mujoco.MjModel.from_xml_path(str(self.scene.abs_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        grid, origin_x, origin_y, names = _build_grid(model, data)
        self.map_msg = _to_msg(grid, origin_x, origin_y)
        self.map_msg.header.stamp = self.get_clock().now().to_msg()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, '/map', qos)
        self.timer = self.create_timer(1.0, self._publish)
        self._publish()
        self.get_logger().info(
            f'Map ready for {self.scene.name}: {grid.shape[1]}x{grid.shape[0]} '
            f'at {RESOLUTION:.2f}m, {len(names)} obstacle geoms, inflation {INFLATION:.2f}m'
        )

    def _publish(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.map_msg)


def main():
    scene_arg = sys.argv[1] if len(sys.argv) > 1 else 'default'
    rclpy.init()
    node = SceneMapServer(scene_arg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
