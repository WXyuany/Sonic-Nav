#!/usr/bin/env -S /usr/bin/python3
import heapq
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

os.environ.update({
    'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
    'ROS_LOCALHOST_ONLY': '1',
    'ROS_DOMAIN_ID': '42',
})


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


OCC_THRESHOLD = int(_env_float('SONIC_ASTAR_OCC_THRESHOLD', 50))
PLAN_PERIOD = _env_float('SONIC_ASTAR_PERIOD', 0.7)
MIN_REPLAN_DIST = _env_float('SONIC_ASTAR_REPLAN_DIST', 0.25)
LOOKUP_RADIUS_M = _env_float('SONIC_ASTAR_FREE_LOOKUP_RADIUS', 0.8)
PATH_SPACING = _env_float('SONIC_ASTAR_PATH_SPACING', 0.22)
MAX_EXPANSIONS = int(_env_float('SONIC_ASTAR_MAX_EXPANSIONS', 300000))


class AStarGlobalPlanner(Node):
    def __init__(self):
        super().__init__('astar_global_planner')
        qos_map = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.plan_alias_pub = self.create_publisher(Path, '/plan', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos_map)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.timer = self.create_timer(PLAN_PERIOD, self.tick)

        self.grid = None
        self.resolution = 0.05
        self.origin = (0.0, 0.0)
        self.width = 0
        self.height = 0
        self.odom = None
        self.goal = None
        self.last_plan_start = None
        self.last_goal = None
        self.last_path = None
        self.get_logger().info('A* global planner ready. Waiting for /map, /odom, and /goal_pose.')

    def on_map(self, msg):
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.grid = data
        self.resolution = float(msg.info.resolution)
        self.origin = (float(msg.info.origin.position.x), float(msg.info.origin.position.y))
        self.width = int(msg.info.width)
        self.height = int(msg.info.height)
        self.get_logger().info(
            f'Map received: {self.width}x{self.height} at {self.resolution:.2f}m',
            throttle_duration_sec=5.0,
        )

    def on_odom(self, msg):
        self.odom = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            self._yaw_from_quat(msg.pose.pose.orientation),
        )

    def on_goal(self, msg):
        self.goal = self._goal_to_odom(msg)
        self.last_goal = None
        self.get_logger().info(f'Global goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
        self.tick(force=True)

    def tick(self, force=False):
        if self.grid is None or self.odom is None or self.goal is None:
            return
        start_xy = np.array([self.odom[0], self.odom[1]], dtype=np.float32)
        goal_xy = np.array(self.goal, dtype=np.float32)
        if (
            not force
            and self.last_plan_start is not None
            and self.last_goal is not None
            and np.linalg.norm(start_xy - self.last_plan_start) < MIN_REPLAN_DIST
            and np.linalg.norm(goal_xy - self.last_goal) < 1e-3
        ):
            if self.last_path is not None:
                self._publish_path(self.last_path)
            return

        start = self._nearest_free(self._world_to_grid(float(start_xy[0]), float(start_xy[1])))
        goal = self._nearest_free(self._world_to_grid(float(goal_xy[0]), float(goal_xy[1])))
        if start is None:
            self.get_logger().warn('Start is outside the map or cannot be projected to free space')
            self._publish_empty_path()
            return
        if goal is None:
            self.get_logger().warn('Goal is outside the map or cannot be projected to free space')
            self._publish_empty_path()
            return

        cells = self._astar(start, goal)
        if not cells:
            self.get_logger().warn(
                f'No global path from {start} to {goal}; try reducing SONIC_MAP_INFLATION'
            )
            self._publish_empty_path()
            return

        cells = self._smooth_cells(cells)
        points = self._densify([self._grid_to_world(ix, iy) for ix, iy in cells])
        self.last_plan_start = start_xy
        self.last_goal = goal_xy
        self.last_path = points
        self._publish_path(points)
        length = self._path_length(points)
        self.get_logger().info(f'Global path: {len(points)} poses, {length:.2f}m')

    def _astar(self, start, goal):
        sx, sy = start
        gx, gy = goal
        total = self.width * self.height
        inf = np.float32(np.inf)
        g_cost = np.full(total, inf, dtype=np.float32)
        parent = np.full(total, -1, dtype=np.int32)
        closed = np.zeros(total, dtype=bool)
        start_i = sy * self.width + sx
        goal_i = gy * self.width + gx
        g_cost[start_i] = 0.0
        heap = [(self._heuristic(sx, sy, gx, gy), 0.0, sx, sy)]
        expansions = 0
        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )

        while heap and expansions < MAX_EXPANSIONS:
            _, cur_g, x, y = heapq.heappop(heap)
            idx = y * self.width + x
            if closed[idx]:
                continue
            closed[idx] = True
            expansions += 1
            if idx == goal_i:
                return self._reconstruct(parent, start_i, goal_i)

            for dx, dy, step_cost in neighbors:
                nx = x + dx
                ny = y + dy
                if not self._is_free(nx, ny):
                    continue
                if dx != 0 and dy != 0 and (not self._is_free(x + dx, y) or not self._is_free(x, y + dy)):
                    continue
                nidx = ny * self.width + nx
                if closed[nidx]:
                    continue
                ng = cur_g + step_cost
                if ng < float(g_cost[nidx]):
                    g_cost[nidx] = ng
                    parent[nidx] = idx
                    f = ng + self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(heap, (f, ng, nx, ny))
        return []

    def _reconstruct(self, parent, start_i, goal_i):
        cells = []
        idx = goal_i
        while idx >= 0:
            x = idx % self.width
            y = idx // self.width
            cells.append((x, y))
            if idx == start_i:
                break
            idx = int(parent[idx])
        if not cells or cells[-1] != (start_i % self.width, start_i // self.width):
            return []
        cells.reverse()
        return cells

    def _smooth_cells(self, cells):
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            nxt = len(cells) - 1
            while nxt > anchor + 1 and not self._line_free(cells[anchor], cells[nxt]):
                nxt -= 1
            out.append(cells[nxt])
            anchor = nxt
        return out

    def _line_free(self, a, b):
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if not self._is_free(x, y):
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _densify(self, points):
        if len(points) <= 1:
            return points
        dense = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            ax, ay = a
            bx, by = b
            dist = math.hypot(bx - ax, by - ay)
            steps = max(1, int(math.ceil(dist / PATH_SPACING)))
            for i in range(1, steps + 1):
                t = i / steps
                dense.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        return dense

    def _nearest_free(self, cell):
        if cell is None:
            return None
        if self._is_free(cell[0], cell[1]):
            return cell
        max_cells = max(1, int(math.ceil(LOOKUP_RADIUS_M / self.resolution)))
        cx, cy = cell
        best = None
        best_d2 = float('inf')
        for r in range(1, max_cells + 1):
            for y in range(cy - r, cy + r + 1):
                for x in (cx - r, cx + r):
                    if self._is_free(x, y):
                        d2 = (x - cx) ** 2 + (y - cy) ** 2
                        if d2 < best_d2:
                            best = (x, y)
                            best_d2 = d2
            for x in range(cx - r + 1, cx + r):
                for y in (cy - r, cy + r):
                    if self._is_free(x, y):
                        d2 = (x - cx) ** 2 + (y - cy) ** 2
                        if d2 < best_d2:
                            best = (x, y)
                            best_d2 = d2
            if best is not None:
                return best
        return None

    def _is_free(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        value = int(self.grid[y, x])
        return 0 <= value <= OCC_THRESHOLD

    def _world_to_grid(self, x, y):
        ix = int(math.floor((x - self.origin[0]) / self.resolution))
        iy = int(math.floor((y - self.origin[1]) / self.resolution))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return None
        return ix, iy

    def _grid_to_world(self, ix, iy):
        return (
            self.origin[0] + (ix + 0.5) * self.resolution,
            self.origin[1] + (iy + 0.5) * self.resolution,
        )

    def _goal_to_odom(self, msg):
        frame = msg.header.frame_id or 'odom'
        if frame == 'odom':
            return (float(msg.pose.position.x), float(msg.pose.position.y))
        try:
            tf = self.tf_buffer.lookup_transform('odom', frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f'No TF {frame}->odom for goal, using raw goal: {exc}')
            return (float(msg.pose.position.x), float(msg.pose.position.y))
        q = tf.transform.rotation
        t = tf.transform.translation
        rot = self._quat_to_rot(q.w, q.x, q.y, q.z)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        out = p @ rot.T + np.array([t.x, t.y, t.z], dtype=np.float32)
        return (float(out[0]), float(out[1]))

    def _publish_path(self, points):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        if not points:
            self.plan_pub.publish(path)
            self.plan_alias_pub.publish(path)
            return
        for i, point in enumerate(points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            if i + 1 < len(points):
                yaw = math.atan2(points[i + 1][1] - point[1], points[i + 1][0] - point[0])
            elif len(points) > 1:
                yaw = math.atan2(point[1] - points[i - 1][1], point[0] - points[i - 1][0])
            else:
                yaw = self.odom[2] if self.odom is not None else 0.0
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            path.poses.append(pose)
        self.plan_pub.publish(path)
        self.plan_alias_pub.publish(path)

    def _publish_empty_path(self):
        self.last_path = []
        self._publish_path([])

    @staticmethod
    def _path_length(points):
        if len(points) < 2:
            return 0.0
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:]))

    @staticmethod
    def _heuristic(x, y, gx, gy):
        return math.hypot(gx - x, gy - y)

    @staticmethod
    def _yaw_from_quat(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _quat_to_rot(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)


def main():
    rclpy.init()
    node = AStarGlobalPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
