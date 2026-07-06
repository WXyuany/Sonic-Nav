#!/usr/bin/env -S /usr/bin/python3
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PolygonStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.nav.params import load_config, overlay_env_scalars

os.environ.update({'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp', 'ROS_LOCALHOST_ONLY': '1', 'ROS_DOMAIN_ID': '42'})

DWA_DEFAULTS = {
    'controller': {'max_v': 0.60, 'max_w': 0.80, 'max_dv': 0.060, 'max_dw': 0.095,
                   'dt': 0.12, 'horizon': 22, 'v_deadband': 0.035, 'w_deadband': 0.040},
    'robot': {'radius': 0.42, 'safety_radius': 0.72, 'stop_radius': 0.48},
    'goal': {'tolerance': 0.45, 'final_slow_radius': 0.95},
    'heading': {'turn_in_place_error': 1.25, 'spot_turn_error': 1.55,
                'forward_full_error': 0.45, 'min_curve_speed': 0.10},
    'detour': {'trigger_distance': 1.55, 'clear_distance': 2.20, 'lookahead': 1.45, 'offset': 1.15},
    'global_plan': {'lookahead': 1.10, 'max_deviation': 2.0},
    'obstacles': {'max_range': 7.0, 'min_z': -0.55, 'max_z': 0.85, 'voxel': 0.10, 'max_points': 900},
    'topics': {'cmd_vel': '/cmd_vel_nav', 'local_costmap': '/local_costmap',
               'local_plan': '/local_plan', 'debug_prefix': '/sonic_nav/dwa'},
}

ENV_OVERRIDES = {
    'SONIC_DWA_MAX_V': ('controller.max_v', float),
    'SONIC_DWA_MAX_W': ('controller.max_w', float),
    'SONIC_DWA_MAX_DV': ('controller.max_dv', float),
    'SONIC_DWA_MAX_DW': ('controller.max_dw', float),
    'SONIC_DWA_ROBOT_RADIUS': ('robot.radius', float),
    'SONIC_DWA_SAFETY_RADIUS': ('robot.safety_radius', float),
    'SONIC_DWA_STOP_RADIUS': ('robot.stop_radius', float),
    'SONIC_DWA_GOAL_TOL': ('goal.tolerance', float),
    'SONIC_DWA_FINAL_SLOW_RADIUS': ('goal.final_slow_radius', float),
    'SONIC_DWA_TURN_ERR': ('heading.turn_in_place_error', float),
    'SONIC_DWA_SPOT_TURN_ERR': ('heading.spot_turn_error', float),
    'SONIC_DWA_FORWARD_FULL_ERR': ('heading.forward_full_error', float),
    'SONIC_DWA_MIN_CURVE_SPEED': ('heading.min_curve_speed', float),
}

DWA_CFG = overlay_env_scalars(load_config('dwa', DWA_DEFAULTS, 'SONIC_DWA_CONFIG'), ENV_OVERRIDES)
CTRL = DWA_CFG['controller']
ROBOT = DWA_CFG['robot']
GOAL = DWA_CFG['goal']
HEADING = DWA_CFG['heading']
DETOUR = DWA_CFG['detour']
GLOBAL = DWA_CFG['global_plan']
OBS = DWA_CFG['obstacles']
TOPICS = DWA_CFG['topics']

MAX_V = float(CTRL['max_v'])
MAX_W = float(CTRL['max_w'])
MAX_DV = float(CTRL['max_dv'])
MAX_DW = float(CTRL['max_dw'])
DT = float(CTRL['dt'])
HORIZON = int(CTRL['horizon'])
ROBOT_RADIUS = float(ROBOT['radius'])
SAFETY_RADIUS = float(ROBOT['safety_radius'])
STOP_RADIUS = float(ROBOT['stop_radius'])
GOAL_TOL = float(GOAL['tolerance'])
TURN_IN_PLACE_ERR = float(HEADING['turn_in_place_error'])
SPOT_TURN_ERR = float(HEADING['spot_turn_error'])
FORWARD_FULL_ERR = float(HEADING['forward_full_error'])
MIN_CURVE_SPEED = float(HEADING['min_curve_speed'])
FINAL_SLOW_RADIUS = float(GOAL['final_slow_radius'])
CMD_V_DEADBAND = float(CTRL['v_deadband'])
CMD_W_DEADBAND = float(CTRL['w_deadband'])
DETOUR_TRIGGER = float(DETOUR['trigger_distance'])
DETOUR_CLEAR = float(DETOUR['clear_distance'])
DETOUR_LOOKAHEAD = float(DETOUR['lookahead'])
DETOUR_OFFSET = float(DETOUR['offset'])
GLOBAL_LOOKAHEAD = float(GLOBAL['lookahead'])
GLOBAL_MAX_DEVIATION = float(GLOBAL['max_deviation'])
OBS_MAX_RANGE = float(OBS['max_range'])
OBS_MIN_Z = float(OBS['min_z'])
OBS_MAX_Z = float(OBS['max_z'])
OBS_VOXEL = float(OBS['voxel'])
OBS_MAX_POINTS = int(OBS['max_points'])
LOCAL_COSTMAP_THRESHOLD = int(DWA_CFG.get('local_costmap_threshold', 35))


class DWANav(Node):
    def __init__(self):
        super().__init__('dwa_nav')
        self.cmd_pub = self.create_publisher(Twist, TOPICS['cmd_vel'], 10)
        self.path_pub = self.create_publisher(Path, TOPICS['local_plan'], 10)
        self.target_pub = self.create_publisher(PoseStamped, f"{TOPICS['debug_prefix']}/target", 10)
        self.footprint_pub = self.create_publisher(PolygonStamped, f"{TOPICS['debug_prefix']}/footprint", 10)
        self.status_pub = self.create_publisher(String, f"{TOPICS['debug_prefix']}/status", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, '/mid360_points', self.on_cloud, 10)
        self.create_subscription(OccupancyGrid, TOPICS['local_costmap'], self.on_local_costmap, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_subscription(Path, '/global_plan', self.on_global_plan, 10)
        self.timer = self.create_timer(0.08, self.tick)
        self.local_pts = np.zeros((0, 2), dtype=np.float32)
        self.global_plan = np.zeros((0, 2), dtype=np.float32)
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0
        self.goal = None
        self.nav_target = None
        self.detour_side = 0.0
        self.last_v = 0.0
        self.last_w = 0.0
        self.last_costmap_time = 0.0
        self.get_logger().info(f"DWA ready. Publishing {TOPICS['cmd_vel']}; set 2D Goal in RViz.")

    def on_cloud(self, msg):
        if time.monotonic() - self.last_costmap_time < 0.35:
            return
        if msg.point_step != 12 or len(msg.fields) < 3:
            self.get_logger().warn('Unsupported PointCloud2 layout; expected packed float32 xyz')
            return
        xyz = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(xyz).all(axis=1)
        horiz = np.linalg.norm(xyz[:, :2], axis=1)
        pre_mask = finite & (horiz > ROBOT_RADIUS) & (horiz < OBS_MAX_RANGE)
        pts3 = self._cloud_to_base(msg, xyz[pre_mask].astype(np.float32))
        if len(pts3) == 0:
            self.local_pts = np.zeros((0, 2), dtype=np.float32)
            return
        base_horiz = np.linalg.norm(pts3[:, :2], axis=1)
        mask = ((base_horiz > ROBOT_RADIUS + 0.08) & (base_horiz < OBS_MAX_RANGE) &
                (pts3[:, 2] > OBS_MIN_Z) & (pts3[:, 2] < OBS_MAX_Z) & (pts3[:, 0] > -1.2))
        self.local_pts = self._voxel_downsample(pts3[mask, :2].astype(np.float32))

    def on_local_costmap(self, msg):
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        idx = np.argwhere(data >= LOCAL_COSTMAP_THRESHOLD)
        if len(idx) == 0:
            self.local_pts = np.zeros((0, 2), dtype=np.float32)
            self.last_costmap_time = time.monotonic()
            return
        res = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        pts = np.empty((len(idx), 2), dtype=np.float32)
        pts[:, 0] = ox + (idx[:, 1].astype(np.float32) + 0.5) * res
        pts[:, 1] = oy + (idx[:, 0].astype(np.float32) + 0.5) * res
        if len(pts) > OBS_MAX_POINTS:
            d = np.linalg.norm(pts, axis=1)
            pts = pts[np.argsort(d)[:OBS_MAX_POINTS]]
        self.local_pts = pts
        self.last_costmap_time = time.monotonic()

    def on_odom(self, msg):
        self.rx = float(msg.pose.pose.position.x)
        self.ry = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.ryaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def on_goal(self, msg):
        self.goal = self._goal_to_odom(msg)
        self.nav_target = None
        self.global_plan = np.zeros((0, 2), dtype=np.float32)
        self.detour_side = 0.0
        self.last_v = 0.0
        self.last_w = 0.0
        self.get_logger().info(f'Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')

    def on_global_plan(self, msg):
        if len(msg.poses) == 0:
            self.global_plan = np.zeros((0, 2), dtype=np.float32)
            return
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) < 2:
            self.global_plan = np.zeros((0, 2), dtype=np.float32)
            return
        self.global_plan = pts
        self.nav_target = None
        self.detour_side = 0.0

    def tick(self):
        if self.goal is None:
            self._publish_empty_path()
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(None, None, [0.0, 0.0, 0.0], 'idle')
            return

        target = self._active_target()
        dx = target[0] - self.rx
        dy = target[1] - self.ry
        dist = math.hypot(dx, dy)
        final_dist = math.hypot(self.goal[0] - self.rx, self.goal[1] - self.ry)
        if final_dist < GOAL_TOL:
            self.goal = None
            self.nav_target = None
            self.detour_side = 0.0
            self.last_v = 0.0
            self.last_w = 0.0
            self.get_logger().info('Reached!')
            self._publish_empty_path()
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(target, final_dist, [0.0, 0.0, 0.0], 'reached')
            return

        target_yaw = math.atan2(dy, dx)
        heading_err = self._wrap(target_yaw - self.ryaw)
        if abs(heading_err) > SPOT_TURN_ERR:
            cmd = self._safe_cmd(0.0, 0.65 * heading_err)
            self._publish_cmd(cmd)
            self._publish_path(self._rollout(0.0, cmd[2]))
            self._publish_debug(target, final_dist, cmd, 'spot_turn')
            return

        v, w, _ = self._plan(target, target_yaw, dist)
        if final_dist < FINAL_SLOW_RADIUS:
            scale = max(0.0, min(1.0, (final_dist - GOAL_TOL) / max(0.05, FINAL_SLOW_RADIUS - GOAL_TOL)))
            v *= max(0.30, scale)
            w *= max(0.35, scale)
        cmd = self._safe_cmd(v, w)
        self._publish_cmd(cmd)
        self._publish_path(self._rollout(cmd[0], cmd[2]))
        self._publish_debug(target, final_dist, cmd, 'tracking')

    def _active_target(self):
        if self.goal is None:
            return (self.rx, self.ry)
        global_target = self._global_target()
        if global_target is not None:
            return global_target
        if self.nav_target is not None:
            if math.hypot(self.nav_target[0] - self.rx, self.nav_target[1] - self.ry) < 0.45:
                self.nav_target = None
                self.detour_side = 0.0
            elif self._front_distance() < DETOUR_CLEAR:
                return self.nav_target
            else:
                self.nav_target = None
                self.detour_side = 0.0

        if self._front_distance() >= DETOUR_TRIGGER or len(self.local_pts) == 0:
            return self.goal

        side = self._choose_detour_side()
        gx = self.goal[0] - self.rx
        gy = self.goal[1] - self.ry
        gnorm = math.hypot(gx, gy)
        if gnorm < 1e-3:
            return self.goal
        ux, uy = gx / gnorm, gy / gnorm
        px, py = -uy * side, ux * side
        look = min(DETOUR_LOOKAHEAD, max(0.8, gnorm * 0.45))
        self.nav_target = (self.rx + ux * look + px * DETOUR_OFFSET,
                           self.ry + uy * look + py * DETOUR_OFFSET)
        self.detour_side = side
        self.get_logger().info(
            f'DWA detour {"left" if side > 0 else "right"}: '
            f'({self.nav_target[0]:.2f}, {self.nav_target[1]:.2f})'
        )
        return self.nav_target

    def _plan(self, target, target_yaw, goal_dist):
        heading_err = abs(self._wrap(target_yaw - self.ryaw))
        align = self._alignment_scale(heading_err)
        v_min = max(0.0, self.last_v - MAX_DV)
        v_max = min(MAX_V, self.last_v + MAX_DV, 0.70 * goal_dist) * align
        if goal_dist > 0.6 and self._front_distance() > 0.65 and heading_err < TURN_IN_PLACE_ERR:
            v_max = max(v_max, min(MAX_V, MIN_CURVE_SPEED))
        w_min = max(-MAX_W, self.last_w - MAX_DW)
        w_max = min(MAX_W, self.last_w + MAX_DW)
        v_samples = np.unique(np.concatenate(([0.0], np.linspace(v_min, max(v_min, v_max), 7))))
        w_samples = np.linspace(w_min, w_max, 13)

        best = None
        best_score = float('inf')
        best_states = []
        obstacle_world = self._obstacles_world()
        local_goal = np.array(target, dtype=np.float32)
        final_goal = np.array(self.goal, dtype=np.float32)

        for v in v_samples:
            for w in w_samples:
                states = self._rollout(float(v), float(w))
                clearance = self._min_clearance(states, obstacle_world)
                if clearance < STOP_RADIUS:
                    continue
                terminal = states[-1, :2]
                dist_cost = float(np.linalg.norm(terminal - local_goal))
                final_cost = float(np.linalg.norm(terminal - final_goal))
                heading_cost = abs(self._wrap(target_yaw - states[-1, 2]))
                safety_span = max(0.05, SAFETY_RADIUS - STOP_RADIUS)
                clear_cost = max(0.0, SAFETY_RADIUS - clearance) / safety_span
                smooth_cost = 0.25 * abs(w - self.last_w) + 0.15 * abs(v - self.last_v)
                cruise_v = min(MAX_V, max(MIN_CURVE_SPEED, 0.55 * min(goal_dist, GLOBAL_LOOKAHEAD)))
                speed_cost = 0.65 * max(0.0, cruise_v - v)
                score = (3.6 * dist_cost + 0.6 * final_cost + 1.1 * heading_cost +
                         2.0 * clear_cost * clear_cost + smooth_cost + speed_cost)
                if score < best_score:
                    best_score = score
                    best = (float(v), float(w))
                    best_states = states

        if best is None:
            side = self._clearer_turn_side()
            return 0.0, side * min(MAX_W, max(0.25, abs(self.last_w))), self._rollout(0.0, side * 0.3)
        return best[0], best[1], best_states

    def _global_target(self):
        if len(self.global_plan) < 2:
            return None
        pos = np.array([self.rx, self.ry], dtype=np.float32)
        d = np.linalg.norm(self.global_plan - pos, axis=1)
        nearest = int(np.argmin(d))
        if float(d[nearest]) > GLOBAL_MAX_DEVIATION:
            return None
        lookahead = GLOBAL_LOOKAHEAD
        if self.goal is not None:
            lookahead = min(max(0.65, math.hypot(self.goal[0] - self.rx, self.goal[1] - self.ry)), lookahead)
        acc = 0.0
        idx = nearest
        while idx + 1 < len(self.global_plan) and acc < lookahead:
            step = self.global_plan[idx + 1] - self.global_plan[idx]
            acc += float(np.linalg.norm(step))
            idx += 1
        target = self.global_plan[idx]
        return (float(target[0]), float(target[1]))

    def _rollout(self, v, w):
        states = np.zeros((HORIZON, 3), dtype=np.float32)
        x, y, yaw = self.rx, self.ry, self.ryaw
        for i in range(HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw = self._wrap(yaw + w * DT)
            states[i] = [x, y, yaw]
        return states

    def _obstacles_world(self):
        if len(self.local_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = self.local_pts
        if len(pts) > 1200:
            d = np.linalg.norm(pts, axis=1)
            pts = pts[np.argsort(d)[:1200]]
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return pts @ rot.T + np.array([self.rx, self.ry], dtype=np.float32)

    def _voxel_downsample(self, pts):
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        grid = np.round(pts / OBS_VOXEL).astype(np.int32)
        _, idx = np.unique(grid, axis=0, return_index=True)
        pts = pts[np.sort(idx)]
        if len(pts) > OBS_MAX_POINTS:
            d = np.linalg.norm(pts, axis=1)
            pts = pts[np.argsort(d)[:OBS_MAX_POINTS]]
        return pts.astype(np.float32)

    @staticmethod
    def _min_clearance(states, obstacles):
        if len(obstacles) == 0:
            return 8.0
        d = np.linalg.norm(states[:, None, :2] - obstacles[None, :, :], axis=2)
        return float(np.min(d))

    def _front_distance(self):
        if len(self.local_pts) == 0:
            return 8.0
        x = self.local_pts[:, 0]
        y = self.local_pts[:, 1]
        front = (x > 0.0) & (np.abs(y) < 0.65)
        if not np.any(front):
            return 8.0
        return float(np.min(x[front]))

    def _choose_detour_side(self):
        if self.detour_side != 0.0:
            return self.detour_side
        return self._clearer_turn_side()

    def _clearer_turn_side(self):
        if len(self.local_pts) == 0:
            return 1.0
        x = self.local_pts[:, 0]
        y = self.local_pts[:, 1]
        near = (x > -0.2) & (x < 2.0) & (np.abs(y) < 1.5)
        if not np.any(near):
            return 1.0
        left = self.local_pts[near & (y > 0.0)]
        right = self.local_pts[near & (y < 0.0)]
        left_clear = float(np.min(np.linalg.norm(left, axis=1))) if len(left) else 8.0
        right_clear = float(np.min(np.linalg.norm(right, axis=1))) if len(right) else 8.0
        return 1.0 if left_clear >= right_clear else -1.0

    def _safe_cmd(self, v, w):
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        v = max(0.0, min(MAX_V, v))
        w = max(-MAX_W, min(MAX_W, w))
        if v <= 1e-4 and abs(w) > 0.01:
            v = 0.0
        elif v > self.last_v + MAX_DV:
            v = self.last_v + MAX_DV
        elif v < max(0.0, self.last_v - MAX_DV):
            v = max(0.0, self.last_v - MAX_DV)
        if w > self.last_w + MAX_DW:
            w = self.last_w + MAX_DW
        elif w < self.last_w - MAX_DW:
            w = self.last_w - MAX_DW
        if v < CMD_V_DEADBAND:
            v = 0.0
        if abs(w) < CMD_W_DEADBAND:
            w = 0.0
        self.last_v = v
        self.last_w = w
        return [v, 0.0, w]

    @staticmethod
    def _alignment_scale(heading_err):
        err = abs(heading_err)
        if err >= SPOT_TURN_ERR:
            return 0.0
        if err <= FORWARD_FULL_ERR:
            return 1.0
        if err <= TURN_IN_PLACE_ERR:
            return max(0.25, (TURN_IN_PLACE_ERR - err) / (TURN_IN_PLACE_ERR - FORWARD_FULL_ERR))
        return max(0.0, 0.25 * (SPOT_TURN_ERR - err) / (SPOT_TURN_ERR - TURN_IN_PLACE_ERR))

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

    def _cloud_to_base(self, msg, pts3):
        if msg.header.frame_id in ('', 'base_link'):
            return pts3
        try:
            tf = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, Time())
        except TransformException as exc:
            self.get_logger().warn(
                f'No TF {msg.header.frame_id}->base_link yet, using raw cloud: {exc}',
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

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _publish_path(self, states):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        for s in states:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(s[0])
            pose.pose.position.y = float(s[1])
            pose.pose.orientation.w = math.cos(float(s[2]) * 0.5)
            pose.pose.orientation.z = math.sin(float(s[2]) * 0.5)
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_empty_path(self):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

    def _publish_cmd(self, cmd):
        msg = Twist()
        msg.linear.x = float(cmd[0])
        msg.angular.z = float(cmd[2])
        self.cmd_pub.publish(msg)

    def _publish_debug(self, target, final_dist, cmd, state):
        if target is not None:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(target[0])
            pose.pose.position.y = float(target[1])
            pose.pose.orientation.w = 1.0
            self.target_pub.publish(pose)

        poly = PolygonStamped()
        poly.header.frame_id = 'base_link'
        poly.header.stamp = self.get_clock().now().to_msg()
        for i in range(24):
            a = 2.0 * math.pi * i / 24
            poly.polygon.points.append(
                Point32(x=float(ROBOT_RADIUS * math.cos(a)), y=float(ROBOT_RADIUS * math.sin(a)), z=0.0)
            )
        self.footprint_pub.publish(poly)

        status = String()
        front = self._front_distance()
        dist_text = 'none' if final_dist is None else f'{final_dist:.2f}'
        status.data = (
            f'state={state} cmd_v={cmd[0]:.3f} cmd_w={cmd[2]:.3f} '
            f'goal_dist={dist_text} front={front:.2f} obs={len(self.local_pts)}'
        )
        self.status_pub.publish(status)


def main():
    rclpy.init()
    node = DWANav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping DWA nav')
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
