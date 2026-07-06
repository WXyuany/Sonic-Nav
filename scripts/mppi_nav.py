#!/usr/bin/env -S /usr/bin/python3
import math
import os
import sys
import time

import numpy as np
import rclpy
import torch
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


CARMA_ROOT = os.environ.get('CARMA_MPPI_ROOT', os.path.expanduser('~/CARMA-MPPI-main'))
if CARMA_ROOT not in sys.path:
    sys.path.insert(0, CARMA_ROOT)

try:
    from carma_mppi.mppi import MPPI
except Exception:
    MPPI = None

try:
    from carma_mppi.planner import Planner
    from carma_mppi.robot import Robot
except Exception:
    Planner = None
    Robot = None

MPPI_DEFAULTS = {
    'controller': {'horizon': 24, 'dt': 0.10, 'samples': 1000, 'sigma': 0.30, 'lambda': 0.45,
                   'max_v': 0.42, 'max_w': 0.50, 'max_dv': 0.035, 'max_dw': 0.055},
    'robot': {'radius': 0.44, 'safety_radius': 0.78, 'hard_obstacle_radius': 0.54},
    'heading': {'turn_in_place_error': 1.35, 'spot_turn_error': 1.65,
                'forward_full_error': 0.45, 'min_curve_speed': 0.055},
    'detour': {'trigger_distance': 1.25, 'clear_distance': 1.65, 'offset': 0.90, 'lookahead': 1.35},
    'global_plan': {'lookahead': 1.15, 'max_deviation': 2.0},
    'obstacles': {'emergency_front': 0.38, 'max_range': 7.0, 'min_z': -0.55,
                  'max_z': 0.85, 'voxel': 0.10, 'max_points': 900},
    'carma': {'enabled': False},
    'topics': {'cmd_vel': '/cmd_vel_nav', 'local_costmap': '/local_costmap',
               'local_plan': '/local_plan', 'debug_prefix': '/sonic_nav/mppi'},
}

ENV_OVERRIDES = {
    'SONIC_MPPI_MAX_V': ('controller.max_v', float),
    'SONIC_MPPI_MAX_W': ('controller.max_w', float),
    'SONIC_MPPI_MAX_DV': ('controller.max_dv', float),
    'SONIC_MPPI_MAX_DW': ('controller.max_dw', float),
    'SONIC_MPPI_ROBOT_RADIUS': ('robot.radius', float),
    'SONIC_MPPI_SAFETY_RADIUS': ('robot.safety_radius', float),
    'SONIC_MPPI_HARD_OBS_RADIUS': ('robot.hard_obstacle_radius', float),
    'SONIC_MPPI_EMERGENCY_FRONT': ('obstacles.emergency_front', float),
    'SONIC_MPPI_TURN_ERR': ('heading.turn_in_place_error', float),
    'SONIC_MPPI_SPOT_TURN_ERR': ('heading.spot_turn_error', float),
    'SONIC_MPPI_FORWARD_FULL_ERR': ('heading.forward_full_error', float),
    'SONIC_MPPI_MIN_CURVE_SPEED': ('heading.min_curve_speed', float),
}

MPPI_CFG = overlay_env_scalars(load_config('mppi', MPPI_DEFAULTS, 'SONIC_MPPI_CONFIG'), ENV_OVERRIDES)
CTRL = MPPI_CFG['controller']
ROBOT = MPPI_CFG['robot']
HEADING = MPPI_CFG['heading']
DETOUR = MPPI_CFG['detour']
GLOBAL = MPPI_CFG['global_plan']
OBS = MPPI_CFG['obstacles']
CARMA = MPPI_CFG['carma']
TOPICS = MPPI_CFG['topics']

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON = int(CTRL['horizon'])
DT = float(CTRL['dt'])
N_SAMPLES = int(CTRL['samples'])
SIGMA = float(CTRL['sigma'])
LAMBDA = float(CTRL.get('lambda', CTRL.get('lambda_temp', 0.45)))
ROBOT_RADIUS = float(ROBOT['radius'])
SAFETY_RADIUS = float(ROBOT['safety_radius'])
HARD_OBS_RADIUS = float(ROBOT['hard_obstacle_radius'])
EMERGENCY_FRONT = float(OBS['emergency_front'])
MAX_V = float(CTRL['max_v'])
MAX_W = float(CTRL['max_w'])
MAX_DV = float(CTRL['max_dv'])
MAX_DW = float(CTRL['max_dw'])
TURN_IN_PLACE_ERR = float(HEADING['turn_in_place_error'])
SPOT_TURN_ERR = float(HEADING['spot_turn_error'])
FORWARD_FULL_ERR = float(HEADING['forward_full_error'])
MIN_CURVE_SPEED = float(HEADING['min_curve_speed'])
DETOUR_TRIGGER = float(DETOUR['trigger_distance'])
DETOUR_CLEAR = float(DETOUR['clear_distance'])
DETOUR_OFFSET = float(DETOUR['offset'])
DETOUR_LOOKAHEAD = float(DETOUR['lookahead'])
GLOBAL_LOOKAHEAD = float(GLOBAL['lookahead'])
GLOBAL_MAX_DEVIATION = float(GLOBAL['max_deviation'])
OBS_MAX_RANGE = float(OBS['max_range'])
OBS_MIN_Z = float(OBS['min_z'])
OBS_MAX_Z = float(OBS['max_z'])
OBS_VOXEL = float(OBS['voxel'])
OBS_MAX_POINTS = int(OBS['max_points'])
LOCAL_COSTMAP_THRESHOLD = int(MPPI_CFG.get('local_costmap_threshold', 35))
USE_CARMA = bool(CARMA.get('enabled', False)) or os.environ.get('SONIC_USE_CARMA', '0') == '1'
CARMA_CHECKPOINT = os.environ.get(
    'CARMA_MPPI_CHECKPOINT',
    os.path.join(CARMA_ROOT, 'carma_train', 'checkpoints', 'carma_final.pth'),
)

class MPPINav(Node):
    def __init__(self):
        super().__init__('mppi_nav')
        self.cmd_pub = self.create_publisher(Twist, TOPICS['cmd_vel'], 10)
        self.path_pub = self.create_publisher(Path, TOPICS['local_plan'], 10)
        self.ref_path_pub = self.create_publisher(Path, '/carma/ref_path', 10)
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
        self.timer = self.create_timer(0.05, self.tick)
        self.pts = np.zeros((0, 2), dtype=np.float32)
        self.local_pts = np.zeros((0, 2), dtype=np.float32)
        self.global_plan = np.zeros((0, 2), dtype=np.float32)
        self.rx = self.ry = 0.0; self.ryaw = 0.0; self.goal = None
        self.detour = None
        self.detour_side = 0
        self.last_costmap_time = 0.0
        self.carma = self._make_carma_planner()
        self._carma_failures = 0
        if MPPI is None:
            raise RuntimeError(f'Unable to import fallback MPPI from {CARMA_ROOT}')
        self.mppi = MPPI(T=HORIZON, num_samples=N_SAMPLES, dt=DT,
                         u_min=[0.0, -MAX_W], u_max=[MAX_V, MAX_W],
                         sigma=SIGMA, lambda_temp=LAMBDA, device=DEV, integration='rk4')
        self._u = torch.zeros(1, 2, device=DEV)
        self._u_seq = torch.zeros(HORIZON, 2, device=DEV)
        mode = 'CARMA planner' if self.carma is not None else 'MPPI fallback'
        self.get_logger().info(f'{mode} ready ({N_SAMPLES} samples, {DEV}); publishing {TOPICS["cmd_vel"]}')

    def _make_carma_planner(self):
        if not USE_CARMA:
            self.get_logger().info('Pure MPPI mode. Set SONIC_USE_CARMA=1 to enable CARMA.')
            return None
        if Planner is None or Robot is None:
            self.get_logger().warn('CARMA Planner import failed; using MPPI fallback')
            return None
        if not torch.cuda.is_available():
            self.get_logger().warn('CARMA Planner requires CUDA; using MPPI fallback')
            return None
        if not os.path.exists(CARMA_CHECKPOINT):
            self.get_logger().warn(f'CARMA checkpoint not found: {CARMA_CHECKPOINT}; using MPPI fallback')
            return None

        robot = Robot(
            length=0.70,
            width=0.55,
            kinematics='diff',
            max_speed=[MAX_V, MAX_W],
            max_acce=[1.2, 1.2],
        )
        cfg = {
            'step_time': 0.05,
            'receding': 20,
            'horizon_steps': 20,
            'mppi_samples': N_SAMPLES,
            'sigma': 0.45,
            'lambda_temp': 0.3,
            'rotation_ratio': 0.2,
            'ref_speed': 0.35,
            'range_max': 5.0,
            'integration': 'rk4',
            'num_anchors': 5,
            'anchor_lookahead': 2.0,
            'anchor_spread': 120.0,
            'w_path_follow': 6.0,
            'w_cte': 10.0,
            'w_obst_rep': 6.0,
            'w_obst_crit': 45.0,
            'w_goal': 5.0,
            'w_vnorm': 0.7,
            'w_u_norm': 0.2,
            'w_du_norm': 0.8,
            'w_angle': 5.0,
            'path_offset': 5,
            'obst_rep_dist': 3.0,
            'obst_crit_dist': 1.0,
            'robot': {
                'length': robot.length,
                'width': robot.width,
                'kinematics': robot.kinematics,
                'max_speed': robot.max_speed.tolist(),
                'max_acce': robot.max_acce.tolist(),
            },
            'carma': {
                'hidden_size': 512,
                'seq_len': 16,
                'edge_dim': 4,
                'frame_points': 4,
                'obs_sample_max': 100,
                'query_nearest_k': 10,
                'coord_max_num': 1,
                'checkpoint': CARMA_CHECKPOINT,
            },
            'ipath': {
                'arrive_threshold': 0.45,
                'waypoints': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            },
        }
        try:
            planner = Planner(cfg, robot, env_cfg=None)
        except Exception as exc:
            self.get_logger().warn(f'CARMA Planner init failed: {exc}; using MPPI fallback')
            return None
        planner.cfg = cfg
        planner.mppi.u_min = torch.tensor([0.0, -MAX_W], dtype=torch.float32)
        planner.mppi.u_max = torch.tensor([MAX_V, MAX_W], dtype=torch.float32)
        return planner

    def on_cloud(self, m):
        if time.monotonic() - self.last_costmap_time < 0.35:
            return
        if m.point_step != 12 or len(m.fields) < 3:
            self.get_logger().warn('Unsupported PointCloud2 layout; expected packed float32 xyz')
            return
        buf = np.frombuffer(m.data, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(buf).all(axis=1)
        horiz = np.linalg.norm(buf[:, :2], axis=1)
        pre_mask = finite & (horiz > ROBOT_RADIUS) & (horiz < OBS_MAX_RANGE)
        pts3 = buf[pre_mask].astype(np.float32)
        pts3 = self._cloud_to_base(m, pts3)
        if len(pts3) == 0:
            self.local_pts = np.zeros((0, 2), dtype=np.float32)
            self.pts = np.zeros((0, 2), dtype=np.float32)
            return
        base_horiz = np.linalg.norm(pts3[:, :2], axis=1)
        mask = ((base_horiz > ROBOT_RADIUS + 0.08) & (base_horiz < OBS_MAX_RANGE) &
                (pts3[:, 2] > OBS_MIN_Z) & (pts3[:, 2] < OBS_MAX_Z) & (pts3[:, 0] > -1.2))
        pts = self._voxel_downsample(pts3[mask, :2].astype(np.float32))
        self._set_local_obstacles(pts)

    def on_local_costmap(self, msg):
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        idx = np.argwhere(data >= LOCAL_COSTMAP_THRESHOLD)
        if len(idx) == 0:
            self._set_local_obstacles(np.zeros((0, 2), dtype=np.float32))
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
        self._set_local_obstacles(pts)
        self.last_costmap_time = time.monotonic()

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x; self.ry = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def on_goal(self, m):
        gx, gy = self._goal_to_odom(m)
        self.goal = (gx, gy)
        self.detour = None
        self.detour_side = 0
        self.global_plan = np.zeros((0, 2), dtype=np.float32)
        self._u = torch.zeros(1, 2, device=DEV)
        self._u_seq = torch.zeros(HORIZON, 2, device=DEV)
        if self.carma is not None:
            gq = m.pose.orientation
            gyaw = math.atan2(
                2.0 * (gq.w * gq.z + gq.x * gq.y),
                1.0 - 2.0 * (gq.y * gq.y + gq.z * gq.z),
            )
            path = np.linspace(
                np.array([self.rx, self.ry, self.ryaw], dtype=np.float32),
                np.array([self.goal[0], self.goal[1], gyaw], dtype=np.float32),
                max(80, int(3 * HORIZON)),
            )
            self.carma.frontend.set_global_path(path)
            self.carma._path = self.carma.frontend.path_world
        self.get_logger().info(f'Goal: ({self.goal[0]:.1f},{self.goal[1]:.1f})')

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
        self.detour = None
        self.detour_side = 0
        if self.carma is not None:
            self._set_carma_path_from_points(pts)

    def _goal_to_odom(self, msg):
        frame = msg.header.frame_id or 'odom'
        if frame == 'odom':
            return msg.pose.position.x, msg.pose.position.y
        try:
            tf = self.tf_buffer.lookup_transform('odom', frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f'No TF {frame}->odom for goal, using raw goal: {exc}')
            return msg.pose.position.x, msg.pose.position.y
        q = tf.transform.rotation
        t = tf.transform.translation
        rot = self._quat_to_rot(q.w, q.x, q.y, q.z)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        out = p @ rot.T + np.array([t.x, t.y, t.z], dtype=np.float32)
        return float(out[0]), float(out[1])

    def tick(self):
        if self.goal is None:
            self._publish_empty_path()
            self._u = torch.zeros(1, 2, device=DEV)
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(None, None, [0.0, 0.0, 0.0], 'idle')
            return
        dx = self.goal[0]-self.rx; dy = self.goal[1]-self.ry
        if math.hypot(dx, dy) < 0.5:
            self.goal = None
            self.detour = None
            self.detour_side = 0
            self._publish_empty_path()
            self._u = torch.zeros(1, 2, device=DEV)
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(self.goal, 0.0, [0.0, 0.0, 0.0], 'reached')
            return

        front_turn = self._emergency_turn()
        if front_turn is not None:
            if self.detour is None:
                self._fallback_target()
            cmd = self._safe_cmd(0.0, front_turn)
            self._publish_path(self._rollout_np(cmd[0], cmd[2]))
            self._publish_cmd(cmd)
            self._publish_debug(self.detour or self.goal, math.hypot(dx, dy), cmd, 'emergency_turn')
            return

        if self.carma is not None:
            if self._tick_carma():
                return

        nav_target = self._global_target()
        if nav_target is None:
            nav_target = self._fallback_target()
        tx = nav_target[0] - self.rx
        ty = nav_target[1] - self.ry
        target_yaw = math.atan2(ty, tx)
        heading_err = math.atan2(math.sin(target_yaw - self.ryaw), math.cos(target_yaw - self.ryaw))
        target_dist = math.hypot(tx, ty)
        align = self._alignment_scale(heading_err)
        v_ref = max(0.0, min(MAX_V, 0.35 * target_dist)) * align
        w_ref = max(-MAX_W, min(MAX_W, 0.55 * heading_err))
        ref_controls = self._u_seq.roll(-1, dims=0)
        ref_controls[-1] = ref_controls[-2]
        ref_controls[:, 0] = 0.78 * ref_controls[:, 0] + 0.22 * v_ref
        ref_controls[:, 1] = 0.78 * ref_controls[:, 1] + 0.22 * w_ref
        state = torch.tensor([self.rx, self.ry, self.ryaw], device=DEV)
        front_dist = self._front_distance()
        need_spin = abs(heading_err) > SPOT_TURN_ERR or front_dist < EMERGENCY_FRONT
        trajs, controls = self.mppi.sample(state, None, ref_controls, need_spot_turn=need_spin)

        costs = torch.zeros(trajs.shape[0], device=DEV)
        goal = torch.tensor([self.goal[0], self.goal[1]], device=DEV)
        target = torch.tensor([nav_target[0], nav_target[1]], device=DEV)
        start = torch.tensor([self.rx, self.ry], device=DEV)
        path_vec = target - start
        path_len = torch.norm(path_vec) + 1e-6
        path_dir = path_vec / path_len
        for t in range(HORIZON):
            s = trajs[:, t]
            rel = s[:, :2] - start
            progress = torch.clamp((rel @ path_dir) / path_len, 0.0, 1.2)
            closest = start + progress.unsqueeze(1) * path_vec
            cte = torch.norm(s[:, :2] - closest, dim=1)
            costs += 2.5 * cte * cte
            costs += -2.0 * progress
            yaw_err = target_yaw - s[:, 2]
            yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
            costs += 0.25 * yaw_err.abs()
        terminal = trajs[:, -1, :2]
        costs += 5.0 * torch.norm(terminal - target, dim=1)
        costs += 0.8 * torch.norm(terminal - goal, dim=1)
        costs += 0.16 * (controls * controls).sum(dim=(1, 2))
        costs += 3.0 * ((controls[:, 1:, :] - controls[:, :-1, :]) ** 2).sum(dim=(1, 2))

        if len(self.pts) > 8:
            op = torch.from_numpy(self._obstacle_subset()).to(DEV)
            for t in range(HORIZON):
                d = torch.cdist(trajs[:, t, :2], op).min(dim=1).values
                penetration = torch.clamp(SAFETY_RADIUS - d, min=0.0)
                costs += 320.0 * penetration * penetration
                costs += torch.where(d < HARD_OBS_RADIUS, torch.full_like(costs, 2500.0), torch.zeros_like(costs))

        weights = self.mppi.weight(costs)
        u_seq = (weights[:, None, None] * controls).sum(dim=0)
        self._u_seq = 0.55 * self._u_seq.roll(-1, dims=0) + 0.45 * u_seq
        u_opt = self._u_seq[0].clone()
        u_opt[0] = u_opt[0].clamp(0.0, MAX_V); u_opt[1] = u_opt[1].clamp(-MAX_W, MAX_W)
        u_opt = 0.75 * self._u.squeeze(0) + 0.25 * u_opt
        if not torch.isfinite(u_opt).all():
            self.get_logger().warn('Fallback MPPI produced non-finite command; stopping')
            self._u = torch.zeros(1, 2, device=DEV)
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(nav_target, math.hypot(dx, dy), [0.0, 0.0, 0.0], 'nonfinite_stop')
            return
        self._publish_path(self._rollout_controls_np(self._u_seq))

        u_v = float(u_opt[0].item())
        u_w = float(u_opt[1].item())
        u_v *= max(0.20, self._alignment_scale(heading_err))
        if target_dist > 0.6 and front_dist > 0.65 and abs(heading_err) < TURN_IN_PLACE_ERR:
            u_v = max(u_v, MIN_CURVE_SPEED)
        if abs(heading_err) > SPOT_TURN_ERR:
            u_v = 0.0
        cmd = self._safe_cmd(
            max(0.0, u_v),
            u_w,
        )
        self._publish_cmd(cmd)
        self._publish_debug(nav_target, math.hypot(dx, dy), cmd, 'tracking')

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

    def _set_local_obstacles(self, pts):
        self.local_pts = pts.astype(np.float32)
        if len(self.local_pts) == 0:
            self.pts = np.zeros((0, 2), dtype=np.float32)
            return
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        self.pts = self.local_pts @ rot.T + np.array([self.rx, self.ry], dtype=np.float32)

    @staticmethod
    def _quat_to_rot(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    def _tick_carma(self):
        pts_robot = self.local_pts.astype(np.float32).T
        state = np.array([self.rx, self.ry, self.ryaw], dtype=np.float32)
        try:
            action, info = self.carma(state, pts_robot)
        except Exception as exc:
            self._carma_failures += 1
            self.get_logger().warn(
                f'CARMA step failed, falling back this tick: {exc}', throttle_duration_sec=1.0
            )
            if self._carma_failures >= 3:
                self.get_logger().error('CARMA failed 3 consecutive ticks; disabling CARMA planner')
                self.carma = None
            return False
        self._carma_failures = 0

        if info.get('arrive'):
            self.goal = None
            self._u = torch.zeros(1, 2, device=DEV)
            self._publish_cmd([0.0, 0.0, 0.0])
            self._publish_debug(None, 0.0, [0.0, 0.0, 0.0], 'carma_reached')
            self.get_logger().info('Reached!')
            return True

        if not np.isfinite(action).all():
            self.get_logger().warn('CARMA produced non-finite command; using fallback MPPI')
            return False

        v = max(0.0, min(MAX_V, float(action[0, 0])))
        w = max(-MAX_W, min(MAX_W, float(action[1, 0])))
        prev = self._u.squeeze(0)
        u = torch.tensor([v, w], device=DEV)
        u = 0.70 * prev + 0.30 * u
        cmd = self._safe_cmd(max(0.0, float(u[0].item())), float(u[1].item()))
        self._publish_carma_viz()
        final_dist = None
        if self.goal is not None:
            final_dist = math.hypot(self.goal[0] - self.rx, self.goal[1] - self.ry)
        self._publish_cmd(cmd)
        self._publish_debug(self.goal, final_dist, cmd, 'carma')
        return True

    def _safe_cmd(self, v, w):
        if not (math.isfinite(v) and math.isfinite(w)):
            self._u = torch.zeros(1, 2, device=DEV)
            return [0.0, 0.0, 0.0]
        prev = self._u.squeeze(0)
        pv = float(prev[0].item())
        pw = float(prev[1].item())
        speed = max(0.0, min(MAX_V, v))
        if speed <= 1e-4 and abs(w) > 0.01:
            speed = 0.0
        elif speed > pv + MAX_DV:
            speed = pv + MAX_DV
        elif speed < max(0.0, pv - MAX_DV):
            speed = max(0.0, pv - MAX_DV)
        w = max(pw - MAX_DW, min(pw + MAX_DW, max(-MAX_W, min(MAX_W, w))))
        self._u = torch.tensor([[speed, w]], device=DEV)
        return [speed, 0.0, w]

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

    def _obstacle_subset(self):
        if len(self.pts) <= OBS_MAX_POINTS:
            return self.pts
        d = np.linalg.norm(self.pts - np.array([self.rx, self.ry], dtype=np.float32), axis=1)
        idx = np.argsort(d)[:OBS_MAX_POINTS]
        return self.pts[idx]

    def _fallback_target(self):
        if self.goal is None:
            return (self.rx, self.ry)
        if self.detour is not None:
            if math.hypot(self.detour[0] - self.rx, self.detour[1] - self.ry) < 0.45:
                self.detour = None
                self.detour_side = 0
            elif self._front_distance() < DETOUR_CLEAR:
                return self.detour
            else:
                self.detour = None
                self.detour_side = 0

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
        self.detour = (self.rx + ux * look + px * DETOUR_OFFSET, self.ry + uy * look + py * DETOUR_OFFSET)
        self.detour_side = side
        self.get_logger().info(
            f'Detour {"left" if side > 0 else "right"}: ({self.detour[0]:.2f}, {self.detour[1]:.2f})'
        )
        return self.detour

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

    def _set_carma_path_from_points(self, pts):
        if self.carma is None or len(pts) < 2:
            return
        path = np.zeros((len(pts), 3), dtype=np.float32)
        path[:, :2] = pts[:, :2]
        for i in range(len(pts) - 1):
            d = pts[i + 1] - pts[i]
            path[i, 2] = math.atan2(float(d[1]), float(d[0]))
        path[-1, 2] = path[-2, 2]
        self.carma.frontend.set_global_path(path)
        self.carma._path = self.carma.frontend.path_world

    def _choose_detour_side(self):
        if self.detour_side != 0:
            return self.detour_side
        if len(self.local_pts) == 0:
            return 1
        pts = self.local_pts
        x = pts[:, 0]
        y = pts[:, 1]
        near = (x > -0.2) & (x < 2.2) & (np.abs(y) < 2.0)
        if not np.any(near):
            return 1
        left = pts[near & (y > 0.0)]
        right = pts[near & (y < 0.0)]
        left_clear = float(np.min(np.linalg.norm(left, axis=1))) if len(left) else 40.0
        right_clear = float(np.min(np.linalg.norm(right, axis=1))) if len(right) else 40.0
        return 1 if left_clear >= right_clear else -1

    def _front_distance(self):
        if len(self.local_pts) == 0:
            return 40.0
        x = self.local_pts[:, 0]
        y = self.local_pts[:, 1]
        front = (x > 0.05) & (np.abs(y) < 0.55)
        if not np.any(front):
            return 40.0
        return float(np.min(x[front]))

    def _emergency_turn(self):
        if len(self.local_pts) == 0:
            return None
        x = self.local_pts[:, 0]
        y = self.local_pts[:, 1]
        front = (x > 0.0) & (x < EMERGENCY_FRONT) & (np.abs(y) < 0.45)
        if not np.any(front):
            return None
        left_clear = (
            np.min(np.linalg.norm(self.local_pts[(y > 0.15)], axis=1))
            if np.any(y > 0.15)
            else 40.0
        )
        right_clear = (
            np.min(np.linalg.norm(self.local_pts[(y < -0.15)], axis=1))
            if np.any(y < -0.15)
            else 40.0
        )
        return -0.45 if left_clear > right_clear else 0.45

    def _publish_path(self, states):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        for s in states:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(s[0])
            pose.pose.position.y = float(s[1])
            yaw = float(s[2])
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_empty_path(self):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

    def _rollout_np(self, v, w):
        states = np.zeros((HORIZON, 3), dtype=np.float32)
        x, y, yaw = self.rx, self.ry, self.ryaw
        for i in range(HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw = math.atan2(math.sin(yaw + w * DT), math.cos(yaw + w * DT))
            states[i] = [x, y, yaw]
        return states

    def _rollout_controls_np(self, controls):
        controls_np = controls.detach().cpu().numpy()
        states = np.zeros((len(controls_np), 3), dtype=np.float32)
        x, y, yaw = self.rx, self.ry, self.ryaw
        for i, (v, w) in enumerate(controls_np):
            x += float(v) * math.cos(yaw) * DT
            y += float(v) * math.sin(yaw) * DT
            yaw = math.atan2(math.sin(yaw + float(w) * DT), math.cos(yaw + float(w) * DT))
            states[i] = [x, y, yaw]
        return states

    def _publish_carma_viz(self):
        if self.carma is None:
            return
        if getattr(self.carma, 'opt_traj', None) is not None and len(self.carma.opt_traj) > 0:
            self._publish_path(self.carma.opt_traj)
        ref = getattr(self.carma, 'frontend', None)
        if ref is not None:
            gp = ref.path_from_pose(np.array([self.rx, self.ry, self.ryaw], dtype=np.float32))
            if gp is not None and len(gp) > 1:
                path = Path()
                path.header.frame_id = 'odom'
                path.header.stamp = self.get_clock().now().to_msg()
                for s in gp:
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x = float(s[0])
                    pose.pose.position.y = float(s[1])
                    pose.pose.orientation.w = 1.0
                    path.poses.append(pose)
                self.ref_path_pub.publish(path)

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
    rclpy.init(); n = MPPINav()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__': main()
