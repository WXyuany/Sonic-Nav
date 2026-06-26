#!/usr/bin/env -S /usr/bin/python3
"""MPPI-based local planner for Sonic-Nav. Uses GPU trajectory sampling."""
import os, sys, math, time
import numpy as np
import torch
import rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray

sys.path.insert(0, os.path.expanduser("~/CARMA-MPPI-main"))
from carma_mppi.mppi import MPPI

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON = 20
DT = 0.1
NUM_SAMPLES = 500
SIGMA = 0.5
LAMBDA_TEMP = 0.3

class MPPIPlanner(Node):
    def __init__(self):
        super().__init__('mppi_planner')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.timer = self.create_timer(0.05, self.tick)
        self.scan = None
        self.rx = 0.0; self.ry = 0.0; self.ryaw = 0.0
        self.goal = None
        self.started = False
        self.mppi = MPPI(T=HORIZON, num_samples=NUM_SAMPLES, dt=DT,
                         u_min=[-1.5, -1.0], u_max=[1.5, 1.0],
                         sigma=SIGMA, lambda_temp=LAMBDA_TEMP,
                         device=DEVICE, integration='rk4')
        self._u_prev = torch.zeros(1, 2, device=DEVICE)
        self.get_logger().info(f'MPPI ready ({DEVICE}, {NUM_SAMPLES} samples)')

    def on_scan(self, m):
        self.scan = m

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x
        self.ry = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z), 1-2*q.z*q.z)

    def on_goal(self, m):
        self.goal = (m.pose.position.x, m.pose.position.y)
        self._u_prev = torch.zeros(1, 2, device=DEVICE)
        self.get_logger().info(f'Goal: ({self.goal[0]:.1f},{self.goal[1]:.1f})')

    def tick(self):
        pl = {'toggle_policy_action': not self.started, 'locomotion_mode': 0,
              'base_height_command': 0.78, 'navigate_cmd': [0, 0, 0]}
        self.started = True
        if self.goal is None or self.scan is None:
            self._send(pl); return

        dx = self.goal[0] - self.rx; dy = self.goal[1] - self.ry
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            self.goal = None
            self.get_logger().info('Goal reached!')
            self._send(pl); return

        # State: [x, y, theta]
        u_prev = self._u_prev.clone()
        state = torch.tensor([[self.rx, self.ry, self.ryaw]], device=DEVICE, dtype=torch.float32)

        # Sample trajectories
        noise = torch.randn(NUM_SAMPLES, HORIZON, 2, device=DEVICE) * SIGMA
        noise[:, :, 0] += u_prev[0, 0]
        noise[:, :, 1] += u_prev[0, 1]
        noise = noise.clamp(-1.5, 1.5)

        # Rollout
        traj = state.repeat(NUM_SAMPLES, 1)
        all_states = []
        for t in range(HORIZON):
            u = noise[:, t, :]
            v, w = u[:, 0:1], u[:, 1:2]
            dt_t = torch.tensor(DT, device=DEVICE)
            dx_t = v * torch.cos(traj[:, 2:3]) * dt_t
            dy_t = v * torch.sin(traj[:, 2:3]) * dt_t
            dtheta_t = w * dt_t
            traj = torch.cat([traj[:, 0:1] + dx_t, traj[:, 1:2] + dy_t,
                              traj[:, 2:3] + dtheta_t], dim=1)
            all_states.append(traj.clone())

        # Costs
        target_yaw = math.atan2(dy, dx)
        costs = torch.zeros(NUM_SAMPLES, device=DEVICE)

        for t in range(HORIZON):
            s = all_states[t]
            ex = s[:, 0] - self.goal[0]; ey = s[:, 1] - self.goal[1]
            costs += 3.0 * (ex * ex + ey * ey)  # goal cost

            # Obstacle cost from lidar
            if self.scan and t == HORIZON - 1:
                ranges = np.array(self.scan.ranges)
                angles = np.linspace(self.scan.angle_min, self.scan.angle_max, len(ranges))
                valid = (ranges > 0.1) & (ranges < self.scan.range_max)
                for i in range(min(NUM_SAMPLES, 100)):
                    sx, sy, st = s[i, 0].item(), s[i, 1].item(), s[i, 2].item()
                    min_obs = 5.0
                    for j in range(0, len(ranges), 3):
                        if not valid[j]: continue
                        ox = sx + ranges[j] * math.cos(st + angles[j])
                        oy = sy + ranges[j] * math.sin(st + angles[j])
                        min_obs = min(min_obs, ranges[j])
                    if min_obs < 0.5:
                        costs[i] += 100.0

            # Control smoothness
            if t > 0:
                du = noise[:, t, :] - noise[:, t-1, :]
                costs += 0.1 * (du[:, 0]**2 + du[:, 1]**2)

        # Softmax weights
        costs_min = costs.min()
        weights = torch.exp(-(costs - costs_min) / LAMBDA_TEMP)
        weights /= weights.sum() + 1e-8

        # Weighted average control
        u_opt = (weights.unsqueeze(1) * noise[:, 0, :]).sum(dim=0)
        u_opt[0] = u_opt[0].clamp(-1.0, 1.0)
        u_opt[1] = u_opt[1].clamp(-1.0, 1.0)
        self._u_prev = u_opt.unsqueeze(0)

        v, w = u_opt[0].item(), u_opt[1].item()
        pl['navigate_cmd'] = [v, 0, w]
        self._send(pl)

    def _send(self, pl):
        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)


def main():
    rclpy.init()
    n = MPPIPlanner()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
