#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, time, torch, numpy as np, rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
sys.path.insert(0, os.path.expanduser("~/CARMA-MPPI-main"))
from carma_mppi.mppi import MPPI
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON, DT, N_SAMPLES = 30, 0.1, 800
SIGMA, LAMBDA = 0.5, 0.2

class MPPINav(Node):
    def __init__(self):
        super().__init__('mppi_nav')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.create_subscription(PointCloud2, '/mid360_points', self.on_cloud, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.timer = self.create_timer(0.05, self.tick)
        self.pts = np.zeros((0, 2), dtype=np.float32)
        self.rx = self.ry = 0.0; self.ryaw = 0.0; self.goal = None
        self.mppi = MPPI(T=HORIZON, num_samples=N_SAMPLES, dt=DT,
                         u_min=[-1.5, -1.0], u_max=[1.5, 1.0],
                         sigma=SIGMA, lambda_temp=LAMBDA, device=DEV, integration='rk4')
        self._u = torch.zeros(1, 2, device=DEV)
        self.get_logger().info(f'MPPI ready ({N_SAMPLES}s, {DEV})')

    def on_cloud(self, m):
        buf = np.frombuffer(m.data, dtype=np.float32).reshape(-1, 3)
        self.pts = buf[(np.abs(buf[:, 0]) < 30) & (np.abs(buf[:, 1]) < 30)][:, :2].astype(np.float32)

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x; self.ry = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z), 1-2*q.z*q.z)

    def on_goal(self, m):
        self.goal = (m.pose.position.x, m.pose.position.y)
        self._u = torch.zeros(1, 2, device=DEV)
        self.get_logger().info(f'Goal: ({self.goal[0]:.1f},{self.goal[1]:.1f})')

    def tick(self):
        pl = {'toggle_policy_action': False, 'locomotion_mode': 0,
              'base_height_command': 0.78, 'navigate_cmd': [0, 0, 0]}
        if self.goal is None: self._send(pl); return
        dx = self.goal[0]-self.rx; dy = self.goal[1]-self.ry
        if math.hypot(dx, dy) < 0.5: self.goal = None; self._send(pl); return

        state = torch.tensor([[self.rx, self.ry, self.ryaw]], device=DEV)
        noise = torch.randn(N_SAMPLES, HORIZON, 2, device=DEV) * SIGMA
        noise[:, :, 0] += self._u[0, 0]; noise[:, :, 1] += self._u[0, 1]
        noise = noise.clamp(-1.5, 1.5)

        traj = state.repeat(N_SAMPLES, 1)
        all_s = []
        for t in range(HORIZON):
            u = noise[:, t, :]
            v, w = u[:, 0:1], u[:, 1:2]
            traj = torch.cat([traj[:, 0:1]+v*torch.cos(traj[:, 2:3])*DT,
                              traj[:, 1:2]+v*torch.sin(traj[:, 2:3])*DT,
                              traj[:, 2:3]+w*DT], dim=1)
            all_s.append(traj.clone())

        costs = torch.zeros(N_SAMPLES, device=DEV)
        target_yaw = math.atan2(dy, dx)
        for t in range(HORIZON):
            s = all_s[t]
            ex = s[:, 0]-self.goal[0]; ey = s[:, 1]-self.goal[1]
            costs += 3.0 * (ex*ex + ey*ey)
            yaw_err = target_yaw - s[:, 2]
            yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
            costs += 1.0 * yaw_err.abs()
            if t > 0: costs += 0.1 * ((noise[:, t]-noise[:, t-1])**2).sum(dim=1)

        if len(self.pts) > 0:
            op = torch.from_numpy(self.pts).to(DEV)
            for i in range(N_SAMPLES):
                sx, sy = all_s[-1][i, 0].item(), all_s[-1][i, 1].item()
                d = torch.norm(op - torch.tensor([sx, sy], device=DEV), dim=1).min().item()
                if d < 0.6: costs[i] += 200

        w = torch.exp(-(costs - costs.min()) / LAMBDA)
        w /= w.sum() + 1e-8
        u_opt = (w.unsqueeze(1) * noise[:, 0, :]).sum(dim=0)
        u_opt[0] = u_opt[0].clamp(-1.0, 1.0); u_opt[1] = u_opt[1].clamp(-0.8, 0.8)
        u_opt = 0.5 * self._u.squeeze(0) + 0.5 * u_opt
        self._u = u_opt.unsqueeze(0)

        pl['navigate_cmd'] = [u_opt[0].item(), 0, u_opt[1].item()]
        self._send(pl)

    def _send(self, pl):
        m = ByteMultiArray(); m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)

def main():
    rclpy.init(); n = MPPINav()
    try: rclpy.spin(n)
    except: pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
