#!/usr/bin/env -S /usr/bin/python3
import os, sys, math, time, struct, numpy as np, torch, rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray
sys.path.insert(0, os.path.expanduser("~/CARMA-MPPI-main"))
from carma_mppi.mppi import MPPI

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON, DT, NUM_SAMPLES = 20, 0.1, 500
SIGMA, LAMBDA_TEMP = 0.5, 0.3

class MPPIPlanner(Node):
    def __init__(self):
        super().__init__('mppi_planner')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.create_subscription(PointCloud2, '/mid360_points', self.on_cloud, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.timer = self.create_timer(0.05, self.tick)
        self.pts = np.zeros((0, 2), dtype=np.float32)
        self.rx = self.ry = 0.0; self.ryaw = 0.0
        self.goal = None; self.started = False
        self.mppi = MPPI(T=HORIZON, num_samples=NUM_SAMPLES, dt=DT,
                         u_min=[-1.5,-1.0], u_max=[1.5,1.0],
                         sigma=SIGMA, lambda_temp=LAMBDA_TEMP,
                         device=DEVICE, integration='rk4')
        self._u_prev = torch.zeros(1, 2, device=DEVICE)
        self.get_logger().info(f'MPPI ready ({NUM_SAMPLES}s, {DEVICE})')

    def on_cloud(self, m):
        buf = np.frombuffer(m.data, dtype=np.float32).reshape(-1, 3)
        mask = (np.abs(buf[:, 0]) < 30) & (np.abs(buf[:, 1]) < 30)
        self.pts = buf[mask][:, :2].astype(np.float32)

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x; self.ry = m.pose.pose.position.y
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
        if self.goal is None: self._send(pl); return
        dx = self.goal[0] - self.rx; dy = self.goal[1] - self.ry
        dist = math.hypot(dx, dy)
        if dist < 0.5: self.goal = None; self._send(pl); return

        state = torch.tensor([[self.rx, self.ry, self.ryaw]], device=DEVICE)
        noise = torch.randn(NUM_SAMPLES, HORIZON, 2, device=DEVICE) * SIGMA
        noise[:, :, 0] += self._u_prev[0, 0]; noise[:, :, 1] += self._u_prev[0, 1]
        noise = noise.clamp(-1.5, 1.5)

        traj = state.repeat(NUM_SAMPLES, 1); all_states = []
        for t in range(HORIZON):
            u = noise[:, t, :]
            v, w = u[:, 0:1], u[:, 1:2]
            traj = torch.cat([traj[:, 0:1]+v*torch.cos(traj[:, 2:3])*DT,
                              traj[:, 1:2]+v*torch.sin(traj[:, 2:3])*DT,
                              traj[:, 2:3]+w*DT], dim=1)
            all_states.append(traj.clone())

        costs = torch.zeros(NUM_SAMPLES, device=DEVICE)
        for t in range(HORIZON):
            s = all_states[t]
            ex = s[:,0] - self.goal[0]; ey = s[:,1] - self.goal[1]
            costs += 3.0 * (ex*ex + ey*ey)
            if t > 0: costs += 0.1 * ((noise[:,t]-noise[:,t-1])**2).sum(dim=1)

        if len(self.pts) > 0:
            op = torch.from_numpy(self.pts).to(DEVICE)
            for i in range(NUM_SAMPLES):
                sx, sy = all_states[-1][i,0].item(), all_states[-1][i,1].item()
                dists = torch.norm(op - torch.tensor([sx, sy], device=DEVICE), dim=1)
                min_d = dists.min().item()
                if min_d < 0.8: costs[i] += 100

        w = torch.exp(-(costs - costs.min()) / LAMBDA_TEMP)
        w /= w.sum() + 1e-8
        u_opt = (w.unsqueeze(1) * noise[:, 0, :]).sum(dim=0)
        u_opt[0] = u_opt[0].clamp(-1.0, 1.0); u_opt[1] = u_opt[1].clamp(-1.0, 1.0)
        self._u_prev = u_opt.unsqueeze(0)

        pl['navigate_cmd'] = [u_opt[0].item(), 0, u_opt[1].item()]
        self._send(pl)

    def _send(self, pl):
        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)

def main():
    rclpy.init(); n = MPPIPlanner()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
