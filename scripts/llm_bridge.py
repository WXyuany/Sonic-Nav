#!/usr/bin/env -S /usr/bin/python3
"""LLM Decision Bridge — sensor data → scene description → action command.

API: POST /decide  with JSON {"scene": "...", "goal": "..."}
Returns: JSON {"action": "navigate|stop|slow|crawl", "params": {...}}

The user replaces the _call_llm() method with their Qwen API.
"""

import os, sys, math, json, threading, time
import numpy as np
import rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import ByteMultiArray, String
from http.server import HTTPServer, BaseHTTPRequestHandler

# ============================================================
# USER: Replace this with your Qwen API call
# ============================================================
def _call_llm(scene_text: str, goal_text: str) -> dict:
    """
    Call your Qwen model here.
    Input: scene description + goal description
    Output: {"action": "navigate|stop|slow|crawl|grab", 
             "target": [x, y], "speed": 0.5, "reason": "..."}
    """
    # --- REPLACE WITH YOUR QWEN API ---
    # Example with OpenAI-compatible API:
    # import openai
    # client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    # prompt = f"Scene: {scene_text}\nGoal: {goal_text}\nChoose action: navigate/stop/slow/crawl. Output JSON."
    # resp = client.chat.completions.create(model="qwen", messages=[{"role":"user","content":prompt}])
    # return json.loads(resp.choices[0].message.content)
    
    # Fallback: simple rule-based
    if "obstacle" in scene_text.lower() and "close" in scene_text.lower():
        return {"action": "stop", "reason": "obstacle nearby"}
    return {"action": "navigate", "target": [2.0, 0.0], "speed": 0.5, "reason": "default"}
# ============================================================


class SceneDescriber:
    def __init__(self):
        self.obs_count = 0
        self.robot_pos = (0, 0)
        self.min_obs_dist = 999

    def update(self, pts, rx, ry):
        self.robot_pos = (rx, ry)
        if len(pts) > 0:
            self.obs_count = len(pts)
            dists = np.linalg.norm(pts - [rx, ry], axis=1)
            self.min_obs_dist = float(dists.min())
        else:
            self.obs_count = 0
            self.min_obs_dist = 999

    def describe(self, goal=None):
        parts = [f"Robot at ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f})"]
        if self.obs_count > 0:
            parts.append(f"{self.obs_count} lidar points detected")
            if self.min_obs_dist < 1.0:
                parts.append(f"obstacle VERY CLOSE ({self.min_obs_dist:.2f}m)")
            elif self.min_obs_dist < 3.0:
                parts.append(f"obstacle nearby ({self.min_obs_dist:.2f}m)")
        else:
            parts.append("clear path ahead")
        if goal:
            dx = goal[0] - self.robot_pos[0]; dy = goal[1] - self.robot_pos[1]
            parts.append(f"goal at ({goal[0]:.1f}, {goal[1]:.1f}), distance {math.hypot(dx,dy):.1f}m")
        return "; ".join(parts)


class LLMBridge(Node):
    def __init__(self):
        super().__init__('llm_bridge')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.create_subscription(PointCloud2, '/mid360_points', self.on_cloud, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.scene = SceneDescriber()
        self.goal = None
        self.current_action = {"action": "navigate", "target": [2, 0], "speed": 0.5}
        self.rx = self.ry = self.ryaw = 0.0
        self.timer = self.create_timer(0.2, self.tick)
        self.get_logger().info('LLM Bridge ready. POST /decide to control.')

    def on_cloud(self, m):
        buf = np.frombuffer(m.data, dtype=np.float32).reshape(-1, 3)
        pts = buf[(np.abs(buf[:,0])<30)&(np.abs(buf[:,1])<30)][:,:2].astype(np.float32)
        self.scene.update(pts, self.rx, self.ry)

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x; self.ry = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z), 1-2*q.z*q.z)

    def set_goal(self, goal):
        self.goal = goal

    def set_action(self, action):
        self.current_action = action
        self.get_logger().info(f'LLM decided: {action["action"]} ({action.get("reason","")})')

    def get_scene_text(self):
        return self.scene.describe(self.goal)

    def tick(self):
        act = self.current_action
        pl = {'toggle_policy_action': False, 'locomotion_mode': 0,
              'base_height_command': 0.78, 'navigate_cmd': [0, 0, 0]}

        if act['action'] == 'navigate' and 'target' in act:
            tx, ty = act['target']
            spd = act.get('speed', 0.5)
            dx, dy = tx - self.rx, ty - self.ry
            dist = math.hypot(dx, dy)
            if dist < 0.5:
                self.get_logger().info('Target reached')
                self.current_action = {"action": "idle"}
            else:
                target_yaw = math.atan2(dy, dx)
                err = target_yaw - self.ryaw
                err = math.atan2(math.sin(err), math.cos(err))
                turn = max(-0.5, min(0.5, err * 1.0))
                fwd = max(0, min(spd, dist * 0.5 - abs(err) * 0.5))
                pl['navigate_cmd'] = [fwd, 0, turn]
        elif act['action'] == 'slow':
            pl['navigate_cmd'] = [0.2, 0, 0]
        elif act['action'] == 'crawl':
            pl['locomotion_mode'] = 0
            pl['base_height_command'] = 0.3
            pl['navigate_cmd'] = [0.2, 0, 0]

        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)


class LLMHandler(BaseHTTPRequestHandler):
    bridge = None

    def do_POST(self):
        if self.path == '/decide':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            scene_text = data.get('scene', '')
            goal_text = data.get('goal', '')
            result = _call_llm(scene_text, goal_text)
            if self.bridge:
                self.bridge.set_action(result)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == '/scene':
            if self.bridge:
                desc = self.bridge.get_scene_text()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(desc.encode())
        elif self.path == '/goal':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read())
            if self.bridge:
                self.bridge.set_goal((data['x'], data['y']))
            self.send_response(200); self.end_headers()

    def log_message(self, *args): pass


def main():
    rclpy.init()
    bridge = LLMBridge()
    LLMHandler.bridge = bridge

    def spin():
        while rclpy.ok():
            rclpy.spin_once(bridge, timeout_sec=0.05)
    threading.Thread(target=spin, daemon=True).start()

    server = HTTPServer(('0.0.0.0', 8765), LLMHandler)
    print('LLM Bridge: http://localhost:8765')
    print('  POST /scene  → get scene description')
    print('  POST /goal   → set goal {"x":2,"y":0}')
    print('  POST /decide → llm decides action')
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    bridge.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
