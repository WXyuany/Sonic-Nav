#!/usr/bin/env -S /usr/bin/python3
"""LLM Brain Bridge — sensor data → scene description → action → feedback loop."""

import os, sys, math, json, threading, time
import numpy as np
import rclpy, msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import ByteMultiArray
from http.server import HTTPServer, BaseHTTPRequestHandler

ACTION_HISTORY = []
LIDAR_OFFSET = np.array([-0.0039635, 0.0], dtype=np.float32)
NEUTRAL_WRIST_POSE = [
    0.0903, 0.1615, -0.2411, 0.7295, 0.3145, 0.5533, -0.2506,
    0.1280, -0.1522, -0.2461, 0.7320, -0.2639, 0.5395, 0.3217,
]
NEUTRAL_HAND_JOINTS = [0.0] * 7


def _call_llm(scene_text, history_text):
    """Replace with Qwen API. Returns {"action":"...", "params":{...}, "reason":"..."}"""
    return {"action": "navigate", "params": {"target": [2, 0], "speed": 0.5},
            "reason": "default fallback — replace with Qwen"}


class SceneDescriber:
    def __init__(self):
        self.robot_pos = (0, 0); self.robot_yaw = 0
        self.obs_count = 0; self.min_obs_dist = 999
        self.obs_front = 999

    def update(self, pts, rx, ry, ryaw):
        self.robot_pos = (rx, ry); self.robot_yaw = ryaw
        if len(pts) > 0:
            self.obs_count = len(pts)
            dists = np.linalg.norm(pts - [rx, ry], axis=1)
            self.min_obs_dist = float(dists.min())
            diff = np.arctan2(pts[:,1]-ry, pts[:,0]-rx) - ryaw
            front_mask = np.abs(np.arctan2(np.sin(diff), np.cos(diff))) < 0.5
            self.obs_front = float(dists[front_mask].min()) if front_mask.any() else 999
        else:
            self.obs_count = 0; self.min_obs_dist = 999; self.obs_front = 999

    def describe(self, goal, current_action):
        p = [f"position({self.robot_pos[0]:.1f},{self.robot_pos[1]:.1f}) heading({math.degrees(self.robot_yaw):.0f}°)"]
        if self.obs_count > 0:
            p.append(f"{self.obs_count} lidar points, closest {self.min_obs_dist:.1f}m, front {self.obs_front:.1f}m")
        else:
            p.append("clear path")
        if goal:
            dx, dy = goal[0]-self.robot_pos[0], goal[1]-self.robot_pos[1]
            p.append(f"goal({goal[0]:.1f},{goal[1]:.1f}) dist {math.hypot(dx,dy):.1f}m bearing {math.degrees(math.atan2(dy,dx)-self.robot_yaw):.0f}°")
        p.append(f"current action: {current_action.get('action','idle')}")
        return " | ".join(p)


class LLMBridge(Node):
    def __init__(self):
        super().__init__('llm_bridge')
        self.pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 10)
        self.create_subscription(PointCloud2, '/mid360_points', self.on_cloud, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.scene = SceneDescriber()
        self.goal = None
        self.current_action = {"action": "idle"}
        self.rx = self.ry = self.ryaw = 0.0
        self.action_start = time.time()
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('LLM Brain ready')

    def on_cloud(self, m):
        if m.point_step != 12 or len(m.fields) < 3:
            self.get_logger().warn('Unsupported PointCloud2 layout; expected packed float32 xyz')
            return
        buf = np.frombuffer(m.data, dtype=np.float32).reshape(-1, 3)
        pts = buf[(np.abs(buf[:,0])<40)&(np.abs(buf[:,1])<40)][:,:2].astype(np.float32)
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = (pts + LIDAR_OFFSET) @ rot.T + np.array([self.rx, self.ry], dtype=np.float32)
        self.scene.update(pts, self.rx, self.ry, self.ryaw)

    def on_odom(self, m):
        self.rx = m.pose.pose.position.x; self.ry = m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z), 1-2*q.z*q.z)

    def set_action(self, action):
        global ACTION_HISTORY
        ACTION_HISTORY.append({"time": time.time(), "action": action,
                               "pos": (self.rx, self.ry)})
        if len(ACTION_HISTORY) > 20:
            ACTION_HISTORY.pop(0)
        self.current_action = action
        self.action_start = time.time()
        self.get_logger().info(f'LLM: {action["action"]} — {action.get("reason","")}')

    def get_status(self):
        return {
            "robot": {"x": self.rx, "y": self.ry, "yaw": self.ryaw},
            "obstacles": {"count": self.scene.obs_count,
                          "min_dist": self.scene.min_obs_dist,
                          "front_dist": self.scene.obs_front},
            "goal": self.goal,
            "current_action": self.current_action,
            "action_elapsed": time.time() - self.action_start,
            "history": ACTION_HISTORY[-5:]
        }

    def tick(self):
        act = self.current_action
        pl = {'toggle_policy_action': False, 'locomotion_mode': 0,
              'base_height_command': 0.78, 'navigate_cmd': [0, 0, 0],
              'wrist_pose': list(NEUTRAL_WRIST_POSE),
              'left_hand_joint': NEUTRAL_HAND_JOINTS,
              'right_hand_joint': NEUTRAL_HAND_JOINTS}
        p = act.get('params', {})

        if act['action'] == 'navigate':
            tx, ty = p.get('target', (0, 0))
            spd = p.get('speed', 0.5)
            dx, dy = tx - self.rx, ty - self.ry
            if math.hypot(dx, dy) < 0.3:
                self.current_action = {"action": "idle", "reason": "arrived"}
            else:
                err = math.atan2(dy, dx) - self.ryaw
                err = math.atan2(math.sin(err), math.cos(err))
                turn = max(-0.5, min(0.5, err * 1.0))
                fwd = max(0, min(spd, math.hypot(dx,dy)*0.5 - abs(err)*0.5))
                pl['navigate_cmd'] = [fwd, 0, turn]

        elif act['action'] == 'turn':
            target_yaw = math.radians(p.get('angle', 90))
            err = target_yaw - self.ryaw
            err = math.atan2(math.sin(err), math.cos(err))
            if abs(err) < 0.05:
                self.current_action = {"action": "idle", "reason": "facing correct direction"}
            else:
                pl['navigate_cmd'] = [0, 0, max(-0.5, min(0.5, err))]

        elif act['action'] == 'approach':
            tx, ty = p.get('target', (0, 0))
            dx, dy = tx - self.rx, ty - self.ry
            dist = math.hypot(dx, dy)
            stop_at = p.get('stop_at', 1.0)
            if dist < stop_at:
                self.current_action = {"action": "idle", "reason": f"stopped at {stop_at}m from target"}
            else:
                err = math.atan2(dy, dx) - self.ryaw
                err = math.atan2(math.sin(err), math.cos(err))
                pl['navigate_cmd'] = [min(0.3, dist*0.3), 0, max(-0.3, min(0.3, err*0.5))]

        elif act['action'] == 'crouch':
            pl['base_height_command'] = max(0.3, p.get('height', 0.4))

        elif act['action'] == 'reach':
            hand = p.get('hand', 'right')
            target = p.get('position', [0.2, 0.15, 0.8])
            wrist = list(NEUTRAL_WRIST_POSE)
            if hand == 'left':
                wrist[0:7] = target + [1, 0, 0, 0]
            else:
                wrist[7:14] = target + [1, 0, 0, 0]
            pl['wrist_pose'] = wrist

        elif act['action'] == 'stop':
            pass  # all zeros = stop

        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl, use_bin_type=True)]
        self.pub.publish(m)


class LLMHandler(BaseHTTPRequestHandler):
    bridge = None

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/decide':
            scene = self.bridge.scene.describe(self.bridge.goal, self.bridge.current_action)
            history = json.dumps(self.bridge.get_status()['history'])
            result = _call_llm(scene, history)
            self.bridge.set_action(result)
            self._json(result)

        elif self.path == '/scene':
            self._text(self.bridge.scene.describe(self.bridge.goal, self.bridge.current_action))

        elif self.path == '/goal':
            self.bridge.goal = (data['x'], data['y'])
            self._ok()

        elif self.path == '/status':
            self._json(self.bridge.get_status())

        elif self.path == '/say':
            print(f"[LLM SAYS] {data.get('text','')}", flush=True)
            self._ok()

    def do_GET(self):
        if self.path == '/scene':
            self._text(self.bridge.scene.describe(self.bridge.goal, self.bridge.current_action))
        elif self.path == '/status':
            self._json(self.bridge.get_status())

    def _json(self, data):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _text(self, text):
        self.send_response(200); self.send_header('Content-Type','text/plain'); self.end_headers()
        self.wfile.write(text.encode())

    def _ok(self):
        self.send_response(200); self.end_headers()

    def log_message(self, *args): pass


def main():
    rclpy.init(); bridge = LLMBridge(); LLMHandler.bridge = bridge

    def spin():
        while rclpy.ok(): rclpy.spin_once(bridge, timeout_sec=0.05)
    threading.Thread(target=spin, daemon=True).start()

    server = HTTPServer(('0.0.0.0', 8765), LLMHandler)
    print("LLM Brain: http://localhost:8765")
    print("  GET  /status  — robot state + action history")
    print("  POST /goal    — set goal {\"x\":2,\"y\":0}")
    print("  POST /decide  — LLM decision {\"scene\":\"...\",\"history\":\"...\"}")
    print("  POST /say     — LLM verbal output {\"text\":\"...\"}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    bridge.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
