#!/usr/bin/env python3
"""Qwen Client — polls scene, sends to Qwen, executes decisions."""
import time, json, requests

BRIDGE = "http://localhost:8765"
QWEN = "http://localhost:8000/v1/chat/completions"  # replace with your Qwen endpoint

SYSTEM_PROMPT = """You control a humanoid robot (Unitree G1). Output ONLY valid JSON.

Available actions:
  navigate  — walk to target coordinates {"target":[x,y], "speed":0.5}
  approach  — walk near target, stop N meters before {"target":[x,y], "stop_at":1.0}
  turn      — rotate in place {"angle":90}
  crouch    — lower body height {"height":0.4}
  stop      — stop all movement
  idle      — do nothing

Rules:
- If obstacle is closer than 1m, STOP first
- If goal is farther than 0.5m and path is clear, NAVIGATE
- If you arrived at goal, output idle
- Always include "reason" explaining your decision

Output format: {"action":"navigate","params":{"target":[2,0],"speed":0.5},"reason":"clear path to target"}
"""

goal = None

while True:
    try:
        status = requests.get(f"{BRIDGE}/status").json()
        scene = requests.get(f"{BRIDGE}/scene").text

        if goal is None:
            goal = input("Goal (x y) or press Enter for default (3,1.5): ") or "3 1.5"
            gx, gy = map(float, goal.split())
            requests.post(f"{BRIDGE}/goal", json={"x": gx, "y": gy})
            goal = (gx, gy)

        prompt = f"Scene: {scene}\nGoal: table at ({goal[0]},{goal[1]})\nDecide next action."
        resp = requests.post(QWEN, json={
            "model": "qwen",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }, timeout=30)

        text = resp.json()["choices"][0]["message"]["content"]
        action = json.loads(text)
        print(f"[Qwen] {action['action']} — {action.get('reason','')}")

        requests.post(f"{BRIDGE}/decide", json={"scene": scene, "goal": str(goal)})
        time.sleep(2)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
