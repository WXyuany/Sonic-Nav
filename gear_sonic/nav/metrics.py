from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class NavigationMetrics:
    goal_tolerance: float = 0.45
    collision_radius: float = 0.18
    stuck_speed: float = 0.035
    stuck_cmd: float = 0.08
    stuck_timeout: float = 3.0
    active: bool = False
    start_time: float | None = None
    last_time: float | None = None
    goal: tuple[float, float] | None = None
    last_pose: tuple[float, float, float] | None = None
    path_length: float = 0.0
    min_clearance: float = float("inf")
    collision_count: int = 0
    stuck_events: int = 0
    stuck_accum: float = 0.0
    angular_abs_sum: float = 0.0
    angular_samples: int = 0
    samples: int = 0
    reached: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def start(self, t: float, pose: tuple[float, float, float], goal: tuple[float, float]) -> None:
        self.active = True
        self.start_time = t
        self.last_time = t
        self.goal = goal
        self.last_pose = pose
        self.path_length = 0.0
        self.min_clearance = float("inf")
        self.collision_count = 0
        self.stuck_events = 0
        self.stuck_accum = 0.0
        self.angular_abs_sum = 0.0
        self.angular_samples = 0
        self.samples = 0
        self.reached = False
        self.history.clear()

    def update(
        self,
        t: float,
        pose: tuple[float, float, float],
        cmd: tuple[float, float],
        clearance: float | None = None,
    ) -> None:
        if not self.active or self.last_time is None or self.last_pose is None:
            return
        dt = max(0.0, t - self.last_time)
        dx = pose[0] - self.last_pose[0]
        dy = pose[1] - self.last_pose[1]
        step = math.hypot(dx, dy)
        self.path_length += step
        speed = step / dt if dt > 1e-6 else 0.0
        v_cmd, w_cmd = cmd

        if clearance is not None and math.isfinite(clearance):
            self.min_clearance = min(self.min_clearance, clearance)
            if clearance < self.collision_radius:
                self.collision_count += 1

        if abs(w_cmd) > 1e-5:
            self.angular_abs_sum += abs(w_cmd)
            self.angular_samples += 1

        if abs(v_cmd) > self.stuck_cmd and speed < self.stuck_speed:
            self.stuck_accum += dt
            if self.stuck_accum >= self.stuck_timeout:
                self.stuck_events += 1
                self.stuck_accum = 0.0
        else:
            self.stuck_accum = 0.0

        if self.goal is not None and math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]) <= self.goal_tolerance:
            self.reached = True

        self.samples += 1
        self.last_time = t
        self.last_pose = pose
        self.history.append(
            {
                "t": t,
                "x": pose[0],
                "y": pose[1],
                "yaw": pose[2],
                "cmd_v": v_cmd,
                "cmd_w": w_cmd,
                "clearance": clearance,
            }
        )

    def summary(self, now: float | None = None) -> dict[str, Any]:
        end_time = now if now is not None else self.last_time
        duration = 0.0
        if self.start_time is not None and end_time is not None:
            duration = max(0.0, end_time - self.start_time)
        return {
            "active": self.active,
            "reached": self.reached,
            "duration_s": duration,
            "path_length_m": self.path_length,
            "min_clearance_m": None if self.min_clearance == float("inf") else self.min_clearance,
            "collision_count": self.collision_count,
            "stuck_events": self.stuck_events,
            "mean_abs_cmd_w": self.angular_abs_sum / max(1, self.angular_samples),
            "samples": self.samples,
            "goal": self.goal,
        }
