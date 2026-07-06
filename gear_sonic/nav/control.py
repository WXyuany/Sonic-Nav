from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


NEUTRAL_WRIST_POSE = [
    0.0903, 0.1615, -0.2411, 0.7295, 0.3145, 0.5533, -0.2506,
    0.1280, -0.1522, -0.2461, 0.7320, -0.2639, 0.5395, 0.3217,
]
NEUTRAL_HAND_JOINTS = [0.0] * 7


@dataclass
class ControlConfig:
    locomotion_mode: int = 0
    base_height: float = 0.78
    max_v: float = 0.65
    max_w: float = 0.85
    max_dv: float = 0.08
    max_dw: float = 0.12
    v_deadband: float = 0.025
    w_deadband: float = 0.030
    command_timeout: float = 0.35
    upper_body_mode: str = "navigation"
    publish_rate: float = 25.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ControlConfig":
        limits = cfg.get("limits", {})
        return cls(
            locomotion_mode=int(cfg.get("locomotion_mode", 0)),
            base_height=float(cfg.get("base_height", 0.78)),
            max_v=float(limits.get("max_v", cfg.get("max_v", 0.65))),
            max_w=float(limits.get("max_w", cfg.get("max_w", 0.85))),
            max_dv=float(limits.get("max_dv", cfg.get("max_dv", 0.08))),
            max_dw=float(limits.get("max_dw", cfg.get("max_dw", 0.12))),
            v_deadband=float(limits.get("v_deadband", cfg.get("v_deadband", 0.025))),
            w_deadband=float(limits.get("w_deadband", cfg.get("w_deadband", 0.030))),
            command_timeout=float(cfg.get("command_timeout", 0.35)),
            upper_body_mode=str(cfg.get("upper_body_mode", "navigation")),
            publish_rate=float(cfg.get("publish_rate", 25.0)),
        )


class VelocityLimiter:
    def __init__(self, config: ControlConfig):
        self.config = config
        self.last_v = 0.0
        self.last_w = 0.0

    def reset(self) -> None:
        self.last_v = 0.0
        self.last_w = 0.0

    def limit(self, v: float, w: float, slew: bool = True) -> tuple[float, float]:
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        cfg = self.config
        v = max(-cfg.max_v, min(cfg.max_v, v))
        w = max(-cfg.max_w, min(cfg.max_w, w))
        if slew:
            v = max(self.last_v - cfg.max_dv, min(self.last_v + cfg.max_dv, v))
            w = max(self.last_w - cfg.max_dw, min(self.last_w + cfg.max_dw, w))
        if abs(v) < cfg.v_deadband:
            v = 0.0
        if abs(w) < cfg.w_deadband:
            w = 0.0
        self.last_v = v
        self.last_w = w
        return v, w


class SonicControlPayloadBuilder:
    def __init__(self, config: ControlConfig):
        self.config = config

    def payload(
        self,
        v: float = 0.0,
        w: float = 0.0,
        *,
        toggle: bool = False,
        upper_body_mode: str | None = None,
    ) -> dict[str, Any]:
        mode = upper_body_mode or self.config.upper_body_mode
        payload: dict[str, Any] = {
            "toggle_policy_action": bool(toggle),
            "locomotion_mode": int(self.config.locomotion_mode),
            "base_height_command": float(self.config.base_height),
            "navigate_cmd": [float(v), 0.0, float(w)],
        }

        if mode in {"navigation", "locked", "idle"}:
            payload["wrist_pose"] = list(NEUTRAL_WRIST_POSE)
            payload["left_hand_joint"] = list(NEUTRAL_HAND_JOINTS)
            payload["right_hand_joint"] = list(NEUTRAL_HAND_JOINTS)
        return payload
