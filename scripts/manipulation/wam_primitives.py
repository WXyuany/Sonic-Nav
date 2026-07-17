from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_vec(value: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.minimum(limits, np.maximum(-limits, value))


def smoothstep(value: float) -> float:
    x = clamp(float(value), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return float(a) * (1.0 - float(t)) + float(b) * float(t)


@dataclass(frozen=True)
class WorkspacePlan:
    movement_y: float
    duration: float
    error_y: float
    ready: bool


@dataclass(frozen=True)
class WorkspaceAligner:
    target_y: float
    tolerance: float
    lateral_gain: float
    speed: float
    min_duration: float
    max_duration: float
    duration_gain: float = 1.0
    response_sign: float = -1.0

    def plan(self, object_y: float) -> WorkspacePlan:
        error = float(object_y) - float(self.target_y)
        if abs(error) <= float(self.tolerance):
            return WorkspacePlan(0.0, float(self.min_duration), error, True)
        # Approximate local kinematics as d(error_y)/dt = b_y * movement_y.
        # For base-frame object coordinates, moving the robot +Y should reduce
        # object_y, so b_y is usually negative. The command is therefore the
        # projected inverse of that signed one-dimensional Jacobian.
        response_sign = -1.0 if float(self.response_sign) < 0.0 else 1.0
        movement_y = clamp(-response_sign * float(self.lateral_gain) * error, -1.0, 1.0)
        duration = clamp(
            abs(error) / max(0.03, float(self.speed)) * max(0.1, float(self.duration_gain)),
            float(self.min_duration),
            float(self.max_duration),
        )
        return WorkspacePlan(movement_y, duration, error, False)


@dataclass(frozen=True)
class GraspQuality:
    mean_error: np.ndarray
    rms_error: float
    max_error: float
    lateral_error: float
    vertical_error: float
    ready: bool

    @classmethod
    def from_contact_errors(
        cls,
        errors: dict[str, np.ndarray] | None,
        *,
        ready_error: float,
        max_ready_error: float | None = None,
        required_names: Sequence[str] | None = None,
    ) -> "GraspQuality | None":
        if not errors:
            return None
        values = np.vstack([np.asarray(value, dtype=np.float64) for value in errors.values()])
        norms = np.linalg.norm(values, axis=1)
        mean = np.mean(values, axis=0)
        rms = float(math.sqrt(np.mean(norms * norms)))
        max_error = float(np.max(norms))
        lateral = float(math.sqrt(mean[0] * mean[0] + mean[1] * mean[1]))
        vertical = float(abs(mean[2]))
        required_ready = True
        if required_names is not None:
            required_ready = all(name in errors for name in required_names)
        if max_ready_error is None:
            max_ready_error = float(ready_error) * 1.55
        return cls(
            mean_error=mean,
            rms_error=rms,
            max_error=max_error,
            lateral_error=lateral,
            vertical_error=vertical,
            ready=required_ready
            and rms <= float(ready_error)
            and max_error <= float(max_ready_error),
        )

    @classmethod
    def from_finger_errors(
        cls,
        errors: dict[str, np.ndarray] | None,
        *,
        ready_error: float,
    ) -> "GraspQuality | None":
        return cls.from_contact_errors(errors, ready_error=ready_error)


@dataclass(frozen=True)
class ContactServoConfig:
    error_gain: float
    hold_error_gain: float
    max_x_comp: float
    max_y_comp: float
    max_z_comp: float
    table_down_radius: float
    contact_ready_error: float
    capture_close_ratio: float
    close_ratio: float
    preload_close_ratio: float
    hold_close_ratio: float
    max_hold_close_ratio: float
    lift_detect_z: float
    lift_x_lead: float
    lift_z_lead: float
    lift_z_max_lead: float
    lift_ramp_start: float
    hold_z_lead: float
    transfer_lead: float
    place_lead: float
    place_down_lead: float


class ContactServoPolicy:
    def __init__(self, config: ContactServoConfig):
        self.config = config

    def contact_bias(
        self,
        quality: GraspQuality | None,
        *,
        radius: float,
        table_contact: bool,
    ) -> np.ndarray:
        if quality is None:
            return np.zeros(3, dtype=np.float64)
        gain = self.config.error_gain if table_contact else self.config.hold_error_gain
        limits = np.asarray(
            [self.config.max_x_comp, self.config.max_y_comp, self.config.max_z_comp],
            dtype=np.float64,
        )
        # Errors are measured as actual_contact - target_contact. Bias the
        # next target past the desired contact point so IK keeps correcting
        # toward the object instead of parking where the hand already is.
        bias = clamp_vec(-gain * quality.mean_error, limits)
        if table_contact:
            min_z = -float(self.config.table_down_radius) * max(0.025, float(radius))
            bias[2] = max(float(bias[2]), min_z)
        return bias

    def close_ratio(self, phase: str, ratio: float, quality: GraspQuality | None, lifted: bool) -> float:
        good_contact = quality.ready if quality is not None else False
        if phase == "lower_to_ball_open":
            return 0.0
        if phase == "capture_ball_contact":
            base = lerp(0.0, self.config.capture_close_ratio, smoothstep(ratio))
            if good_contact:
                base = max(base, self.config.capture_close_ratio)
            return clamp(base, 0.0, self.config.close_ratio)
        if phase == "close_on_ball":
            target = self.config.preload_close_ratio if good_contact else self.config.close_ratio
            return clamp(
                lerp(self.config.capture_close_ratio, target, smoothstep(ratio)),
                0.0,
                self.config.hold_close_ratio,
            )
        if phase == "squeeze_ball_secure":
            start = self.config.preload_close_ratio if good_contact else self.config.close_ratio
            target = self.config.hold_close_ratio if good_contact else self.config.preload_close_ratio
            return clamp(
                lerp(start, target, smoothstep(ratio)),
                0.0,
                self.config.max_hold_close_ratio,
            )
        if phase == "lift_ball":
            target = self.config.hold_close_ratio if lifted else self.config.preload_close_ratio
            return clamp(
                lerp(self.config.preload_close_ratio, target, smoothstep(ratio)),
                0.0,
                self.config.max_hold_close_ratio,
            )
        if phase in {"secure_ball", "move_to_place", "lower_to_place"}:
            return clamp(self.config.hold_close_ratio, 0.0, self.config.max_hold_close_ratio)
        return clamp(self.config.close_ratio, 0.0, self.config.max_hold_close_ratio)

    def lift_lead(self, ratio: float, *, lifted: bool) -> np.ndarray:
        if lifted:
            z_lead = self.config.hold_z_lead
        else:
            ramp = smoothstep(
                (float(ratio) - self.config.lift_ramp_start)
                / max(1e-3, 1.0 - self.config.lift_ramp_start)
            )
            z_lead = lerp(self.config.lift_z_lead, self.config.lift_z_max_lead, ramp)
        return np.asarray([self.config.lift_x_lead, 0.0, z_lead], dtype=np.float64)

    def transport_lead(self, object_base: np.ndarray, target_base: np.ndarray, *, lowering: bool) -> np.ndarray:
        direction = np.asarray(target_base, dtype=np.float64) - np.asarray(object_base, dtype=np.float64)
        if lowering:
            max_lead = self.config.place_lead
            direction[2] = min(float(direction[2]), -self.config.place_down_lead)
        else:
            max_lead = self.config.transfer_lead
            direction[2] = max(float(direction[2]), self.config.hold_z_lead)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return np.zeros(3, dtype=np.float64)
        return direction * min(1.0, max_lead / norm)
