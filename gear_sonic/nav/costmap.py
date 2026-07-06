from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalCostmapConfig:
    resolution: float = 0.06
    forward_range: float = 6.0
    backward_range: float = 1.2
    lateral_range: float = 3.2
    obstacle_radius: float = 0.05
    inflation_radius: float = 0.36
    occupied_threshold: int = 35

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LocalCostmapConfig":
        return cls(
            resolution=float(cfg.get("resolution", 0.06)),
            forward_range=float(cfg.get("forward_range", 6.0)),
            backward_range=float(cfg.get("backward_range", 1.2)),
            lateral_range=float(cfg.get("lateral_range", 3.2)),
            obstacle_radius=float(cfg.get("obstacle_radius", 0.05)),
            inflation_radius=float(cfg.get("inflation_radius", 0.36)),
            occupied_threshold=int(cfg.get("occupied_threshold", 35)),
        )

    @property
    def width(self) -> int:
        return max(1, int(math.ceil((self.forward_range + self.backward_range) / self.resolution)))

    @property
    def height(self) -> int:
        return max(1, int(math.ceil((2.0 * self.lateral_range) / self.resolution)))

    @property
    def origin(self) -> tuple[float, float]:
        return (-self.backward_range, -self.lateral_range)


def voxel_downsample_2d(points: np.ndarray, voxel_size: float, max_points: int | None = None) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    grid = np.round(points / max(1e-4, voxel_size)).astype(np.int32)
    _, idx = np.unique(grid, axis=0, return_index=True)
    out = points[np.sort(idx)]
    if max_points is not None and len(out) > max_points:
        d = np.linalg.norm(out, axis=1)
        out = out[np.argsort(d)[:max_points]]
    return out.astype(np.float32)


def filter_base_points(
    points_xyz: np.ndarray,
    *,
    robot_radius: float,
    max_range: float,
    min_z: float,
    max_z: float,
    min_x: float = -1.2,
) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    finite = np.isfinite(points_xyz).all(axis=1)
    horiz = np.linalg.norm(points_xyz[:, :2], axis=1)
    mask = (
        finite
        & (horiz > robot_radius)
        & (horiz < max_range)
        & (points_xyz[:, 2] > min_z)
        & (points_xyz[:, 2] < max_z)
        & (points_xyz[:, 0] > min_x)
    )
    return points_xyz[mask, :2].astype(np.float32)


def build_local_costmap(points_xy: np.ndarray, cfg: LocalCostmapConfig) -> np.ndarray:
    grid = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    if len(points_xy) == 0:
        return grid

    origin_x, origin_y = cfg.origin
    res = cfg.resolution
    max_radius = max(cfg.obstacle_radius, cfg.inflation_radius)
    max_cells = max(0, int(math.ceil(max_radius / res)))

    for px, py in points_xy:
        if px < origin_x - max_radius or px > cfg.forward_range + max_radius:
            continue
        if py < origin_y - max_radius or py > cfg.lateral_range + max_radius:
            continue
        cx = int(math.floor((float(px) - origin_x) / res))
        cy = int(math.floor((float(py) - origin_y) / res))
        ix0 = max(0, cx - max_cells)
        ix1 = min(cfg.width - 1, cx + max_cells)
        iy0 = max(0, cy - max_cells)
        iy1 = min(cfg.height - 1, cy + max_cells)
        for iy in range(iy0, iy1 + 1):
            wy = origin_y + (iy + 0.5) * res
            for ix in range(ix0, ix1 + 1):
                wx = origin_x + (ix + 0.5) * res
                d = math.hypot(wx - float(px), wy - float(py))
                if d <= cfg.obstacle_radius:
                    value = 100
                elif d <= cfg.inflation_radius:
                    span = max(1e-4, cfg.inflation_radius - cfg.obstacle_radius)
                    value = int(round(100.0 * (1.0 - (d - cfg.obstacle_radius) / span)))
                else:
                    continue
                if value > int(grid[iy, ix]):
                    grid[iy, ix] = np.int8(max(0, min(100, value)))
    return grid


def occupied_points_from_grid(
    grid: np.ndarray,
    cfg: LocalCostmapConfig,
    threshold: int | None = None,
) -> np.ndarray:
    thresh = cfg.occupied_threshold if threshold is None else threshold
    idx = np.argwhere(grid >= thresh)
    if len(idx) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    origin_x, origin_y = cfg.origin
    pts = np.empty((len(idx), 2), dtype=np.float32)
    pts[:, 0] = origin_x + (idx[:, 1].astype(np.float32) + 0.5) * cfg.resolution
    pts[:, 1] = origin_y + (idx[:, 0].astype(np.float32) + 0.5) * cfg.resolution
    return pts


def min_front_distance(points_xy: np.ndarray, half_width: float = 0.65) -> float:
    if len(points_xy) == 0:
        return 8.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    front = (x > 0.0) & (np.abs(y) < half_width)
    if not np.any(front):
        return 8.0
    return float(np.min(x[front]))
