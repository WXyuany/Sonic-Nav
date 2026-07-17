from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any


@dataclass(frozen=True)
class VlmDetection:
    object_id: str
    category: str
    pose_map: dict[str, Any] | None = None
    pose_base: dict[str, Any] | None = None
    pose_camera: dict[str, Any] | None = None
    shape: dict[str, Any] | str | None = None
    confidence: float = 1.0
    tracking_id: str | None = None
    support: str | None = None
    uncertainty: dict[str, Any] = field(default_factory=dict)
    pixel: Any = None
    affordances: list[Any] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VlmDetection":
        object_id = _string_first(payload, "object_id", "id", "name", "tracking_id")
        category = _string_first(payload, "category", "class", "label", "object_category") or "object"
        if not object_id:
            object_id = category
        return cls(
            object_id=object_id,
            category=category,
            pose_map=_mapping_first(payload, "pose_map", "center_map", "map_pose"),
            pose_base=_mapping_first(payload, "pose_base", "point_base", "base_pose"),
            pose_camera=_mapping_first(payload, "pose_camera", "point_camera", "camera_pose", "point_camera_depth"),
            shape=payload.get("shape"),
            confidence=_float(payload.get("confidence"), 1.0),
            tracking_id=_string_first(payload, "tracking_id", "track_id"),
            support=_string_first(payload, "support", "support_surface"),
            uncertainty=payload.get("uncertainty") if isinstance(payload.get("uncertainty"), dict) else {},
            pixel=payload.get("pixel") or payload.get("bbox") or payload.get("bbox_2d") or payload.get("mask_id"),
            affordances=payload.get("affordances") if isinstance(payload.get("affordances"), list) else [],
            properties=payload.get("properties") if isinstance(payload.get("properties"), dict) else {},
        )

    def to_anchor_object(self) -> dict[str, Any]:
        properties = dict(self.properties)
        properties.update(
            {
                "raw_anchor_kind": "vlm_detection",
                "confidence": self.confidence,
                "tracking_id": self.tracking_id,
                "uncertainty": self.uncertainty,
            }
        )
        return {
            "object_id": self.object_id,
            "category": self.category,
            "shape": self.shape or "unknown",
            "pose_map": self.pose_map,
            "pose_base": self.pose_base,
            "pose_camera": self.pose_camera,
            "support": self.support,
            "pixel": self.pixel,
            "affordances": self.affordances,
            "properties": properties,
            "source": "vlm_anchor_backend",
        }


def detections_to_anchor(
    detections: list[dict[str, Any]] | list[VlmDetection],
    *,
    scene: str | None = None,
    frame_id: str = "map",
    source: str = "vlm_anchor_backend",
    relations: list[dict[str, Any]] | None = None,
    robot_pose_map: dict[str, Any] | list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = [item if isinstance(item, VlmDetection) else VlmDetection.from_dict(item) for item in detections]
    anchor: dict[str, Any] = {
        "scene": scene,
        "source": source,
        "frame_id": frame_id,
        "objects": [item.to_anchor_object() for item in normalized],
        "relations": list(relations or []),
        "stamp": _stamp(),
        "properties": {
            "anchor_backend": source,
            "detection_count": len(normalized),
            **dict(metadata or {}),
        },
    }
    if robot_pose_map is not None:
        anchor["robot_pose_map"] = robot_pose_map
    return anchor


def detection_payload_to_anchor(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("objects") or payload.get("detections") or payload.get("instances") or []
    if not isinstance(raw, list):
        raise ValueError("VLM detection payload must contain a detections/objects list")
    return detections_to_anchor(
        raw,
        scene=_string_first(payload, "scene"),
        frame_id=_string_first(payload, "frame_id") or "map",
        source=_string_first(payload, "source") or "vlm_anchor_backend",
        relations=payload.get("relations") if isinstance(payload.get("relations"), list) else None,
        robot_pose_map=payload.get("robot_pose_map") or payload.get("base_pose_map"),
        metadata=payload.get("properties") if isinstance(payload.get("properties"), dict) else {},
    )


def project_detection_with_depth(
    detection: dict[str, Any],
    depth: Any,
    camera_k: list[float] | tuple[float, ...],
    *,
    frame_id: str = "camera_depth_optical_frame",
    patch_radius: int = 3,
) -> dict[str, Any]:
    """Add a robust RGB-D centroid and uncertainty to a 2D VLM detection."""

    import numpy as np

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2 or len(camera_k) < 9:
        raise ValueError("depth must be HxW and camera_k must contain 9 values")
    bbox = detection.get("bbox") or detection.get("bbox_2d") or detection.get("pixel")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError("detection requires bbox=[x0,y0,x1,y1]")
    x0, y0, x1, y1 = (float(value) for value in bbox[:4])
    u = int(round(0.5 * (x0 + x1)))
    v = int(round(0.5 * (y0 + y1)))
    radius = max(1, int(patch_radius))
    xa, xb = max(0, u - radius), min(array.shape[1], u + radius + 1)
    ya, yb = max(0, v - radius), min(array.shape[0], v + radius + 1)
    samples = array[ya:yb, xa:xb]
    valid = samples[np.isfinite(samples) & (samples > 0.0)]
    if valid.size == 0:
        raise ValueError("no valid depth samples at detection center")
    z = float(np.median(valid))
    fx, fy = float(camera_k[0]), float(camera_k[4])
    cx, cy = float(camera_k[2]), float(camera_k[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    point = [(u - cx) * z / fx, (v - cy) * z / fy, z]
    spread = float(np.median(np.abs(valid - z))) if valid.size > 1 else 0.0
    out = dict(detection)
    out["pose_camera"] = {"frame_id": frame_id, "position": point}
    out["uncertainty"] = {
        **(dict(detection.get("uncertainty")) if isinstance(detection.get("uncertainty"), dict) else {}),
        "depth_mad_m": spread,
        "depth_sample_count": int(valid.size),
        "pixel_radius": radius,
    }
    out.setdefault("tracking_id", _tracking_id(out, u=u, v=v))
    return out


def transform_point_pose(pose: dict[str, Any], transform: dict[str, Any], *, frame_id: str) -> dict[str, Any]:
    """Transform a point pose using translation + quaternion xyzw."""

    position = pose.get("position") if isinstance(pose, dict) else None
    translation = transform.get("translation") if isinstance(transform, dict) else None
    quaternion = transform.get("rotation_xyzw") if isinstance(transform, dict) else None
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError("pose position is missing")
    if not isinstance(translation, (list, tuple)) or len(translation) < 3:
        raise ValueError("transform translation is missing")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) < 4:
        raise ValueError("transform rotation_xyzw is missing")
    x, y, z, w = (float(value) for value in quaternion[:4])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("transform quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    px, py, pz = (float(value) for value in position[:3])
    # Quaternion-vector rotation expanded to avoid a geometry dependency.
    tx, ty, tz = 2.0 * (y * pz - z * py), 2.0 * (z * px - x * pz), 2.0 * (x * py - y * px)
    rx = px + w * tx + (y * tz - z * ty)
    ry = py + w * ty + (z * tx - x * tz)
    rz = pz + w * tz + (x * ty - y * tx)
    return {
        "frame_id": frame_id,
        "position": [rx + float(translation[0]), ry + float(translation[1]), rz + float(translation[2])],
    }


def _tracking_id(detection: dict[str, Any], *, u: int, v: int) -> str:
    category = str(detection.get("category") or detection.get("label") or "object")
    return f"{category}:{u // 16}:{v // 16}"


def _stamp() -> dict[str, int]:
    now = time.time()
    sec = int(now)
    return {"sec": sec, "nanosec": int((now - sec) * 1e9)}


def _string_first(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _mapping_first(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return {"frame_id": _frame_for_key(key), "position": list(value[:3])}
    return None


def _frame_for_key(key: str) -> str:
    if "base" in key:
        return "base_link"
    if "camera" in key:
        return "camera_depth_optical_frame"
    return "map"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
