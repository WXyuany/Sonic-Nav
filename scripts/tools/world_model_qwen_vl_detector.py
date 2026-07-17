#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


SYSTEM_PROMPT = """You are the perception backend for a robot. Return exactly one valid JSON object and nothing else.
Use this compact schema exactly: {"detections":[{"object_id":"object_1","category":"object","bbox":[x0,y0,x1,y1],"confidence":0.9}]}.
Coordinates are pixels in the supplied image and bbox must satisfy x1>x0 and y1>y0. Use an empty
array, {"detections":[]}, when nothing is visible. Include place_target regions when the instruction
names a destination. Do not use markdown, prose, XML tags, or comments. Do not invent hidden objects."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible Qwen-VL endpoint as a 2D anchor detector.")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--request-topic", default="/sonic_world/perception_reobserve_cmd")
    parser.add_argument("--task-request-topic", default="/sonic_world/task_request")
    parser.add_argument("--detections-topic", default="/sonic_world/vlm_detections_2d")
    parser.add_argument("--status-topic", default="/sonic_world/qwen_vl_status")
    parser.add_argument("--endpoint", default=os.environ.get("QWEN_VL_ENDPOINT", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("QWEN_VL_MODEL", "qwen-vl"))
    parser.add_argument("--instruction", default="Locate manipulable objects, support surfaces, and visible target regions.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--api-key", default=os.environ.get("QWEN_VL_API_KEY", ""))
    parser.add_argument("--period", type=float, default=0.0, help="Optional continuous inference period; 0 means request-driven.")
    parser.add_argument("--audit-output", default="", help="Optional JSONL response audit; records text/parse metadata but never image bytes.")
    parser.add_argument("--audit-max-chars", type=int, default=1200)
    return parser.parse_args()


class QwenVlDetector(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_qwen_vl_detector")
        self.args = args
        self.latest_image: np.ndarray | None = None
        self.latest_frame = "camera_depth_optical_frame"
        self.latest_sensor_stamp: dict[str, int] | None = None
        self.busy = False
        self.last_run = 0.0
        self.active_instruction = str(args.instruction)
        self.audit_output = Path(args.audit_output).expanduser() if args.audit_output else None
        if self.audit_output is not None:
            self.audit_output.parent.mkdir(parents=True, exist_ok=True)
        self.detections_pub = self.create_publisher(String, args.detections_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 2)
        self.create_subscription(String, args.request_topic, self._request_cb, 10)
        self.create_subscription(String, args.task_request_topic, self._task_cb, 10)
        if args.period > 0.0:
            self.create_timer(max(0.2, float(args.period)), self._periodic)

    def _image_cb(self, msg: Image) -> None:
        if msg.encoding not in {"rgb8", "bgr8"}:
            return
        channels = 3
        image = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(int(msg.height), int(msg.width), channels)
        self.latest_image = image[:, :, ::-1].copy() if msg.encoding == "rgb8" else image.copy()
        self.latest_frame = msg.header.frame_id or self.latest_frame
        self.latest_sensor_stamp = _stamp_dict(msg.header.stamp)

    def _request_cb(self, msg: String) -> None:
        request = _json_object(msg.data)
        instruction = str(request.get("instruction") or request.get("purpose") or self.args.instruction)
        self._start(instruction, request=request)

    def _task_cb(self, msg: String) -> None:
        request = _json_object(msg.data)
        object_id = str(request.get("object_id") or "").strip()
        category = str(request.get("object_category") or "object").strip()
        target_id = str(request.get("target_id") or "").strip()
        if not object_id:
            return
        target_clause = (
            f" Also locate visible target region object_id='{target_id}' category='place_target'."
            if target_id
            else ""
        )
        self.active_instruction = (
            f"Locate only visible {category} object_id='{object_id}' category='{category}'."
            f"{target_clause} Ignore robot, background, and unrelated objects."
        )

    def _periodic(self) -> None:
        self._start(self.active_instruction, request=None)

    def _start(self, instruction: str, *, request: dict[str, Any] | None) -> None:
        if self.busy:
            self._status("qwen_vl_busy", "skipped")
            return
        if self.latest_image is None:
            self._status("qwen_vl_no_image", "blocked")
            return
        self.busy = True
        image = self.latest_image.copy()
        threading.Thread(
            target=self._infer,
            args=(image, self.latest_frame, dict(self.latest_sensor_stamp or {}), instruction, request),
            daemon=True,
        ).start()

    def _infer(
        self, image: np.ndarray, frame_id: str, sensor_stamp: dict[str, int], instruction: str, request: dict[str, Any] | None
    ) -> None:
        started = time.monotonic()
        try:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("failed to encode RGB image")
            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
            headers = {"Content-Type": "application/json"}
            if self.args.api_key:
                headers["Authorization"] = f"Bearer {self.args.api_key}"
            response = requests.post(
                self.args.endpoint,
                headers=headers,
                json={
                    "model": self.args.model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                },
                timeout=max(1.0, float(self.args.timeout)),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = _parse_model_json(content)
            raw_detections = _extract_raw_detections(payload)
            if not isinstance(raw_detections, list):
                raise ValueError(f"Qwen-VL response has no detections list (keys={','.join(sorted(str(key) for key in payload)[:12])})")
            detections = _normalize_detections(raw_detections, image_width=int(image.shape[1]), image_height=int(image.shape[0]))
            if not detections:
                raise ValueError("Qwen-VL response has no valid 2D detections")
            self._audit(content, payload=payload, detections=detections, status="completed")
            out = String()
            out.data = json.dumps(
                {
                    "schema": "sonic_vlm_detections_2d_v0",
                    "source": "qwen_vl",
                    "frame_id": frame_id,
                    "sensor_stamp": sensor_stamp,
                    "detections": detections,
                    "properties": {"model": self.args.model, "request": request},
                },
                separators=(",", ":"),
            )
            self.detections_pub.publish(out)
            self._status("qwen_vl_detections", "completed", object_count=len(detections), elapsed_s=time.monotonic() - started)
        except Exception as exc:
            self._audit(locals().get("content"), payload=locals().get("payload"), detections=[], status="failed", error=str(exc))
            self._status("qwen_vl_failed", "failed", error=str(exc), elapsed_s=time.monotonic() - started)
        finally:
            self.last_run = time.monotonic()
            self.busy = False

    def _audit(
        self,
        content: Any,
        *,
        payload: Any,
        detections: list[dict[str, Any]],
        status: str,
        error: str | None = None,
    ) -> None:
        if self.audit_output is None:
            return
        text = str(content or "")
        record = {
            "schema": "sonic_qwen_vl_response_audit_v0",
            "recorded_at": time.time(),
            "status": status,
            "response_text": text[: max(0, int(self.args.audit_max_chars))],
            "payload_keys": sorted(str(key) for key in payload) if isinstance(payload, dict) else [],
            "normalized_detection_count": len(detections),
            "detections": detections,
            "error": error,
        }
        with self.audit_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def _status(self, action: str, status: str, **extra: Any) -> None:
        msg = String()
        msg.data = json.dumps({"event": "qwen_vl_status", "action": action, "status": status, **extra}, separators=(",", ":"))
        self.status_pub.publish(msg)
        detail = str(extra.get("error") or "")
        if status in {"failed", "blocked"}:
            self.get_logger().warn(f"{action}: {detail or status}")
        elif action == "qwen_vl_detections":
            self.get_logger().info(
                f"{action}: objects={int(extra.get('object_count') or 0)} elapsed_s={float(extra.get('elapsed_s') or 0.0):.2f}"
            )


def _stamp_dict(stamp: Any) -> dict[str, int]:
    return {"sec": int(getattr(stamp, "sec", 0)), "nanosec": int(getattr(stamp, "nanosec", 0))}


def _parse_model_json(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        value = "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in value)
    text = str(value).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    compact = re.sub(r"\s+", " ", text)
    raise ValueError(f"Qwen-VL response did not contain a JSON object: {compact[:320]}")


def _extract_raw_detections(payload: dict[str, Any]) -> list[Any] | None:
    for key in ("detections", "objects", "results", "items", "regions", "boxes"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if _bbox_value(payload)[0] is not None:
        return [payload]
    for key in ("object", "result", "region"):
        value = payload.get(key)
        if isinstance(value, dict):
            return [value]
    return None


def _normalize_detections(raw: list[Any], *, image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Normalize common Qwen grounding field names to the RGB-D projection contract."""

    detections: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        detection = dict(item)
        bbox, bbox_key = _bbox_value(detection)
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            continue
        if bbox_key in {"bbox_2d", "box_2d", "coordinates"} and max(abs(value) for value in (x0, y0, x1, y1)) <= 1000.0:
            x0, x1 = x0 * image_width / 1000.0, x1 * image_width / 1000.0
            y0, y1 = y0 * image_height / 1000.0, y1 * image_height / 1000.0
        category = str(detection.get("category") or detection.get("label") or detection.get("class") or "object").strip()
        if not category:
            category = "object"
        detection["bbox"] = [x0, y0, x1, y1]
        detection["category"] = category
        detection.setdefault("object_id", f"{category}_{index}")
        detection.setdefault("confidence", 1.0)
        detection.setdefault("tracking_id", str(detection["object_id"]))
        detection.setdefault("support", "table")
        detection.setdefault("shape", _default_shape(category))
        detections.append(detection)
    return detections


def _bbox_value(payload: dict[str, Any]) -> tuple[Any, str | None]:
    for key in ("bbox", "bbox_2d", "box", "box_2d", "bounding_box", "coordinates", "rect"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            return value, key
    return None, None


def _default_shape(category: str) -> str:
    category = str(category).lower()
    if category in {"ball", "fruit", "sphere"}:
        return "sphere"
    if category in {"place_target", "target", "region"}:
        return "target"
    return "unknown"


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = QwenVlDetector(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
