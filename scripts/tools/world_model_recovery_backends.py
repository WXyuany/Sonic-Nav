#!/usr/bin/env -S /usr/bin/python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sonic_world.world_model import WorldObject, WorldState, anchor_to_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute routable world-model recovery backend requests."
    )
    parser.add_argument("--world-topic", default="/sonic_world/model")
    parser.add_argument("--object-anchor-topic", default="/sonic_world/object_anchor")
    parser.add_argument("--perception-reobserve-command-topic", default="/sonic_world/perception_reobserve_cmd")
    parser.add_argument("--status-topic", default="/sonic_world/recovery_backend_status")
    parser.add_argument("--perception-topic", default="/sonic_world/perception_recovery_request")
    parser.add_argument("--navigation-topic", default="/sonic_world/navigation_recovery_request")
    parser.add_argument("--runtime-topic", default="/sonic_world/runtime_recovery_request")
    parser.add_argument("--manual-topic", default="/sonic_world/manual_recovery_request")
    parser.add_argument("--navigation-command-topic", default="/sonic_world/navigation_micro_adjust_cmd")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel_nav")
    parser.add_argument("--runtime-replan-topic", default="/sonic_world/runtime_replan_request")
    parser.add_argument("--default-place-offset-y", type=float, default=0.28)
    return parser.parse_args()


class WorldModelRecoveryBackends(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("sonic_world_recovery_backends")
        self.args = args
        self.latest_world: WorldState | None = None
        self.anchor_pub = self.create_publisher(String, args.object_anchor_topic, 10)
        self.reobserve_pub = self.create_publisher(String, args.perception_reobserve_command_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.navigation_cmd_pub = self.create_publisher(String, args.navigation_command_topic, 10)
        self.runtime_replan_pub = self.create_publisher(String, args.runtime_replan_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.active_micro_adjust: dict[str, Any] | None = None
        self.create_subscription(String, args.world_topic, self._world_cb, 10)
        self.create_subscription(String, args.perception_topic, self._perception_cb, 10)
        self.create_subscription(String, args.navigation_topic, self._navigation_cb, 10)
        self.create_subscription(String, args.runtime_topic, self._runtime_cb, 10)
        self.create_subscription(String, args.manual_topic, self._manual_cb, 10)
        self.create_timer(0.05, self._navigation_tick)
        self.get_logger().info(
            "Recovery backends listening: "
            f"world={args.world_topic} perception={args.perception_topic} "
            f"navigation={args.navigation_topic} runtime={args.runtime_topic} "
            f"anchor_out={args.object_anchor_topic}"
        )

    def _world_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            world = _world_from_payload(payload)
            if world is None:
                raise ValueError("payload does not contain a world model")
            self.latest_world = world
        except Exception as exc:
            self._publish_status(
                {
                    "route": "world",
                    "status": "error",
                    "action": "world_update_failed",
                    "error": str(exc),
                }
            )

    def _perception_cb(self, msg: String) -> None:
        request = self._parse_request(msg, route="perception")
        if request is None:
            return
        handler = str(request.get("handler") or "")
        if handler == "perception_reobserve":
            routed = _routed_request(request)
            command = dict(routed.get("command") or {})
            command.update(
                {
                    "schema": "sonic_perception_reobserve_command_v0",
                    "task_id": routed.get("task_id"),
                    "action_id": routed.get("action_id"),
                    "target_id": routed.get("target_id"),
                    "stamp": time.time(),
                }
            )
            self._publish_json(self.reobserve_pub, command)
            self._publish_status(
                _status(
                    routed,
                    route="perception",
                    status="issued",
                    action="request_rgbd_vlm_reobserve",
                    command_topic=self.args.perception_reobserve_command_topic,
                )
            )
            return
        if handler == "place_target_recovery":
            self._handle_place_target_recovery(request)
            return
        if handler == "support_surface_inference":
            self._handle_support_surface(request)
            return
        if handler == "affordance_repair":
            self._handle_object_anchor(request, action="publish_affordance_repair", include_affordances=True)
            return
        self._handle_object_anchor(request, action="publish_object_anchor")

    def _navigation_cb(self, msg: String) -> None:
        request = self._parse_request(msg, route="navigation")
        if request is None:
            return
        routed = _routed_request(request)
        command = dict(routed.get("command") or {})
        command.setdefault("schema", "sonic_navigation_recovery_command_v0")
        command.setdefault("target_id", routed.get("target_id"))
        command.setdefault("handler", routed.get("handler"))
        command.setdefault("task_id", routed.get("task_id"))
        command.setdefault("action_id", routed.get("action_id"))
        command.setdefault("reason", routed.get("reason"))
        command["stamp"] = time.time()
        self._publish_json(self.navigation_cmd_pub, command)
        if str(routed.get("handler") or "") == "navigation_micro_adjust":
            self._start_micro_adjust(routed, command)
        self._publish_status(
            _status(
                routed,
                route="navigation",
                status="issued",
                action="publish_navigation_recovery_command",
                command_topic=self.args.navigation_command_topic,
            )
        )

    def _start_micro_adjust(self, request: dict[str, Any], command: dict[str, Any]) -> None:
        if self.cmd_vel_pub.get_subscription_count() < 1:
            self._publish_status(
                _status(
                    request,
                    route="navigation",
                    status="blocked",
                    action="navigation_micro_adjust_blocked",
                    reason=f"no subscriber on {self.args.cmd_vel_topic}",
                )
            )
            return
        max_step = max(0.0, min(0.12, _finite(command.get("max_step_m"), 0.12)))
        speed = max(0.02, min(0.10, _finite(command.get("speed_mps"), 0.06)))
        direction = -1.0 if _finite(command.get("direction"), 1.0) < 0.0 else 1.0
        duration = max_step / speed if speed > 0.0 else 0.0
        self.active_micro_adjust = {
            "request": request,
            "speed": direction * speed,
            "deadline": time.monotonic() + duration,
            "distance_m": direction * max_step,
        }

    def _navigation_tick(self) -> None:
        active = self.active_micro_adjust
        if active is None:
            return
        msg = Twist()
        if time.monotonic() < float(active["deadline"]):
            msg.linear.x = float(active["speed"])
            self.cmd_vel_pub.publish(msg)
            return
        self.cmd_vel_pub.publish(msg)
        request = active["request"]
        self.active_micro_adjust = None
        replan_request = {
            "event": "runtime_replan_request",
            "source": "navigation_micro_adjust",
            "task_id": request.get("task_id"),
            "action_id": request.get("action_id"),
            "target_id": request.get("target_id"),
            "handler": request.get("handler"),
            "reason": "navigation_micro_adjust_completed",
            "distance_m": active["distance_m"],
            "stamp": time.time(),
        }
        self._publish_json(self.runtime_replan_pub, replan_request)
        self._publish_status(
            _status(
                request,
                route="navigation",
                status="completed",
                action="navigation_micro_adjust_completed",
                distance_m=active["distance_m"],
                command_topic=self.args.cmd_vel_topic,
                runtime_replan_topic=self.args.runtime_replan_topic,
            )
        )

    def _runtime_cb(self, msg: String) -> None:
        request = self._parse_request(msg, route="runtime")
        if request is None:
            return
        routed = _routed_request(request)
        command = dict(routed.get("command") or {})
        command.setdefault("schema", "sonic_runtime_replan_request_v0")
        command.setdefault("target_id", routed.get("target_id"))
        command.setdefault("handler", routed.get("handler"))
        command.setdefault("task_id", routed.get("task_id"))
        command.setdefault("action_id", routed.get("action_id"))
        command["stamp"] = time.time()
        self._publish_json(self.runtime_replan_pub, command)
        self._publish_status(
            _status(
                routed,
                route="runtime",
                status="issued",
                action="publish_runtime_replan_request",
                command_topic=self.args.runtime_replan_topic,
            )
        )

    def _manual_cb(self, msg: String) -> None:
        request = self._parse_request(msg, route="manual")
        if request is None:
            return
        self._publish_status(
            _status(
                _routed_request(request),
                route="manual",
                status="blocked",
                action="manual_review_required",
                error="no automatic backend is registered for this recovery handler",
            )
        )

    def _parse_request(self, msg: String, *, route: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("recovery backend payload must be an object")
            return payload
        except Exception as exc:
            self._publish_status(
                {
                    "route": route,
                    "status": "error",
                    "action": "parse_request_failed",
                    "error": str(exc),
                    "raw": msg.data[:500],
                }
            )
            return None

    def _handle_object_anchor(
        self,
        request: dict[str, Any],
        *,
        action: str,
        include_affordances: bool = False,
    ) -> None:
        routed = _routed_request(request)
        obj = self._target_object(routed.get("target_id"))
        if obj is None:
            self._publish_status(
                _status(routed, route="perception", status="blocked", action=action, error="target object not in latest world")
            )
            return
        anchor = self._anchor_for_object(
            obj,
            handler=str(routed.get("handler") or ""),
            include_affordances=include_affordances,
            synthesize_base=True,
        )
        self._publish_json(self.anchor_pub, anchor)
        self._publish_status(
            _status(
                routed,
                route="perception",
                status="issued",
                action=action,
                anchor_topic=self.args.object_anchor_topic,
                synthesized_base=bool(anchor["objects"][0].get("properties", {}).get("pose_base_synthesized")),
            )
        )
        self._request_runtime_replan(routed, reason=action)

    def _handle_place_target_recovery(self, request: dict[str, Any]) -> None:
        routed = _routed_request(request)
        target_id = str(routed.get("target_id") or "place_target")
        existing = self._target_object(target_id)
        if existing is not None and existing.category == "place_target":
            anchor = self._anchor_for_object(existing, handler="place_target_recovery", synthesize_base=True)
        else:
            obj = self._target_object(routed.get("command", {}).get("object_id")) or self._primary_object()
            if obj is None:
                self._publish_status(
                    _status(
                        routed,
                        route="perception",
                        status="blocked",
                        action="publish_place_target_anchor",
                        error="no object available to seed a place target",
                    )
                )
                return
            anchor = self._place_target_anchor(obj, target_id=target_id)
        self._publish_json(self.anchor_pub, anchor)
        self._publish_status(
            _status(
                routed,
                route="perception",
                status="issued",
                action="publish_place_target_anchor",
                anchor_topic=self.args.object_anchor_topic,
            )
        )
        self._request_runtime_replan(routed, reason="place_target_recovery")

    def _handle_support_surface(self, request: dict[str, Any]) -> None:
        routed = _routed_request(request)
        obj = self._target_object(routed.get("target_id"))
        if obj is None:
            self._publish_status(
                _status(
                    routed,
                    route="perception",
                    status="blocked",
                    action="publish_support_surface_anchor",
                    error="target object not in latest world",
                )
            )
            return
        anchor = self._anchor_for_object(obj, handler="support_surface_inference", synthesize_base=True)
        record = anchor["objects"][0]
        record["support"] = record.get("support") or "table"
        anchor["relations"] = [{"subject_id": record["object_id"], "relation": "on", "object_id": record["support"], "confidence": 0.55}]
        self._publish_json(self.anchor_pub, anchor)
        self._publish_status(
            _status(
                routed,
                route="perception",
                status="issued",
                action="publish_support_surface_anchor",
                anchor_topic=self.args.object_anchor_topic,
            )
        )
        self._request_runtime_replan(routed, reason="support_surface_inference")

    def _request_runtime_replan(self, request: dict[str, Any], *, reason: str) -> None:
        payload = {
            "event": "runtime_replan_request",
            "source": "recovery_backend",
            "task_id": request.get("task_id"),
            "action_id": request.get("action_id"),
            "target_id": request.get("target_id"),
            "handler": request.get("handler"),
            "reason": reason,
            "stamp": time.time(),
        }
        self._publish_json(self.runtime_replan_pub, payload)
        self._publish_status(
            _status(
                request,
                route="runtime",
                status="issued",
                action="request_runtime_replan",
                reason=reason,
                command_topic=self.args.runtime_replan_topic,
            )
        )

    def _target_object(self, target_id: Any) -> WorldObject | None:
        if self.latest_world is None:
            return None
        if target_id:
            obj = self.latest_world.get_object(str(target_id))
            if obj is not None:
                return obj
        return None

    def _primary_object(self) -> WorldObject | None:
        return self.latest_world.primary_object() if self.latest_world is not None else None

    def _anchor_for_object(
        self,
        obj: WorldObject,
        *,
        handler: str,
        include_affordances: bool = False,
        synthesize_base: bool = False,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "object_id": obj.object_id,
            "category": obj.category,
            "shape": obj.shape.to_dict(),
            "support": obj.support,
            "properties": dict(obj.properties),
            "source": "world_model_recovery_backend",
        }
        if obj.pose_map is not None:
            record["pose_map"] = obj.pose_map.to_dict()
        if obj.pose_base is not None:
            record["pose_base"] = obj.pose_base.to_dict()
        elif synthesize_base:
            pose_base = _synthesized_base_pose(obj)
            if pose_base is not None:
                record["pose_base"] = pose_base
                record["properties"]["pose_base_synthesized"] = True
        if obj.pose_camera is not None:
            record["pose_camera"] = obj.pose_camera.to_dict()
        if include_affordances and obj.affordances:
            record["affordances"] = [aff.to_dict() for aff in obj.affordances]
        return {
            "scene": self.latest_world.properties.get("scene") if self.latest_world else None,
            "source": "world_model_recovery_backend",
            "frame_id": self.latest_world.frame_id if self.latest_world else "map",
            "objects": [record],
            "relations": _relations_for_object(self.latest_world, obj.object_id) if self.latest_world else [],
            "properties": {
                "recovery_handler": handler,
                "recovery_backend": "world_model_recovery_backends",
                "stamp": time.time(),
            },
        }

    def _place_target_anchor(self, obj: WorldObject, *, target_id: str) -> dict[str, Any]:
        map_pose = obj.pose_map.to_dict() if obj.pose_map else None
        base_pose = obj.pose_base.to_dict() if obj.pose_base else _synthesized_base_pose(obj)
        offset = float(self.args.default_place_offset_y)
        if map_pose and isinstance(map_pose.get("position"), list) and len(map_pose["position"]) >= 3:
            map_pose = dict(map_pose)
            map_pose["position"] = [float(map_pose["position"][0]), float(map_pose["position"][1]) + offset, float(map_pose["position"][2])]
        if base_pose and isinstance(base_pose.get("position"), list) and len(base_pose["position"]) >= 3:
            base_pose = dict(base_pose)
            base_pose["position"] = [float(base_pose["position"][0]), float(base_pose["position"][1]) + offset, float(base_pose["position"][2])]
        record = {
            "object_id": target_id,
            "category": "place_target",
            "shape": "target",
            "pose_map": map_pose,
            "pose_base": base_pose,
            "support": obj.support or "table",
            "source": "world_model_recovery_backend",
            "properties": {"generated_from_object_id": obj.object_id, "place_offset_y": offset},
        }
        return {
            "scene": self.latest_world.properties.get("scene") if self.latest_world else None,
            "source": "world_model_recovery_backend",
            "frame_id": self.latest_world.frame_id if self.latest_world else "map",
            "objects": [record],
            "relations": [{"subject_id": target_id, "relation": "on", "object_id": obj.support or "table", "confidence": 0.55}],
            "properties": {
                "recovery_handler": "place_target_recovery",
                "recovery_backend": "world_model_recovery_backends",
                "stamp": time.time(),
            },
        }

    def _publish_status(self, payload: dict[str, Any]) -> None:
        self._publish_json(self.status_pub, {"event": "recovery_backend_status", **payload})

    def _publish_json(self, publisher, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        publisher.publish(msg)


def _world_from_payload(payload: dict[str, Any]) -> WorldState | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("world"), dict):
        payload = payload["world"]
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        try:
            return anchor_to_world(payload)
        except Exception:
            return None
    anchor_objects = []
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        record = dict(obj)
        record.setdefault("object_id", object_id)
        anchor_objects.append(record)
    try:
        return anchor_to_world(
            {
                "scene": (payload.get("properties") or {}).get("scene"),
                "source": "world_model_payload",
                "frame_id": payload.get("frame_id", "map"),
                "objects": anchor_objects,
                "relations": payload.get("relations") or [],
                "properties": payload.get("properties") or {},
            }
        )
    except Exception:
        return None


def _routed_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    command = payload.get("command") if isinstance(payload.get("command"), dict) else request.get("command")
    if not isinstance(command, dict):
        command = {}
    return {
        "task_id": payload.get("task_id") or request.get("task_id"),
        "action_id": payload.get("action_id") or request.get("action_id"),
        "handler": payload.get("handler") or request.get("handler"),
        "target_id": payload.get("target_id") or request.get("target_id") or command.get("object_id"),
        "source": payload.get("source") or request.get("source"),
        "reason": payload.get("reason") or request.get("reason"),
        "command": command,
        "request": request or payload,
    }


def _status(
    request: dict[str, Any],
    *,
    route: str,
    status: str,
    action: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "route": route,
        "status": status,
        "action": action,
        "task_id": request.get("task_id"),
        "action_id": request.get("action_id"),
        "handler": request.get("handler"),
        "target_id": request.get("target_id"),
        "source": request.get("source"),
        "reason": request.get("reason"),
        **extra,
    }


def _synthesized_base_pose(obj: WorldObject) -> dict[str, Any] | None:
    grasp = obj.properties.get("grasp") if isinstance(obj.properties, dict) else {}
    if not isinstance(grasp, dict):
        grasp = {}
    x = _finite(grasp.get("reach_x") or grasp.get("approach_target_x"), 0.52)
    y = _finite(grasp.get("target_y"), 0.0)
    z = _finite(grasp.get("reach_z"), 0.04)
    if obj.pose_base is not None:
        return obj.pose_base.to_dict()
    if obj.pose_map is None and not grasp:
        return None
    return {
        "frame_id": "base_link",
        "position": [x, y, z],
    }


def _relations_for_object(world: WorldState, object_id: str) -> list[dict[str, Any]]:
    return [
        relation.to_dict()
        for relation in world.relations
        if relation.subject_id == object_id or relation.object_id == object_id
    ]


def _finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if out == out else float(default)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WorldModelRecoveryBackends(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
