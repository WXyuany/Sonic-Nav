from __future__ import annotations

import math
from pathlib import Path
import json
from typing import Any

from .effect_verification import effect_evidence


class MujocoQposEffectObserver:
    """Read the live MuJoCo qpos snapshot and produce physical skill evidence."""

    def __init__(
        self,
        scene_xml: str | Path,
        *,
        qpos_path: str | Path = "/tmp/sonic_qpos.npy",
        qpos_meta_path: str | Path = "/tmp/sonic_qpos_meta.json",
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.scene_xml = Path(scene_xml).expanduser().resolve()
        self.qpos_path = Path(qpos_path).expanduser()
        self.qpos_meta_path = Path(qpos_meta_path).expanduser()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        pelvis = self._named_id(mujoco.mjtObj.mjOBJ_BODY, ("pelvis", "base_link", "trunk"))
        self.robot_body_ids = self._subtree_body_ids(pelvis)

    def snapshot(self, command: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        self._validate_snapshot_metadata()
        qpos = np.load(self.qpos_path, allow_pickle=False)
        count = min(len(qpos), self.model.nq)
        self.data.qpos[:count] = qpos[:count]
        self.mujoco.mj_forward(self.model, self.data)
        payload = command.get("command") if isinstance(command.get("command"), dict) else {}
        target_ref = payload.get("target_ref") if isinstance(payload.get("target_ref"), dict) else {}
        destination_ref = payload.get("destination_ref") if isinstance(payload.get("destination_ref"), dict) else {}
        target_body = self._body_id(target_ref)
        destination_position = self._entity_position(destination_ref)
        base_body = self._named_id(self.mujoco.mjtObj.mjOBJ_BODY, ("pelvis", "base_link", "trunk"))
        target_position = self._body_position(target_body)
        base_position = self._body_position(base_body)
        return {
            "target_body_id": target_body,
            "target_position": target_position,
            "destination_position": destination_position,
            "base_position": base_position,
            "target_contact_count": self._target_contact_count(target_body),
            "base_stable": bool(base_position and base_position[2] >= 0.45),
            "qpos_path": str(self.qpos_path),
        }

    def _validate_snapshot_metadata(self) -> None:
        if not self.qpos_meta_path.exists():
            raise RuntimeError(f"qpos metadata is missing: {self.qpos_meta_path}")
        payload = json.loads(self.qpos_meta_path.read_text(encoding="utf-8"))
        scene = Path(str(payload.get("scene_xml") or "")).expanduser().resolve()
        if scene != self.scene_xml:
            raise RuntimeError(f"qpos scene mismatch: live={scene} observer={self.scene_xml}")
        if int(payload.get("nq") or -1) != int(self.model.nq):
            raise RuntimeError(f"qpos model mismatch: live nq={payload.get('nq')} observer nq={self.model.nq}")

    def evaluate(
        self,
        command: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        skill = str(command.get("skill_name") or "")
        declared = command.get("effects") if isinstance(command.get("effects"), list) else []
        if not declared:
            contract = command.get("contract") if isinstance(command.get("contract"), dict) else {}
            declared = contract.get("effects") if isinstance(contract.get("effects"), list) else []
        results: dict[str, dict[str, Any]] = {}
        target = _vec(after.get("target_position"))
        base = _vec(after.get("base_position"))
        destination = _vec(after.get("destination_position"))
        before_target = _vec(before.get("target_position"))
        contacts = int(after.get("target_contact_count") or 0)

        for effect in declared:
            name = str(effect)
            if name in {"robot_near_object", "object_in_hand_workspace"}:
                distance = _distance(target, base)
                # Navigation must satisfy the same reachability envelope that
                # its immediate workspace-alignment successor requires. A
                # looser navigation success threshold passed unreachable poses
                # into manipulation and converted a base-placement problem
                # into repeated arm-IK failures.
                threshold = 0.78
                results[name] = _result(distance is not None and distance <= threshold, distance_m=distance, threshold_m=threshold)
            elif name == "object_contact_ready":
                results[name] = _result(contacts > 0, contact_count=contacts)
            elif name == "object_in_hand":
                dz = _z_delta(before_target, target)
                results[name] = _result(
                    contacts > 0 and dz is not None and dz >= 0.025 and bool(after.get("base_stable")),
                    contact_count=contacts,
                    z_delta_m=dz,
                    base_stable=bool(after.get("base_stable")),
                )
            elif name == "object_near_destination":
                distance = _distance(target, destination)
                results[name] = _result(distance is not None and distance <= 0.25 and contacts > 0, distance_m=distance, contact_count=contacts)
            elif name == "object_on_destination":
                distance = _distance_xy(target, destination)
                z_error = abs(target[2] - destination[2]) if target and destination else None
                results[name] = _result(
                    distance is not None and distance <= 0.14 and z_error is not None and z_error <= 0.18,
                    distance_xy_m=distance,
                    z_error_m=z_error,
                )
            elif name == "hand_free":
                results[name] = _result(contacts == 0, contact_count=contacts)
            elif name == "robot_at_goal":
                distance = _distance_xy(base, target)
                results[name] = _result(distance is not None and distance <= 0.45, distance_xy_m=distance)
            else:
                results[name] = _result(False, reason="unsupported effect")

        reason = f"MuJoCo qpos/contact verification for {skill}"
        return effect_evidence(results, source="mujoco_qpos", reason=reason)

    def _body_id(self, ref: dict[str, Any]) -> int | None:
        object_id = str(ref.get("object_id") or "")
        names = (
            str(ref.get("body_name") or ""),
            object_id,
            f"{object_id}_body" if object_id else "",
        )
        found = self._named_id(self.mujoco.mjtObj.mjOBJ_BODY, names)
        if found is not None:
            return found
        geom_names = (str(ref.get("geom_name") or ""), f"{object_id}_geom" if object_id else "")
        geom_id = self._named_id(self.mujoco.mjtObj.mjOBJ_GEOM, geom_names)
        return int(self.model.geom_bodyid[geom_id]) if geom_id is not None else None

    def _entity_position(self, ref: dict[str, Any]) -> list[float] | None:
        site_names = (str(ref.get("site_name") or ""), str(ref.get("object_id") or ""))
        site_id = self._named_id(self.mujoco.mjtObj.mjOBJ_SITE, site_names)
        if site_id is not None:
            return [float(item) for item in self.data.site_xpos[site_id]]
        return self._body_position(self._body_id(ref))

    def _named_id(self, kind: Any, names: tuple[str, ...]) -> int | None:
        for name in names:
            if not name:
                continue
            value = int(self.mujoco.mj_name2id(self.model, kind, name))
            if value >= 0:
                return value
        return None

    def _body_position(self, body_id: int | None) -> list[float] | None:
        if body_id is None:
            return None
        return [float(item) for item in self.data.xpos[body_id]]

    def _target_contact_count(self, body_id: int | None) -> int:
        if body_id is None:
            return 0
        count = 0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            other = body2 if body1 == body_id else body1 if body2 == body_id else None
            if other is not None and other in self.robot_body_ids:
                count += 1
        return count

    def _subtree_body_ids(self, root_id: int | None) -> set[int]:
        if root_id is None:
            return set()
        out = {root_id}
        changed = True
        while changed:
            changed = False
            for body_id in range(int(self.model.nbody)):
                if body_id not in out and int(self.model.body_parentid[body_id]) in out:
                    out.add(body_id)
                    changed = True
        return out


def _vec(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(item) for item in out) else None


def _distance(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _distance_xy(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _z_delta(before: tuple[float, float, float] | None, after: tuple[float, float, float] | None) -> float | None:
    if before is None or after is None:
        return None
    return after[2] - before[2]


def _result(passed: bool, **metrics: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **{key: value for key, value in metrics.items() if value is not None}}
