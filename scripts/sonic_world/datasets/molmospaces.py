from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from ..planners import TaskRequest


DEFAULT_BENCHMARK = Path(
    "external_dependencies/molmospaces-src/mlspaces_tests/data_generation/test_benchmark/benchmark.json"
)


@dataclass(frozen=True)
class MolmoSpacesEpisode:
    """Self-contained MolmoSpaces benchmark episode normalized for Sonic."""

    index: int
    source_path: Path
    house_index: int | None
    scene_dataset: str
    data_split: str
    robot_name: str
    robot_init_qpos: dict[str, list[float]]
    robot_base_pose: tuple[float, float, float, float, float, float, float] | None
    task_cls: str
    task_type: str | None
    task: dict[str, Any]
    language_description: str
    referral_expressions: dict[str, str]
    scene_object_poses: dict[str, tuple[float, float, float, float, float, float, float]]
    added_objects: dict[str, str]
    removed_objects: tuple[str, ...] = ()
    cameras: tuple[dict[str, Any], ...] = ()
    img_resolution: tuple[int, int] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def scene_key(self) -> str:
        house = "unknown" if self.house_index is None else str(self.house_index)
        return f"{self.scene_dataset}_house_{house}"

    @property
    def episode_id(self) -> str:
        source = _dict(self.raw.get("source"))
        traj_key = source.get("traj_key")
        if traj_key is not None:
            return f"{self.scene_key}_{traj_key}"
        return f"{self.scene_key}_episode_{self.index:06d}"

    @property
    def robot_position_map(self) -> tuple[float, float, float] | None:
        if self.robot_base_pose is None:
            return None
        return self.robot_base_pose[:3]

    @property
    def robot_yaw_map(self) -> float | None:
        if self.robot_base_pose is None:
            return None
        return pose7_yaw(self.robot_base_pose)

    @property
    def task_kind(self) -> str:
        return infer_task_kind(self.task)

    @property
    def pickup_object_id(self) -> str | None:
        return _string_first(
            self.task,
            "pickup_obj_name",
            "object_name",
            "target_obj_name",
            "door_body_name",
            "articulation_object_name",
        )

    @property
    def place_receptacle_id(self) -> str | None:
        return _string_first(self.task, "place_receptacle_name", "receptacle_name")


class MolmoSpacesBenchmark:
    """Reads MolmoSpaces JSON benchmarks and emits Sonic world-model anchors."""

    def __init__(self, path: str | Path = DEFAULT_BENCHMARK) -> None:
        self.path = Path(path)
        self._episodes: list[MolmoSpacesEpisode] | None = None

    def episodes(
        self,
        *,
        limit: int | None = None,
        task_kind: str | None = None,
        scene_dataset: str | None = None,
        house_index: int | None = None,
    ) -> list[MolmoSpacesEpisode]:
        episodes = list(self._load_episodes())
        if task_kind is not None:
            normalized = _normalize_task_kind(task_kind)
            episodes = [episode for episode in episodes if episode.task_kind == normalized]
        if scene_dataset is not None:
            episodes = [episode for episode in episodes if episode.scene_dataset == scene_dataset]
        if house_index is not None:
            episodes = [episode for episode in episodes if episode.house_index == house_index]
        if limit is not None:
            episodes = episodes[: max(0, limit)]
        return episodes

    def episode(self, episode_index: int = 0) -> MolmoSpacesEpisode:
        episodes = self._load_episodes()
        if not 0 <= episode_index < len(episodes):
            raise IndexError(
                f"MolmoSpaces episode index {episode_index} outside 0..{len(episodes) - 1} for {self.path}"
            )
        return episodes[episode_index]

    def episode_task_request(self, episode: MolmoSpacesEpisode) -> TaskRequest:
        kind = episode.task_kind
        pickup_id = episode.pickup_object_id
        target_id = None
        verb = "pick"
        if kind == "pick_place":
            verb = "pick_place"
            target_id = _place_target_id(episode.place_receptacle_id)
        elif kind == "navigate":
            verb = "navigate"
            pickup_id = _navigation_goal_id(episode)
        elif kind in {"open_close", "door_open"}:
            verb = "approach"
        metadata = self._episode_metadata(episode)
        metadata.update(
            {
                "molmospaces_task_kind": kind,
                "molmospaces_task_cls": episode.task_cls,
                "task_description": episode.language_description,
                "referral_expressions": dict(episode.referral_expressions),
            }
        )
        return TaskRequest(
            verb=verb,
            object_id=pickup_id,
            object_category="navigation_goal" if verb == "navigate" else None,
            target_id=target_id,
            metadata=metadata,
        )

    def episode_anchor(
        self,
        episode: MolmoSpacesEpisode,
        *,
        include_context: bool = True,
        max_context_objects: int = 40,
    ) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        robot_pose = episode.robot_base_pose
        pickup_id = episode.pickup_object_id
        pickup_pose = self._pickup_pose(episode)
        if episode.task_kind == "navigate":
            nav_goal = self._navigation_goal_object(episode)
            objects.append(nav_goal)
            pickup_id = episode.pickup_object_id
            if pickup_id is not None:
                target_pose = self._pickup_pose(episode)
                if target_pose is not None:
                    objects.append(
                        self._object_record(
                            episode,
                            pickup_id,
                            target_pose,
                            role="navigation_target",
                            robot_pose=robot_pose,
                        )
                    )
                    relations.append(
                        {
                            "subject": nav_goal["object_id"],
                            "relation": "viewpoint_for",
                            "object": pickup_id,
                            "confidence": 1.0,
                        }
                    )
        elif pickup_id is not None and pickup_pose is not None:
            objects.append(
                self._object_record(
                    episode,
                    pickup_id,
                    pickup_pose,
                    role="pickup",
                    robot_pose=robot_pose,
                )
            )
            relations.append(
                {
                    "subject": pickup_id,
                    "relation": "on",
                    "object": _support_for_pose(pickup_pose),
                    "confidence": 0.65,
                }
            )

        place_id = episode.place_receptacle_id
        if episode.task_kind == "pick_place" and place_id is not None:
            place_pose = self._place_target_pose(episode)
            if place_pose is not None:
                target_id = _place_target_id(place_id)
                objects.append(
                    {
                        "object_id": target_id,
                        "category": "place_target",
                        "shape": "target",
                        "pose_map": pose7_to_payload(place_pose),
                        "pose_base": pose7_to_base_payload(place_pose, robot_pose),
                        "source": "molmospaces",
                        "support": place_id,
                        "properties": {
                            "molmospaces_object_name": place_id,
                            "role": "place_target",
                            "scene": episode.scene_key,
                            "task_description": episode.language_description,
                        },
                    }
                )
                relations.append(
                    {
                        "subject": target_id,
                        "relation": "on",
                        "object": place_id,
                        "confidence": 0.8,
                    }
                )

        if include_context:
            for name, pose in self._context_objects(episode, pickup_id, place_id, max_context_objects):
                objects.append(
                    self._object_record(
                        episode,
                        name,
                        pose,
                        role="context",
                        robot_pose=robot_pose,
                    )
                )

        anchor: dict[str, Any] = {
            "scene": episode.scene_key,
            "source": "molmospaces_benchmark",
            "frame_id": "map",
            "objects": objects,
            "relations": relations,
            "properties": self._episode_metadata(episode),
        }
        if episode.robot_position_map is not None:
            anchor["robot_start_map"] = list(episode.robot_position_map)
        if episode.robot_yaw_map is not None:
            anchor["robot_start_yaw"] = episode.robot_yaw_map
        return anchor

    def _load_episodes(self) -> list[MolmoSpacesEpisode]:
        if self._episodes is None:
            self._episodes = [
                _episode_from_raw(index, source_path, raw)
                for index, (source_path, raw) in enumerate(_raw_episode_records(self.path))
            ]
        return self._episodes

    def _episode_metadata(self, episode: MolmoSpacesEpisode) -> dict[str, Any]:
        return {
            "dataset": "molmospaces",
            "benchmark_path": str(self.path),
            "episode_index": episode.index,
            "episode_id": episode.episode_id,
            "scene_key": episode.scene_key,
            "scene_dataset": episode.scene_dataset,
            "house_index": episode.house_index,
            "data_split": episode.data_split,
            "robot_name": episode.robot_name,
            "source_path": str(episode.source_path),
            "coordinate_transform": "map=molmospaces_mujoco_world_xyz_z_up",
            "img_resolution": list(episode.img_resolution) if episode.img_resolution is not None else None,
            "camera_count": len(episode.cameras),
            "removed_objects": list(episode.removed_objects),
        }

    def _pickup_pose(self, episode: MolmoSpacesEpisode) -> tuple[float, float, float, float, float, float, float] | None:
        raw_pose = episode.task.get("pickup_obj_start_pose")
        if raw_pose is not None:
            return _pose7_optional(raw_pose)
        pickup_id = episode.pickup_object_id
        if pickup_id is None:
            return None
        return episode.scene_object_poses.get(pickup_id)

    def _place_target_pose(
        self,
        episode: MolmoSpacesEpisode,
    ) -> tuple[float, float, float, float, float, float, float] | None:
        for key in ("place_receptacle_start_pose", "place_obj_goal_pose", "place_goal_pose"):
            pose = _pose7_optional(episode.task.get(key))
            if pose is not None:
                return pose
        place_id = episode.place_receptacle_id
        if place_id is not None and place_id in episode.scene_object_poses:
            return episode.scene_object_poses[place_id]
        return _pose7_optional(episode.task.get("pickup_obj_goal_pose"))

    def _navigation_goal_object(self, episode: MolmoSpacesEpisode) -> dict[str, Any]:
        target_pose = self._pickup_pose(episode)
        robot_pose = episode.robot_base_pose
        if target_pose is None:
            target_pose = robot_pose or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        yaw = None
        if robot_pose is not None:
            dx = target_pose[0] - robot_pose[0]
            dy = target_pose[1] - robot_pose[1]
            yaw = math.atan2(dy, dx)
        return {
            "object_id": _navigation_goal_id(episode),
            "category": "navigation_goal",
            "shape": "target",
            "pose_map": {
                "frame_id": "map",
                "position": [target_pose[0], target_pose[1], target_pose[2]],
                "yaw": yaw,
                "orientation_xyzw": [0.0, 0.0, math.sin((yaw or 0.0) * 0.5), math.cos((yaw or 0.0) * 0.5)],
            },
            "pose_base": pose7_to_base_payload(target_pose, robot_pose),
            "source": "molmospaces",
            "properties": {
                "goal_tolerance": float(episode.task.get("succ_pos_threshold") or 1.0),
                "role": "navigation_goal",
                "task_description": episode.language_description,
            },
        }

    def _context_objects(
        self,
        episode: MolmoSpacesEpisode,
        pickup_id: str | None,
        place_id: str | None,
        max_context_objects: int,
    ) -> list[tuple[str, tuple[float, float, float, float, float, float, float]]]:
        if max_context_objects <= 0:
            return []
        excluded = {value for value in (pickup_id, place_id) if value}
        items = [(name, pose) for name, pose in episode.scene_object_poses.items() if name not in excluded]
        focus = self._pickup_pose(episode) or episode.robot_base_pose
        if focus is not None:
            items.sort(key=lambda item: _distance_xy(item[1], focus))
        return items[:max_context_objects]

    def _object_record(
        self,
        episode: MolmoSpacesEpisode,
        name: str,
        pose: tuple[float, float, float, float, float, float, float],
        *,
        role: str,
        robot_pose: tuple[float, float, float, float, float, float, float] | None,
    ) -> dict[str, Any]:
        description = _referral_for(episode, name)
        category = infer_object_category(name, description, episode.added_objects.get(name))
        shape = infer_object_shape(category, description)
        grasp = _grasp_hint(shape, category, pose, robot_pose)
        record: dict[str, Any] = {
            "object_id": name,
            "category": category,
            "shape": shape,
            "pose_map": pose7_to_payload(pose),
            "pose_base": pose7_to_base_payload(pose, robot_pose),
            "source": "molmospaces",
            "support": _support_for_pose(pose),
            "properties": {
                "molmospaces_object_name": name,
                "asset_path": episode.added_objects.get(name),
                "role": role,
                "description": description,
                "scene": episode.scene_key,
                "task_description": episode.language_description,
                "grasp": grasp,
            },
        }
        affordances = explicit_affordances_for(category, shape, role)
        if affordances:
            record["affordances"] = affordances
        return record


def infer_task_kind(task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "").lower()
    task_cls = str(task.get("task_cls") or "").lower()
    if "navtoobj" in task_cls or "nav" in task_type:
        return "navigate"
    if "dooropening" in task_cls or "door" in task_type:
        return "door_open"
    if "openclose" in task_cls or "open_close" in task_type or "close" in task_type:
        return "open_close"
    if "pickandplace" in task_cls or "pick_and_place" in task_type or task.get("place_receptacle_name"):
        return "pick_place"
    if "pick" in task_cls or "pick" in task_type or task.get("pickup_obj_name"):
        return "pick"
    return "approach"


def infer_object_category(name: str, description: str | None = None, asset_path: str | None = None) -> str:
    candidates = [description or "", name, asset_path or ""]
    for text in candidates:
        category = _category_from_text(text)
        if category is not None:
            return category
    return "object"


def infer_object_shape(category: str, description: str | None = None) -> dict[str, Any]:
    text = f"{category} {description or ''}".lower()
    if category in {"ball", "sphere", "apple", "orange", "tomato", "lemon", "lime", "peach", "fruit"}:
        return {"kind": "sphere", "radius": 0.045}
    if category in {"cup", "mug", "bottle", "can", "vase", "jar", "container"}:
        radius = 0.045 if category in {"cup", "mug", "can"} else 0.035
        height = 0.11 if category in {"cup", "mug", "can"} else 0.16
        return {"kind": "cylinder", "radius": radius, "size": [radius * 2.0, radius * 2.0, height]}
    if category in {"bowl", "plate"}:
        return {"kind": "cylinder", "radius": 0.07, "size": [0.14, 0.14, 0.045]}
    if category in {"knife", "fork", "spoon", "pencil", "pen", "remote", "phone"}:
        return {"kind": "box", "size": [0.16, 0.025, 0.018]}
    if category in {"cloth", "fabric", "towel", "napkin", "paper"} or "cloth" in text:
        return {"kind": "box", "size": [0.12, 0.10, 0.018]}
    if category in {"book", "laptop", "tablet"}:
        return {"kind": "box", "size": [0.22, 0.16, 0.025]}
    if category in {"box", "package", "cube", "pillow"}:
        return {"kind": "box", "size": [0.16, 0.12, 0.08]}
    return {"kind": "box", "size": [0.08, 0.08, 0.06]}


def explicit_affordances_for(category: str, shape: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role not in {"pickup", "navigation_target"}:
        return []
    kind = str(shape.get("kind") or "")
    if category in {"ball", "sphere", "apple", "orange", "tomato", "lemon", "lime", "peach", "fruit"}:
        radius = float(shape.get("radius") or 0.045)
        return [
            {
                "name": "single_hand_pinch",
                "score": 0.96,
                "source": "molmospaces",
                "params": {
                    "hand": "right",
                    "radius": radius,
                    "contact_model": "three_finger",
                    "reach_z": 0.0,
                },
            }
        ]
    if category in {"cup", "mug", "bottle", "can", "vase", "jar", "container", "bowl"} or kind == "cylinder":
        return [
            {
                "name": "side_grasp",
                "score": 0.92,
                "source": "molmospaces",
                "params": {
                    "hand": "right",
                    "radius": float(shape.get("radius") or 0.045),
                    "height": _shape_height(shape, 0.11),
                    "contact_model": "side_finger_wrap",
                    "reach_z": 0.02,
                },
            }
        ]
    if category in {"knife", "fork", "spoon", "pencil", "pen", "remote", "phone", "cloth", "fabric", "towel", "napkin"}:
        return [
            {
                "name": "top_grasp",
                "score": 0.9,
                "source": "molmospaces",
                "params": {
                    "hand": "right",
                    "aperture": 0.055,
                    "contact_model": "top_pinch",
                    "reach_z": 0.035,
                },
            }
        ]
    if _max_shape_extent(shape) <= 0.10:
        return [
            {
                "name": "single_hand_pinch",
                "score": 0.88,
                "source": "molmospaces",
                "params": {
                    "hand": "right",
                    "radius": max(0.025, _max_shape_extent(shape) * 0.5),
                    "contact_model": "three_finger",
                },
            }
        ]
    return []


def pose7_to_payload(pose: Iterable[float]) -> dict[str, Any]:
    x, y, z, qw, qx, qy, qz = _pose7(pose)
    return {
        "frame_id": "map",
        "position": [x, y, z],
        "yaw": quat_wxyz_yaw(qw, qx, qy, qz),
        "orientation_xyzw": [qx, qy, qz, qw],
    }


def pose7_to_base_payload(
    pose: Iterable[float],
    robot_pose: Iterable[float] | None,
) -> dict[str, Any] | None:
    if robot_pose is None:
        return None
    x, y, z, qw, qx, qy, qz = _pose7(pose)
    rx, ry, rz, rqw, rqx, rqy, rqz = _pose7(robot_pose)
    yaw = quat_wxyz_yaw(rqw, rqx, rqy, rqz)
    dx = x - rx
    dy = y - ry
    c = math.cos(yaw)
    s = math.sin(yaw)
    return {
        "frame_id": "base_link",
        "position": [c * dx + s * dy, -s * dx + c * dy, z - rz],
        "yaw": _wrap_pi(quat_wxyz_yaw(qw, qx, qy, qz) - yaw),
        "orientation_xyzw": [qx, qy, qz, qw],
    }


def pose7_yaw(pose: Iterable[float]) -> float:
    _, _, _, qw, qx, qy, qz = _pose7(pose)
    return quat_wxyz_yaw(qw, qx, qy, qz)


def quat_wxyz_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _raw_episode_records(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if path.is_file():
        return _records_from_json_file(path)
    if not path.exists():
        raise FileNotFoundError(f"MolmoSpaces benchmark path not found: {path}")
    benchmark_file = path / "benchmark.json"
    if benchmark_file.exists():
        return _records_from_json_file(benchmark_file)
    records: list[tuple[Path, dict[str, Any]]] = []
    for episode_file in sorted(path.glob("house_*/episode_*.json")):
        records.extend(_records_from_json_file(episode_file))
    if records:
        return records
    for episode_file in sorted(path.glob("episode_*.json")):
        records.extend(_records_from_json_file(episode_file))
    if records:
        return records
    raise FileNotFoundError(f"No MolmoSpaces benchmark episodes found under: {path}")


def _records_from_json_file(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [(path, item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        episodes = data.get("episodes")
        if isinstance(episodes, list):
            return [(path, item) for item in episodes if isinstance(item, dict)]
        if "task" in data:
            return [(path, data)]
    raise ValueError(f"Unsupported MolmoSpaces benchmark JSON shape: {path}")


def _episode_from_raw(index: int, source_path: Path, raw: dict[str, Any]) -> MolmoSpacesEpisode:
    task = _dict(raw.get("task"))
    robot = _dict(raw.get("robot"))
    scene_modifications = _dict(raw.get("scene_modifications"))
    language = _dict(raw.get("language"))
    object_poses: dict[str, tuple[float, float, float, float, float, float, float]] = {}
    for name, pose in _dict(scene_modifications.get("object_poses")).items():
        parsed = _pose7_optional(pose)
        if parsed is not None:
            object_poses[str(name)] = parsed
    return MolmoSpacesEpisode(
        index=index,
        source_path=source_path,
        house_index=_int_optional(raw.get("house_index")),
        scene_dataset=str(raw.get("scene_dataset") or "unknown"),
        data_split=str(raw.get("data_split") or "unknown"),
        robot_name=str(robot.get("robot_name") or "unknown"),
        robot_init_qpos=_qpos(robot.get("init_qpos")),
        robot_base_pose=_pose7_optional(task.get("robot_base_pose")),
        task_cls=str(task.get("task_cls") or ""),
        task_type=str(task.get("task_type")) if task.get("task_type") is not None else None,
        task=task,
        language_description=str(language.get("task_description") or ""),
        referral_expressions=_string_dict(language.get("referral_expressions")),
        scene_object_poses=object_poses,
        added_objects=_string_dict(scene_modifications.get("added_objects")),
        removed_objects=tuple(str(item) for item in _list(scene_modifications.get("removed_objects"))),
        cameras=tuple(_dict(camera) for camera in _list(raw.get("cameras"))),
        img_resolution=_resolution(raw.get("img_resolution")),
        raw=raw,
    )


def _category_from_text(text: str) -> str | None:
    normalized = _normalize_words(text)
    if not normalized:
        return None
    aliases = (
        ("spray bottle", "bottle"),
        ("water bottle", "bottle"),
        ("coffee mug", "mug"),
        ("cell phone", "phone"),
        ("mobile phone", "phone"),
        ("small cloth", "cloth"),
    )
    for phrase, category in aliases:
        if phrase in normalized:
            return category
    known = (
        "ball",
        "sphere",
        "apple",
        "orange",
        "tomato",
        "lemon",
        "lime",
        "peach",
        "fruit",
        "cup",
        "mug",
        "bottle",
        "can",
        "vase",
        "jar",
        "container",
        "bowl",
        "plate",
        "knife",
        "fork",
        "spoon",
        "pencil",
        "pen",
        "remote",
        "phone",
        "cloth",
        "fabric",
        "towel",
        "napkin",
        "paper",
        "book",
        "laptop",
        "tablet",
        "box",
        "package",
        "cube",
        "pillow",
    )
    words = normalized.split()
    for word in reversed(words):
        if word in known:
            return word
    name_match = re.match(r"([a-z][a-z0-9]*)[_/]", normalized.replace(" ", "_"))
    if name_match:
        prefix = name_match.group(1)
        if prefix not in {"obja", "object", "custom"}:
            return prefix
    return None


def _normalize_words(text: str) -> str:
    text = Path(text).stem if "/" in text else text
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).lower()
    words = [word for word in text.split() if not _looks_like_hash(word) and not word.isdigit()]
    return " ".join(words)


def _looks_like_hash(word: str) -> bool:
    return len(word) >= 10 and any(ch.isdigit() for ch in word) and all(ch in "0123456789abcdef" for ch in word)


def _referral_for(episode: MolmoSpacesEpisode, name: str) -> str | None:
    if name in episode.referral_expressions:
        return episode.referral_expressions[name]
    if name == episode.pickup_object_id:
        return episode.referral_expressions.get("pickup_obj_name") or episode.language_description
    if name == episode.place_receptacle_id:
        return episode.referral_expressions.get("place_receptacle_name")
    return None


def _grasp_hint(
    shape: dict[str, Any],
    category: str,
    pose: tuple[float, float, float, float, float, float, float],
    robot_pose: tuple[float, float, float, float, float, float, float] | None,
) -> dict[str, Any]:
    hint: dict[str, Any] = {
        "hand": "right",
        "reach_z": 0.0,
    }
    if robot_pose is not None:
        hint["base_target_map"] = [robot_pose[0], robot_pose[1], pose7_yaw(robot_pose)]
    if shape.get("kind") == "sphere":
        hint["radius"] = float(shape.get("radius") or 0.045)
        hint["approach_target_x"] = 0.54
    elif shape.get("kind") == "cylinder":
        hint["radius"] = float(shape.get("radius") or 0.045)
        hint["height"] = _shape_height(shape, 0.11)
        hint["approach_target_x"] = 0.52
    elif category in {"box", "package", "cube", "pillow"}:
        hint["approach_target_x"] = 0.48
        hint["open_y"] = 0.26
        hint["clamp_y"] = 0.12
    else:
        hint["approach_target_x"] = 0.52
    return hint


def _support_for_pose(pose: tuple[float, float, float, float, float, float, float]) -> str:
    return "floor" if pose[2] < 0.18 else "scene_support"


def _navigation_goal_id(episode: MolmoSpacesEpisode) -> str:
    return f"{episode.episode_id}_nav_goal"


def _place_target_id(place_id: str | None) -> str | None:
    if place_id is None:
        return None
    return f"{place_id}_place_target"


def _normalize_task_kind(task_kind: str) -> str:
    return task_kind.strip().lower().replace("-", "_").replace(" ", "_")


def _pose7(value: Iterable[float]) -> tuple[float, float, float, float, float, float, float]:
    values = list(value)
    if len(values) < 7:
        raise ValueError(f"expected pose [x,y,z,qw,qx,qy,qz], got {value!r}")
    return tuple(float(values[index]) for index in range(7))  # type: ignore[return-value]


def _pose7_optional(value: Any) -> tuple[float, float, float, float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return None
    try:
        return _pose7(value)
    except (TypeError, ValueError):
        return None


def _qpos(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, raw_items in value.items():
        if not isinstance(raw_items, list):
            continue
        out[str(key)] = [float(item) for item in raw_items]
    return out


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _resolution(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (int(value[0]), int(value[1]))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_first(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _int_optional(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _distance_xy(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.hypot(float(av[0]) - float(bv[0]), float(av[1]) - float(bv[1]))


def _shape_height(shape: dict[str, Any], default: float) -> float:
    size = shape.get("size")
    if isinstance(size, list) and len(size) >= 3:
        return float(size[2])
    return default


def _max_shape_extent(shape: dict[str, Any]) -> float:
    if shape.get("radius") is not None:
        return float(shape["radius"]) * 2.0
    size = shape.get("size")
    if isinstance(size, list) and size:
        return max(float(item) for item in size)
    return 0.0


def _wrap_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
