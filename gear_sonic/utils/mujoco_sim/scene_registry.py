"""Shared registry for MuJoCo navigation scenes.

The launch scripts, sensor publishers, and helper tools all need to agree on
which XML is active.  Keeping that mapping here avoids per-script scene lists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
G1_MODEL_DIR_REL = Path("gear_sonic/data/robot_model/model_data/g1")
WBC_CONFIG_REL = Path("gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml")


@dataclass(frozen=True)
class SceneSpec:
    name: str
    xml_file: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneSelection:
    name: str
    xml_file: str
    description: str
    abs_path: Path
    rel_path: Path

    @property
    def rel_path_posix(self) -> str:
        return self.rel_path.as_posix()


SCENES: tuple[SceneSpec, ...] = (
    SceneSpec("default", "scene_43dof.xml", "8m x 8m room with cylinder obstacles"),
    SceneSpec("box_demo", "scene_box_demo.xml", "open room with a visible low box for the forearm clamp demo", aliases=("box",)),
    SceneSpec(
        "ball_demo",
        "scene_ball_demo.xml",
        "tabletop pick-and-place scene with a light ball target",
        aliases=("ball",),
    ),
    SceneSpec("dynamic", "scene_dynamic.xml", "moving rails and rotating obstacle"),
    SceneSpec("stairs", "scene_stairs.xml", "stairs and ramp locomotion stress test"),
    SceneSpec("uneven", "scene_uneven.xml", "bumpy terrain and low rocks"),
    SceneSpec("table", "scene_table.xml", "tabletop interaction area"),
    SceneSpec("indoor", "scene_indoor.xml", "open apartment-scale indoor navigation floor"),
    SceneSpec(
        "robocasa_kitchen",
        "scene_robocasa_kitchen.xml",
        "large RoboCasa-style kitchen and living room",
        aliases=("kitchen",),
    ),
    SceneSpec(
        "robocasa_galley",
        "scene_robocasa_galley.xml",
        "narrow RoboCasa galley kitchen with tight navigation lanes",
        aliases=("galley",),
    ),
    SceneSpec(
        "robocasa_apartment",
        "scene_robocasa_apartment.xml",
        "multi-zone apartment with partial walls and doorways",
        aliases=("apartment", "apt"),
    ),
    SceneSpec(
        "robocasa_cafe",
        "scene_robocasa_cafe.xml",
        "small cafe scene with counter, stools, tables, and clutter",
        aliases=("cafe",),
    ),
)


def _scene_map() -> dict[str, SceneSpec]:
    mapping: dict[str, SceneSpec] = {}
    for spec in SCENES:
        mapping[spec.name] = spec
        mapping[spec.xml_file] = spec
        mapping[Path(spec.xml_file).stem] = spec
        for alias in spec.aliases:
            mapping[alias] = spec
    return mapping


def scene_names() -> list[str]:
    return [spec.name for spec in SCENES]


def scene_help() -> str:
    return "\n".join(f"  {spec.name:<20} {spec.xml_file:<34} {spec.description}" for spec in SCENES)


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root).expanduser().resolve() if repo_root is not None else REPO_ROOT


def resolve_scene(scene_arg: str | Path | None = None, repo_root: str | Path | None = None) -> SceneSelection:
    """Resolve a scene name, alias, XML filename, or path into a scene selection."""

    root = _repo_root(repo_root)
    model_dir = root / G1_MODEL_DIR_REL
    raw = "default" if scene_arg is None else str(scene_arg)
    key = raw.strip()
    if not key:
        key = "default"

    spec = _scene_map().get(key)
    if spec is not None:
        rel_path = G1_MODEL_DIR_REL / spec.xml_file
        return SceneSelection(spec.name, spec.xml_file, spec.description, root / rel_path, rel_path)

    candidate = Path(key).expanduser()
    if not candidate.is_absolute():
        if candidate.suffix == ".xml" and len(candidate.parts) == 1:
            candidate = model_dir / candidate
        else:
            candidate = root / candidate

    candidate = candidate.resolve()
    if candidate.exists() and candidate.suffix == ".xml":
        try:
            rel_path = candidate.relative_to(root)
        except ValueError:
            rel_path = candidate
        return SceneSelection(candidate.stem, candidate.name, "custom MuJoCo scene", candidate, rel_path)

    valid = "|".join(scene_names())
    raise ValueError(f"Unknown scene '{key}'. Valid scenes: {valid}")


def set_wbc_scene(
    scene_arg: str | Path | None = None,
    repo_root: str | Path | None = None,
    wbc_config_path: str | Path | None = None,
) -> SceneSelection:
    """Write the resolved scene into the WBC YAML while preserving the file layout."""

    selection = resolve_scene(scene_arg, repo_root=repo_root)
    root = _repo_root(repo_root)
    yaml_path = Path(wbc_config_path).expanduser() if wbc_config_path else root / WBC_CONFIG_REL
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path

    text = yaml_path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r'^(ROBOT_SCENE:\s*")[^"]*(".*)$',
        lambda match: f"{match.group(1)}{selection.rel_path_posix}{match.group(2)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements == 0:
        raise ValueError(f"ROBOT_SCENE not found in {yaml_path}")
    if updated != text:
        yaml_path.write_text(updated, encoding="utf-8")
    return selection


def iter_missing_scene_files(repo_root: str | Path | None = None) -> Iterable[Path]:
    root = _repo_root(repo_root)
    for spec in SCENES:
        path = root / G1_MODEL_DIR_REL / spec.xml_file
        if not path.exists():
            yield path


def _main() -> int:
    parser = argparse.ArgumentParser(description="List or switch Sonic MuJoCo scenes.")
    parser.add_argument("scene", nargs="?", default=None, help="scene name, alias, XML file, or XML path")
    parser.add_argument("--list", action="store_true", help="list registered scenes")
    parser.add_argument("--check", action="store_true", help="check that all registered XML files exist")
    parser.add_argument("--switch", action="store_true", help="write the selected scene into the WBC YAML")
    args = parser.parse_args()

    if args.list:
        print(scene_help())
        return 0

    if args.check:
        missing = list(iter_missing_scene_files())
        if missing:
            print("Missing scene files:")
            for path in missing:
                print(f"  {path}")
            return 1
        print("All registered scene files exist.")
        return 0

    if args.switch or args.scene is not None:
        selection = set_wbc_scene(args.scene)
        print(f"Switched to: {selection.name} ({selection.xml_file})")
        print(f"ROBOT_SCENE: {selection.rel_path_posix}")
        return 0

    print(scene_help())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
