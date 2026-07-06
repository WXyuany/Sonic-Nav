#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gear_sonic.nav.params import REPO_ROOT, load_config
from gear_sonic.utils.mujoco_sim.scene_registry import resolve_scene, set_wbc_scene


DEFAULTS = {
    "output_dir": "gear_sonic/data/robot_model/model_data/g1",
    "obstacle_count": 8,
    "area": {
        "x": [-5.0, 5.0],
        "y": [-4.0, 4.0],
    },
    "size": {
        "x": [0.12, 0.38],
        "y": [0.12, 0.45],
        "z": [0.25, 0.85],
    },
    "min_clearance": {
        "robot_origin": 1.2,
        "obstacle_obstacle": 0.45,
    },
    "rgba": [0.45, 0.50, 0.58, 1.0],
}


def _range(cfg: dict, section: str, axis: str) -> tuple[float, float]:
    values = cfg[section][axis]
    return (float(values[0]), float(values[1]))


def _fmt(values) -> str:
    return " ".join(f"{float(v):.4g}" for v in values)


def _resolve_output(path: str | None, output_dir: str | None, source: Path, seed: int) -> Path:
    if path:
        out = Path(path).expanduser()
        if not out.is_absolute():
            out = REPO_ROOT / out
        return out

    if output_dir:
        out_dir = Path(output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = source.parent
    return out_dir / f"{source.stem}_rand_{seed}.xml"


def _remove_existing(worldbody: ET.Element, prefix: str) -> int:
    removed = 0
    for child in list(worldbody):
        name = child.get("name", "")
        if name.startswith(prefix):
            worldbody.remove(child)
            removed += 1
    return removed


def _sample_obstacles(cfg: dict, count: int, seed: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    x_range = _range(cfg, "area", "x")
    y_range = _range(cfg, "area", "y")
    sx_range = _range(cfg, "size", "x")
    sy_range = _range(cfg, "size", "y")
    sz_range = _range(cfg, "size", "z")
    origin_clear = float(cfg["min_clearance"]["robot_origin"])
    obstacle_clear = float(cfg["min_clearance"]["obstacle_obstacle"])
    rgba = [float(v) for v in cfg["rgba"]]

    obstacles: list[dict] = []
    attempts = 0
    max_attempts = max(200, count * 200)
    while len(obstacles) < count and attempts < max_attempts:
        attempts += 1
        sx = rng.uniform(*sx_range)
        sy = rng.uniform(*sy_range)
        sz = rng.uniform(*sz_range)
        x = rng.uniform(x_range[0] + sx, x_range[1] - sx)
        y = rng.uniform(y_range[0] + sy, y_range[1] - sy)
        radius = math.hypot(sx, sy)
        if math.hypot(x, y) < origin_clear + radius:
            continue
        ok = True
        for obs in obstacles:
            other_radius = math.hypot(obs["size"][0], obs["size"][1])
            if math.hypot(x - obs["pos"][0], y - obs["pos"][1]) < obstacle_clear + radius + other_radius:
                ok = False
                break
        if not ok:
            continue
        idx = len(obstacles)
        obstacles.append({
            "name": f"{prefix}{idx:02d}",
            "pos": (x, y, sz),
            "size": (sx, sy, sz),
            "yaw": rng.uniform(-math.pi, math.pi),
            "rgba": rgba,
        })
    if len(obstacles) < count:
        raise RuntimeError(f"Only placed {len(obstacles)} of {count} obstacles after {attempts} attempts")
    return obstacles


def randomize_scene(
    scene: str,
    *,
    seed: int,
    count: int | None,
    output: str | None,
    output_dir: str | None,
    prefix: str,
    dry_run: bool,
    switch: bool,
) -> Path:
    cfg = load_config("scene_randomization", DEFAULTS, "SONIC_SCENE_RANDOM_CONFIG")
    selection = resolve_scene(scene)
    source = selection.abs_path
    obstacle_count = int(count if count is not None else cfg["obstacle_count"])
    out_path = _resolve_output(output, output_dir or cfg.get("output_dir"), source, seed)

    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in {source}")

    removed = _remove_existing(worldbody, prefix)
    obstacles = _sample_obstacles(cfg, obstacle_count, seed, prefix)
    worldbody.append(ET.Comment(f" randomized navigation obstacles seed={seed} count={obstacle_count} "))
    for obs in obstacles:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obs["name"],
                "pos": _fmt(obs["pos"]),
                "euler": _fmt((0.0, 0.0, obs["yaw"])),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obs['name']}_geom",
                "type": "box",
                "size": _fmt(obs["size"]),
                "rgba": _fmt(obs["rgba"]),
                "friction": "1.0 0.02 0.001",
            },
        )

    if dry_run:
        print(f"Source: {source}")
        print(f"Output: {out_path}")
        print(f"Removed existing generated bodies: {removed}")
        for obs in obstacles:
            print(f"  {obs['name']}: pos={_fmt(obs['pos'])} size={_fmt(obs['size'])}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    if switch:
        set_wbc_scene(out_path)
    print(f"Wrote randomized scene: {out_path}")
    if switch:
        try:
            display_path = out_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = out_path
        print(f"Switched WBC scene to: {display_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate randomized MuJoCo navigation scenes.")
    parser.add_argument("scene", nargs="?", default="robocasa_apartment", help="scene name, alias, XML file, or XML path")
    parser.add_argument("--seed", type=int, default=None, help="random seed; defaults to current time")
    parser.add_argument("--count", type=int, default=None, help="number of random box obstacles")
    parser.add_argument("--output", default=None, help="output XML path")
    parser.add_argument("--output-dir", default=None, help="directory for generated XML when --output is omitted")
    parser.add_argument("--prefix", default="rand_obstacle_", help="name prefix for generated obstacle bodies")
    parser.add_argument("--dry-run", action="store_true", help="print placements without writing XML")
    parser.add_argument("--switch", action="store_true", help="write generated XML into the WBC scene config")
    args = parser.parse_args()

    seed = int(args.seed if args.seed is not None else time.time())
    randomize_scene(
        args.scene,
        seed=seed,
        count=args.count,
        output=args.output,
        output_dir=args.output_dir,
        prefix=args.prefix,
        dry_run=args.dry_run,
        switch=args.switch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
