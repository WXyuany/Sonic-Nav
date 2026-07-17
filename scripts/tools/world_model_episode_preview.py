#!/usr/bin/env python3
"""Render a MuJoCo carry-state episode scene to a portable RGB preview."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the physical initial state of an episode scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/episodes/episode_preview.png")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--qpos", help="Optional live/final MuJoCo qpos snapshot to render.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import json

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(Path(args.scene).expanduser()))
    data = mujoco.MjData(model)
    if args.qpos:
        qpos = np.load(Path(args.qpos).expanduser(), allow_pickle=False)
        data.qpos[: min(len(qpos), model.nq)] = qpos[: model.nq]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=max(64, args.height), width=max(64, args.width))
    try:
        renderer.update_scene(data, camera=args.camera)
        image = Image.fromarray(renderer.render())
    finally:
        renderer.close()
    draw = ImageDraw.Draw(image)
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), list) else []
    state = "physical qpos snapshot" if args.qpos else "physical initial state"
    title = f"{manifest.get('sequence_id', 'episode')} | {state} | {len(stages)} stages"
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
