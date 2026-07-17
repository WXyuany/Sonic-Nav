#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MolmoSpaces episode proxy scene and launch the Sonic GUI stack."
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="external_dependencies/molmospaces-src/mlspaces_tests/data_generation/test_benchmark/benchmark.json",
        help="Path to a MolmoSpaces benchmark.json file or benchmark directory.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--scene-mode",
        choices=["real", "proxy"],
        default="real",
        help="real uses MolmoSpaces MJCF/assets; proxy uses a fast local tabletop stand-in.",
    )
    parser.add_argument("--output")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--max-context-objects", type=int, default=10)
    parser.add_argument("--object-x", type=float, default=1.34)
    parser.add_argument("--surface-height", type=float, default=0.79)
    parser.add_argument("--raw-local-frame", action="store_true")
    parser.add_argument("--asset-variant", choices=["base", "ceiling"], default="base")
    parser.add_argument("--install-assets", dest="install_assets", action="store_true", default=True)
    parser.add_argument("--no-install-assets", dest="install_assets", action="store_false")
    parser.add_argument("--install-object-assets", dest="install_object_assets", action="store_true", default=True)
    parser.add_argument("--no-install-object-assets", dest="install_object_assets", action="store_false")
    parser.add_argument("--real-collisions", action="store_true")
    parser.add_argument("--real-z-shift", type=float, default=0.0)
    parser.add_argument("--real-z-align", dest="real_z_align", action="store_true")
    parser.add_argument("--no-real-z-align", dest="real_z_align", action="store_false")
    parser.set_defaults(real_z_align=False)
    parser.add_argument("--show-markers", action="store_true")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--no-launch", action="store_true", help="Only generate files and print the launch command.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_for_builder = sys.executable
    builder_cmd = [
        python_for_builder,
        str(SCRIPT_DIR / "tools" / "molmospaces_scene_builder.py"),
        args.benchmark,
        "--episode-index",
        str(args.episode_index),
        "--scene-mode",
        args.scene_mode,
        "--max-context-objects",
        str(args.max_context_objects),
        "--object-x",
        str(args.object_x),
        "--surface-height",
        str(args.surface_height),
    ]
    if args.output:
        builder_cmd += ["--output", args.output]
    if args.no_context:
        builder_cmd += ["--no-context"]
    if args.raw_local_frame:
        builder_cmd += ["--raw-local-frame"]
    if args.asset_variant:
        builder_cmd += ["--asset-variant", args.asset_variant]
    if not args.install_assets:
        builder_cmd += ["--no-install-assets"]
    if not args.install_object_assets:
        builder_cmd += ["--no-install-object-assets"]
    if args.real_collisions:
        builder_cmd += ["--real-collisions"]
    if args.real_z_shift:
        builder_cmd += ["--real-z-shift", str(args.real_z_shift)]
    if args.real_z_align:
        builder_cmd += ["--real-z-align"]
    if args.show_markers:
        builder_cmd += ["--show-markers"]
    if args.validate:
        builder_cmd += ["--validate"]

    result = subprocess.run(builder_cmd, cwd=REPO, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    xml_path = _extract_xml_path(result.stdout)
    if xml_path is None:
        print("Could not find generated XML path in molmospaces_scene_builder output.", file=sys.stderr)
        return 1
    rel_xml = xml_path.relative_to(REPO) if xml_path.is_absolute() and xml_path.is_relative_to(REPO) else xml_path
    if args.no_launch:
        print(f"Launch GUI with: python scripts/start.py {rel_xml}")
        return 0
    os.execv(sys.executable, [sys.executable, str(SCRIPT_DIR / "start.py"), str(rel_xml)])
    return 0


def _extract_xml_path(text: str) -> Path | None:
    prefix = "Wrote MolmoSpaces MuJoCo scene:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return Path(line.removeprefix(prefix).strip()).resolve()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
