#!/bin/bash
# Usage: bash scripts/switch_scene.sh <scene_name>
# Run without an argument to switch to the default scene.

set -euo pipefail

SCENE=${1:-default}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${SCENE}" == "--list" || "${SCENE}" == "list" ]]; then
  python3 -m gear_sonic.utils.mujoco_sim.scene_registry --list
  exit 0
fi

python3 -m gear_sonic.utils.mujoco_sim.scene_registry --switch "$SCENE"
echo "Restart sim to apply."
