#!/usr/bin/env bash
# Bootstrap the reproducible, user-space portion of Sonic-Nav.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="python3"
WITH_DATA=0
WITH_RL=0
WITH_QWEN=0
CHECK_ROS=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [options]

Options:
  --data          Initialize Git LFS and the external_data submodule.
  --rl            Install RSL-RL training dependencies.
  --qwen          Create .venv_qwen_vl and install the local Qwen-VL service dependencies.
  --ros           Verify that ROS 2 Humble is installed at /opt/ros/humble.
  --python PATH   Python interpreter used for the framework virtual environment.
  --venv PATH     Framework virtual environment location (default: .venv).
  -h, --help      Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) WITH_DATA=1 ;;
    --rl) WITH_RL=1 ;;
    --qwen) WITH_QWEN=1 ;;
    --ros) CHECK_ROS=1 ;;
    --python) PYTHON="$2"; shift ;;
    --venv) VENV="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
cd "$ROOT"

if [[ $CHECK_ROS -eq 1 ]]; then
  [[ -f /opt/ros/humble/setup.bash ]] || {
    echo "ROS 2 Humble was not found. Install it on the host, then rerun with --ros." >&2
    exit 1
  }
  echo "ROS 2 Humble: /opt/ros/humble/setup.bash"
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel
"$VENV/bin/python" -m pip install numpy pyyaml mujoco

if [[ $WITH_RL -eq 1 ]]; then
  "$VENV/bin/python" -m pip install torch
  "$VENV/bin/python" -m pip install -r requirements-rl.txt
fi

if [[ $WITH_DATA -eq 1 ]]; then
  command -v git-lfs >/dev/null || { echo "git-lfs is required for --data" >&2; exit 1; }
  git lfs install
  git submodule update --init --recursive
fi

if [[ $WITH_QWEN -eq 1 ]]; then
  "$PYTHON" -m venv "$ROOT/.venv_qwen_vl"
  "$ROOT/.venv_qwen_vl/bin/python" -m pip install --upgrade pip wheel
  "$ROOT/.venv_qwen_vl/bin/python" -m pip install -r requirements-qwen-vl.txt
fi

echo "Bootstrap complete. Activate with: source $VENV/bin/activate"
echo "Validate with: PYTHONPATH=scripts $VENV/bin/python -m unittest discover -s scripts/sonic_world/tests -q"
