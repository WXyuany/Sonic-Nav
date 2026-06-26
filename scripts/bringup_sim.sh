#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Sonic-Nav Bringup ==="
echo "Repo: $REPO_ROOT"

source "$REPO_ROOT/.venv_sim/bin/activate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/g1_ros2_nav:$PYTHONPATH"
export DISPLAY="${DISPLAY:-:1}"

echo "[1/2] Starting MuJoCo simulator + ROS2 bridge..."
python "$REPO_ROOT/g1_ros2_nav/scripts/run_bridge.py" \
    --env-name sim \
    --enable_offscreen False \
    &
SIM_PID=$!

sleep 8

echo "[2/2] Starting C++ deployment..."
source "$REPO_ROOT/gear_sonic_deploy/scripts/setup_env.sh" > /dev/null 2>&1

"$REPO_ROOT/gear_sonic_deploy/target/release/g1_deploy_onnx_ref" lo \
    policy/release/model_decoder.onnx \
    reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type ros2 \
    --output-type all \
    --zmq-host localhost \
    --disable-crc-check &
DEPLOY_PID=$!

echo ""
echo "=== Bringup Complete ==="
echo "  Simulator PID:  $SIM_PID"
echo "  Deploy PID:     $DEPLOY_PID"
echo ""
echo "  Next steps:"
echo "    1. Press ']' in deploy terminal to start control"
echo "    2. ros2 launch g1_ros2_nav bringup.launch.py"
echo "    3. rviz2"
echo ""
echo "  Press Ctrl+C to stop all"

trap "kill $SIM_PID $DEPLOY_PID 2>/dev/null; echo 'Stopped'" EXIT
wait
