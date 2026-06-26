#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Sonic-Nav Sim + Bridge ==="

source "$REPO_ROOT/.venv_sim/bin/activate"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/g1_ros2_nav:$PYTHONPATH"
export DISPLAY="${DISPLAY:-:1}"

echo "[1/2] MuJoCo sim + ROS2 bridge..."
python "$REPO_ROOT/g1_ros2_nav/scripts/run_bridge.py" \
    --env-name sim --enable_offscreen False &
SIM_PID=$!
sleep 12

echo "[2/2] C++ deploy (ROS2 input)..."
cd "$REPO_ROOT/gear_sonic_deploy"
source scripts/setup_env.sh > /dev/null 2>&1
./target/release/g1_deploy_onnx_ref lo \
    policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type ros2 --output-type all --zmq-host localhost \
    --disable-crc-check &
DEPLOY_PID=$!

echo "PIDs: sim=$SIM_PID deploy=$DEPLOY_PID"
echo "Next: ros2 launch g1_ros2_nav bringup.launch.py"
trap "kill $SIM_PID $DEPLOY_PID 2>/dev/null" EXIT
wait
