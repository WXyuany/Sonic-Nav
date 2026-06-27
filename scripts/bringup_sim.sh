#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Sonic-Nav: Sim + Deploy + Bridge ==="
echo "IMPORTANT: Set these env vars in ALL terminals first:"
echo "  export ROS_DOMAIN_ID=42"
echo "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
echo "  export ROS_LOCALHOST_ONLY=1"
echo ""

source "$REPO_ROOT/.venv_sim/bin/activate"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/g1_ros2_nav:$PYTHONPATH"
export DISPLAY="${DISPLAY:-:1}"
export ROS_DOMAIN_ID=42

echo "[1/2] MuJoCo simulator..."
python "$REPO_ROOT/gear_sonic/scripts/run_sim_loop.py" &
SIM_PID=$!
sleep 10

echo "[2/2] C++ deploy (ROS2 input, FastRTPS)..."
cd "$REPO_ROOT/gear_sonic_deploy"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1
source scripts/setup_env.sh > /dev/null 2>&1
./target/release/g1_deploy_onnx_ref lo \
    policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type ros2 --output-type all --zmq-host localhost \
    --disable-crc-check &
DEPLOY_PID=$!

echo ""
echo "=== Ready ==="
echo "  Sim PID:    $SIM_PID"
echo "  Deploy PID: $DEPLOY_PID"
echo ""
echo "  Next terminal:"
echo "    export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1"
echo "    source /opt/ros/humble/setup.bash"
echo "    source ~/ros2_ws/install/setup.bash"
echo "    ros2 launch g1_ros2_nav bringup.launch.py"
echo ""
trap "kill $SIM_PID $DEPLOY_PID 2>/dev/null" EXIT
wait
