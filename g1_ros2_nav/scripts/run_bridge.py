#!/usr/bin/env python3
"""Launch the MuJoCo simulator with ROS2 sensor bridge running alongside."""

import sys
import os
import threading
import time

import rclpy
import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from g1_ros2_nav.g1_bridge import G1ROSBridge


class BridgeRunner:
    def __init__(self, sim_env, bridge_node):
        self._sim_env = sim_env
        self._bridge_node = bridge_node
        self._running = True

    def run(self):
        while self._running and rclpy.ok():
            rclpy.spin_once(self._bridge_node, timeout_sec=0.01)

    def stop(self):
        self._running = False


def main():
    config = tyro.cli(SimLoopConfig)
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name

    robot_model = instantiate_g1_robot_model()
    init_channel(config=wbc_config)

    sim = SimulatorFactory.create_simulator(
        config=wbc_config,
        env_name=config.env_name,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )

    rclpy.init(args=sys.argv)
    bridge = G1ROSBridge(sim_env=sim.sim_env)

    runner = BridgeRunner(sim.sim_env, bridge)
    bridge_thread = threading.Thread(target=runner.run, daemon=True)
    bridge_thread.start()

    SimulatorFactory.start_simulator(
        sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )

    runner.stop()
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
